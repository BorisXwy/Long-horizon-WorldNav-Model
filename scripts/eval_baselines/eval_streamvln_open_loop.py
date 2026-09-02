#!/usr/bin/env python3
"""Evaluate the official StreamVLN checkpoint on fixed R2R trajectories.

This is an open-loop action test: RGB observations are taken from the
pre-collected StreamVLN R2R trajectory, while the model's predicted actions are
never executed in Habitat.  At each four-action decision point the evaluator
uses the same streaming cache and 32-frame slow-memory reset as the official
evaluator, then compares the generated action sequence with the annotated
shortest-path actions.

The reported ``action_accuracy`` and ``chunk_exact_match`` are therefore not
the paper's closed-loop Habitat SR.  They measure how often the model predicts
the fixed trajectory's next action(s) when given ground-truth observations.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
STREAMVLN_ROOT = Path(os.environ.get("STREAMVLN_ROOT", str(ROOT.parent / "StreamVLN")))
sys.path.insert(0, str(STREAMVLN_ROOT))
sys.path.insert(0, str(STREAMVLN_ROOT / "streamvln"))

from model.stream_video_vln import StreamVLNForCausalLM  # noqa: E402
from utils.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_MEMORY_TOKEN, DEFAULT_VIDEO_TOKEN  # noqa: E402


ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
ACTION_TO_TOKEN = {"STOP": [0], "↑": [1], "←": [2], "→": [3]}
ACTION_PATTERNS = re.compile("|".join(re.escape(k) for k in ACTION_TO_TOKEN))
CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]


def parse_actions(text: str) -> list[int]:
    values: list[int] = []
    for match in ACTION_PATTERNS.findall(text):
        values.extend(ACTION_TO_TOKEN[match])
    return values


def load_tokenizer(model_path: Path, model_max_length: int) -> transformers.PreTrainedTokenizer:
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model_path, model_max_length=model_max_length, padding_side="right"
        )
    except Exception as exc:
        print(f"[StreamVLN] fast tokenizer failed ({exc}); using slow tokenizer", flush=True)
        return transformers.AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )


def preprocess_qwen(
    sources: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    *,
    add_system: bool,
    system_message: str = "You are a helpful assistant.",
) -> torch.Tensor:
    """Match StreamVLN's prompt/token construction used by streamvln_eval.py."""

    roles = {"human": "user", "gpt": "assistant"}
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.add_tokens(["<memory>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    tokenizer.chat_template = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
        "message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )

    input_ids: list[list[int]] = []
    for source in sources:
        source = copy.deepcopy(source)
        # This is intentionally conditional: it follows the upstream
        # ``streamvln_eval.py`` behavior, where the first turn receives an
        # image token and later turns extend the cached text context.
        prompt = random.choice(CONJUNCTIONS) + DEFAULT_IMAGE_TOKEN
        if source[0].get("value", ""):
            source[0]["value"] += f" {prompt}."
        if roles.get(source[0]["from"], source[0]["from"]) != "user":
            source = source[1:]
        ids: list[int] = []
        if add_system:
            ids += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        for conv in source:
            role = roles.get(conv.get("from", conv.get("role")), conv.get("from", conv.get("role")))
            content = conv.get("value", conv.get("content", ""))
            ids += tokenizer.apply_chat_template([{"role": role, "content": content}])
        ids = [image_token_index if value == image_token_index else value for value in ids]
        ids = [memory_token_index if value == memory_token_index else value for value in ids]
        input_ids.append(ids)
    # These are always one-sample prompts in the official streaming evaluator.
    del im_start, im_end
    return torch.tensor(input_ids, dtype=torch.long)


def make_prompt(instruction: str, *, first_call: bool, with_memory: bool) -> list[dict[str, str]]:
    if first_call:
        value = (
            "<video>\nYou are an autonomous navigation assistant. Your task is to "
            "<instruction>. Devise an action sequence to follow the instruction using the "
            "four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD "
            "(↑) by 25 centimeters, or STOP."
        )
        value = value.replace(
            " Where should you go next to stay on track?",
            " Please devise an action sequence to follow the instruction which may include "
            "turning left or right by a certain degree, moving forward by a certain distance "
            "or stopping once the task is complete.",
        )
        value = value.replace("<instruction>.", instruction)
        if with_memory:
            value += f" These are your historical observations {DEFAULT_MEMORY_TOKEN}."
        value = value.replace(DEFAULT_VIDEO_TOKEN + "\n", "")
        return [{"from": "human", "value": value}, {"from": "gpt", "value": ""}]
    return [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]


def encode_image(processor: Any, path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]


def build_model(model_path: Path, device: torch.device, dtype: torch.dtype, model_max_length: int):
    tokenizer = load_tokenizer(model_path, model_max_length)
    config = transformers.AutoConfig.from_pretrained(model_path)
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "eager"
    print(f"[StreamVLN] attention implementation: {attn_impl}", flush=True)
    model = StreamVLNForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
        config=config,
        low_cpu_mem_usage=False,
    )
    model.model.num_history = 8
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    model.reset(1)
    return model, tokenizer


def target_actions(raw_actions: list[int], num_frames: int) -> list[int]:
    raw = [int(value) for value in raw_actions]
    if raw and raw[0] == -1:
        raw = raw[1:]
    # StreamVLN's trajectory annotations omit the terminal STOP.  The official
    # training/eval data convention appends it and absorbs it thereafter.
    out = raw[:num_frames]
    if len(out) < num_frames:
        out += [0] * (num_frames - len(out))
    return out


@torch.no_grad()
def evaluate_episode(
    *,
    model: StreamVLNForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    processor: Any,
    item: dict[str, Any],
    data_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    num_frames: int,
    num_future_steps: int,
    num_history: int,
    max_decisions: int,
    save_outputs: bool,
) -> dict[str, Any]:
    video_dir = data_root / item["video"]
    rgb_dir = video_dir / "rgb"
    frame_paths = sorted(rgb_dir.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(rgb_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(rgb_dir)
    # The archive has one image for each action observation.  Loading the
    # trajectory once allows the subsequent streaming calls to reuse tensors.
    rgb = [encode_image(processor, path) for path in frame_paths]
    gt = target_actions(item["actions"], len(rgb))
    instruction = item.get("instructions", [""])
    if isinstance(instruction, list):
        instruction = instruction[0] if instruction else ""

    model.reset_for_env(0)
    output_ids = None
    past_key_values = None
    time_ids: list[int] = []
    predicted_actions: list[int] = []
    decision_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    decisions = list(range(0, len(rgb), num_future_steps))
    if max_decisions > 0:
        decisions = decisions[:max_decisions]

    for decision_idx, step_id in enumerate(decisions):
        # Official StreamVLN resets its fast KV cache every num_frames steps;
        # observations from the prior window are then supplied as slow memory.
        if step_id > 0 and step_id % num_frames == 0:
            model.reset_for_env(0)
            output_ids = None
            past_key_values = None
            time_ids = []
        previous = time_ids[-1] + 1 if time_ids else step_id
        if not time_ids:
            time_ids.append(step_id)
        else:
            time_ids.extend(range(previous, step_id + 1))

        first_call = output_ids is None
        sources = make_prompt(
            str(instruction), first_call=first_call, with_memory=step_id != 0
        )
        input_ids = preprocess_qwen([sources], tokenizer, add_system=first_call).to(device)
        if output_ids is not None:
            input_ids = torch.cat([output_ids, input_ids], dim=1)

        selected_images = [rgb[step_id]]
        if step_id != 0 and step_id % num_frames == 0:
            stride = max(step_id // num_history, 1)
            history_ids = list(range(0, step_id, stride))
            selected_images = [rgb[index] for index in history_ids] + selected_images
        images = torch.stack(selected_images).unsqueeze(0).to(device=device, dtype=dtype)
        views = images.shape[1]
        depths = torch.zeros((1, views, 1, 1), device=device, dtype=dtype)
        poses = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).expand(1, views, 4, 4)
        intrinsics = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).expand(1, views, 4, 4)

        outputs = model.generate(
            inputs=input_ids,
            images=images,
            depths=depths,
            poses=poses,
            intrinsics=intrinsics,
            env_id=0,
            time_ids=[time_ids],
            task_type=[0],
            do_sample=False,
            num_beams=1,
            max_new_tokens=64,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
        )
        output_ids = outputs.sequences
        past_key_values = outputs.past_key_values
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
        parsed = parse_actions(decoded)
        if not parsed:
            parsed = [0]
        parsed = parsed[:num_future_steps]
        target = gt[step_id : min(step_id + num_future_steps, len(gt))]
        valid = len(target)
        pred = parsed[:valid] + [0] * max(0, valid - len(parsed))
        predicted_actions.extend(pred)
        decision_rows.append(
            {
                "step": step_id,
                "gt": [ACTION_NAMES.get(value, str(value)) for value in target],
                "pred": [ACTION_NAMES.get(value, str(value)) for value in pred],
                "decoded": decoded,
                "exact": bool(pred == target),
            }
        )

    # Use the numeric stream for an unambiguous count; JSON rows retain readable
    # action names for qualitative inspection.
    numeric_gt = [value for step in decisions for value in gt[step : min(step + num_future_steps, len(gt))]]
    numeric_pred = predicted_actions[: len(numeric_gt)]
    correct = sum(int(a == b) for a, b in zip(numeric_pred, numeric_gt))
    exact_chunks = sum(bool(row["exact"]) for row in decision_rows)
    result = {
        "episode_id": item.get("id"),
        "video": item.get("video"),
        "decisions": len(decision_rows),
        "actions": len(numeric_gt),
        "action_correct": correct,
        "action_accuracy": correct / max(len(numeric_gt), 1),
        "chunk_exact": exact_chunks,
        "chunk_exact_match": exact_chunks / max(len(decision_rows), 1),
        "seconds": time.perf_counter() - started,
    }
    if save_outputs:
        result["decision_rows"] = decision_rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=64)
    parser.add_argument("--max-decisions", type=int, default=-1)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--num-future-steps", type=int, default=4)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--save-outputs", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=False)
    with args.annotations.open() as handle:
        annotations = json.load(handle)
    if args.max_episodes > 0:
        annotations = annotations[: args.max_episodes]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tokenizer = build_model(args.model_path, device, dtype, args.model_max_length)
    processor = model.get_vision_tower().image_processor

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(annotations, start=1):
        try:
            row = evaluate_episode(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                item=item,
                data_root=args.data_root,
                device=device,
                dtype=dtype,
                num_frames=args.num_frames,
                num_future_steps=args.num_future_steps,
                num_history=args.num_history,
                max_decisions=args.max_decisions,
                save_outputs=args.save_outputs,
            )
        except Exception as exc:
            row = {"episode_id": item.get("id"), "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(json.dumps({"event": "progress", "index": index, **row}, ensure_ascii=False), flush=True)

    valid = [row for row in rows if "error" not in row]
    total_actions = sum(row["actions"] for row in valid)
    total_correct = sum(row["action_correct"] for row in valid)
    total_chunks = sum(row["decisions"] for row in valid)
    exact_chunks = sum(row["chunk_exact"] for row in valid)
    summary = {
        "protocol": {
            "benchmark": "StreamVLN R2R pre-collected trajectory data",
            "mode": "teacher-forced open-loop; predictions are not executed",
            "streaming": f"num_frames={args.num_frames}, num_history={args.num_history}, future_steps={args.num_future_steps}",
            "checkpoint": str(args.model_path),
            "episodes_requested": len(annotations),
            "episodes_valid": len(valid),
            "episodes_failed": len(rows) - len(valid),
        },
        "metrics": {
            "action_accuracy": total_correct / max(total_actions, 1),
            "chunk_exact_match": exact_chunks / max(total_chunks, 1),
            "actions": total_actions,
            "chunks": total_chunks,
            "gt_distribution": dict(Counter(value for row in valid for value in [])),
        },
        "runtime": {"seconds_total": time.perf_counter() - started},
        "rows": rows,
    }
    # Store readable aggregate class counts from the per-decision JSON.
    gt_counts = Counter()
    pred_counts = Counter()
    for row in valid:
        for decision in row.get("decision_rows", []):
            gt_counts.update(decision["gt"])
            pred_counts.update(decision["pred"])
    summary["metrics"]["gt_distribution"] = dict(gt_counts)
    summary["metrics"]["pred_distribution"] = dict(pred_counts)
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "summary", **summary["metrics"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
