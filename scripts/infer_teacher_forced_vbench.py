#!/usr/bin/env python3
"""Teacher-forced 评测：用 episode 的 n 个 GT 历史 chunk 生成第 n+1 个 chunk，
再和 GT 第 n+1 个 chunk 比对（VBench 质量 + LPIPS/SSIM/L1 保真度）。

支持三种模型：nav_a / nav_b（InfiniteRegisterAdapter）与 infworld（原生 HPMC）。
所有模型统一用空 text（null token），隔离 history+action 的预测能力。
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

NAV_ROOT = Path(__file__).resolve().parents[1]
INFINITE_ROOT = NAV_ROOT.parent / "Infinite-World"
sys.path.insert(0, str(NAV_ROOT / "src"))
sys.path.insert(0, str(INFINITE_ROOT))
import infworld.context_parallel.context_parallel_util as cp_util
from infworld.utils.data_utils import save_silent_video
from infworld.utils.prepare_dataloader import get_obj_from_str
from nav.infinite_adapter import InfiniteRegisterAdapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["nav_a", "nav_b", "infworld"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--target-chunk", type=int, default=3)
    p.add_argument("--history-chunks", type=int, default=3)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s); torch.backends.cudnn.deterministic = True


class CFGRegisterWrapper(nn.Module):
    def __init__(self, m):
        super().__init__(); self.model = m
    @property
    def y_embedder(self):
        return self.model.backbone.y_embedder
    def forward(self, x, t, *, image_cond=None, **kw):
        del image_cond
        return self.model(x=x, t=t, **kw)


def build_nav(model_id, ckpt, device):
    variant = "latent_prefix" if model_id == "nav_a" else "dit_condition"
    m = InfiniteRegisterAdapter(
        variant=variant,
        model_cfg={"model_type": "t2v", "dim": 1536, "in_channels": 20,
                   "ffn_dim": 8960, "freq_dim": 256, "num_heads": 12,
                   "num_layers": 30, "out_channels": 16,
                   "caption_channels": 4096, "model_max_length": 512},
        register_cfg=({"channels": 16, "register_frames": 4, "hidden_dim": 64,
                       "num_heads": 4, "num_update_layers": 2, "chunk_time_tokens": 8}
                      if variant == "latent_prefix"
                      else {"latent_channels": 16, "register_dim": 256,
                            "num_registers": 16, "num_update_layers": 2,
                            "num_heads": 8, "caption_channels": 4096}))
    m.remove_hmpc()
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    if st.get("variant") != variant:
        raise ValueError(f"variant mismatch: {st.get('variant')} vs {variant}")
    m.backbone.load_state_dict(st["backbone"], strict=True)
    m.register_memory.load_state_dict(st["register_memory"], strict=True)
    print(f"[NAV] loaded {ckpt} step={st.get('step')}")
    return m.to(device=device, dtype=torch.bfloat16).eval()


def build_infworld(ckpt, device, config):
    dit = get_obj_from_str(config.model_target)(
        out_channels=16, caption_channels=4096, model_max_length=512,
        enable_context_parallel=False, **config.model_cfg).to(torch.bfloat16)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = st.get("state_dict", st)
    st.pop("pos_embed_temporal", None); st.pop("pos_embed", None)
    dit.load_state_dict(st, strict=False)
    print(f"[InfWorld] loaded {ckpt}")
    return dit.to(device).eval()


def main():
    args = parse_args(); setup_seed(args.seed)
    cp_util.dp_rank = cp_util.cp_rank = 0; cp_util.dp_size = cp_util.cp_size = 1
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(INFINITE_ROOT / "configs/infworld_config.yaml")
    vae = get_obj_from_str(config.vae_target)(**config.vae_cfg).to(device)
    te = get_obj_from_str(config.text_encoder_target)(device=device, **config.text_encoder_cfg)
    te.t5.model.to(device)
    sched = get_obj_from_str(config.scheduler_target)(**config.val_scheduler_cfg)
    sched.num_sampling_steps = args.steps; sched.shift = 7

    ep = torch.load(args.episode, map_location="cpu", weights_only=False)
    chunks = ep["chunks"][0]
    move_all = ep["move"].long(); view_all = ep["view"].long()
    t = args.target_chunk; n = args.history_chunks
    assert t - n >= 0 and t < chunks.shape[0], f"target {t}/hist {n} 越界(共{chunks.shape[0]}chunk)"
    histories = [chunks[i].to(device=device, dtype=torch.bfloat16).unsqueeze(0) for i in range(t - n, t)]
    target_gt = chunks[t].to(device=device, dtype=torch.bfloat16)
    a0 = t * 81
    move = move_all[a0:a0 + 81].to(device); view = view_all[a0:a0 + 81].to(device)
    print(f"[TF] ep={args.episode.stem} t={t} n={n} move_dist={torch.bincount(move, minlength=10).tolist()}")

    tc = torch.load("/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt",
                    map_location="cpu", weights_only=False)
    y = tc["y"].to(device=device, dtype=torch.bfloat16); y_mask = tc["y_mask"].to(device)
    latent_size = target_gt.unsqueeze(0).shape

    if args.model in ("nav_a", "nav_b"):
        model = build_nav(args.model, args.checkpoint, device)
        reg = model.register_memory.extract(histories[0])
        for h in histories[1:]:
            reg = model.register_memory.update(reg, h)
        local = histories[-1][:, :, -1:]
        cfgm = CFGRegisterWrapper(model)
        add = {"image_cond": histories[0], "registers": reg.repeat(2, *([1]*(reg.ndim-1))),
               "local_latent": local.repeat(2, 1, 1, 1, 1),
               "move": move[None].repeat(2, 1), "view": view[None].repeat(2, 1)}
        null_emb = cfgm.y_embedder
        sampler = cfgm
    else:
        model = build_infworld(args.checkpoint, device, config)
        image_cond = torch.cat(histories, dim=2)
        add = {"image_cond": image_cond,
               "move": move[None].repeat(2, 1), "view": view[None].repeat(2, 1)}
        null_emb = model.y_embedder
        sampler = model

    with torch.inference_mode():
        samples = sched.sample(model=sampler, text_encoder=te, null_embedder=null_emb,
                                z_size=latent_size, prompts=[""], guidance_scale=args.cfg_scale,
                                negative_prompts=[""], device=device, additional_args=add)
    torch.cuda.empty_cache()
    print(f"[TF] gen latent {tuple(samples.shape)} gt latent {tuple(target_gt.unsqueeze(0).shape)}")

    with torch.inference_mode():
        gen_px = vae.decode(samples).cpu(); gt_px = vae.decode(target_gt.unsqueeze(0)).cpu()
    print(f"[TF] gen px {tuple(gen_px.shape)} gt px {tuple(gt_px.shape)}")
    save_silent_video(gen_px.to(device), str(args.output_dir / "gen"), fps=30, quality=10)
    save_silent_video(gt_px.to(device), str(args.output_dir / "gt"), fps=30, quality=10)
    print(f"[TF] saved gen.mp4/gt.mp4 to {args.output_dir}")
    _fidelity(args, samples, target_gt, gen_px, gt_px, device)


def _fidelity(args, gen_latent, gt_latent, gen_px, gt_px, device):
    import json, numpy as np
    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as e:
        print(f"[TF] LPIPS unavailable: {e}")
    ssim_fn = None
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except Exception as e:
        print(f"[TF] SSIM unavailable: {e}")
    g = gen_px.to(device).float() / 2 + 0.5
    t_gt = gt_px.to(device).float() / 2 + 0.5
    T = g.shape[2]; l1, ss, lps = [], [], []
    with torch.inference_mode():
        for i in range(T):
            gi = g[:, :, i]; ti = t_gt[:, :, i]
            l1.append(float((gi - ti).abs().mean()))
            if ssim_fn is not None:
                ss.append(float(ssim_fn(gi, ti)))
            if lp is not None:
                lps.append(float(lp(gi * 2 - 1, ti * 2 - 1)))
    out = {"model": args.model, "checkpoint": str(args.checkpoint),
           "episode": args.episode.stem, "target_chunk": args.target_chunk,
           "history_chunks": args.history_chunks, "seed": args.seed,
           "steps": args.steps, "cfg_scale": args.cfg_scale, "num_frames": T,
           "L1_mean": float(np.mean(l1)),
           "SSIM_mean": float(np.mean(ss)) if ss else None,
           "LPIPS_mean": float(np.mean(lps)) if lps else None,
           "L1_per_frame": l1, "SSIM_per_frame": ss, "LPIPS_per_frame": lps}
    (args.output_dir / "fidelity.json").write_text(json.dumps(out, indent=2))
    print(f"[TF] fidelity: L1={out['L1_mean']:.4f}"
          + (f" SSIM={out['SSIM_mean']:.4f}" if out['SSIM_mean'] is not None else "")
          + (f" LPIPS={out['LPIPS_mean']:.4f}" if out['LPIPS_mean'] is not None else ""))


if __name__ == "__main__":
    main()
