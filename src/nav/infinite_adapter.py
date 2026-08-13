"""Infinite-World / Wan 权重兼容层。

加载 checkpoint 时忽略 HPMC (`latent_encoder.*`)；前向时用 Identity 彻底旁路
HPMC，并分别把 Register 注入 history latent 或文本条件。V1 final 的 action
接口必须进入 shared Wan/DiT backbone：`A_cur` 作为 DiT condition/context，
`A_query` 作为 shared action tokens，而不是旁路小 head。
"""

from __future__ import annotations

import torch
from torch import nn
from safetensors.torch import load_file

from infworld.models.dit_model import WanModel

from .register_memory import RegisterMemory
from .spatial_register_memory import SpatialRegisterMemory
from .stage_one_action_interface import StageOneActionInterface


class InfiniteRegisterAdapter(nn.Module):
    VARIANTS = {"latent_prefix", "dit_condition"}

    def __init__(
        self,
        variant: str,
        model_cfg: dict,
        register_cfg: dict,
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"未知 variant: {variant}")
        self.variant = variant
        # 先以原结构建立，保证 checkpoint key 可审计；加载后 remove_hmpc()。
        self.backbone = WanModel(**model_cfg, use_convenc=True)
        self.register_memory = (
            SpatialRegisterMemory(**register_cfg)
            if variant == "latent_prefix"
            else RegisterMemory(**register_cfg)
        )
        self.action_interface = StageOneActionInterface()
        hidden_dim = int(model_cfg.get("dim", 1536))
        self.action_horizon = 4
        self.action_vocab_size = 10
        self.current_move_embedding = nn.Embedding(self.action_vocab_size, hidden_dim)
        self.current_view_embedding = nn.Embedding(self.action_vocab_size, hidden_dim)
        self.current_action_type = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.action_query = nn.Parameter(torch.randn(self.action_horizon, hidden_dim) * 0.02)
        self.action_query_type = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.action_out_norm = nn.LayerNorm(hidden_dim)
        self.move_head = nn.Linear(hidden_dim, self.action_vocab_size)
        self.view_head = nn.Linear(hidden_dim, self.action_vocab_size)
        self.hmpc_removed = False

    def load_infinite_checkpoint(self, path: str) -> dict:
        """兼容旧 InfiniteWorld ckpt 与官方 Wan2.1 safetensors。

        - InfiniteWorld ckpt：旧实验复现用，严格加载。
        - Wan2.1 safetensors：V1 正式 Stage One 用，只加载 backbone 中 key/shape
          匹配的 DiT 主体权重；HPMC/local action/token-type 等增量层保持初始化。
        """
        if path.endswith(".safetensors"):
            raw_state = load_file(path, device="cpu")
            backbone_state = self.backbone.state_dict()
            state = {
                key: value
                for key, value in raw_state.items()
                if key in backbone_state and tuple(value.shape) == tuple(backbone_state[key].shape)
            }
            mismatched = [
                key
                for key, value in raw_state.items()
                if key in backbone_state
                and tuple(value.shape) != tuple(backbone_state[key].shape)
                and key not in state
            ]
            result = self.backbone.load_state_dict(state, strict=False)
            self.remove_hmpc()
            return {
                "source_type": "wan2.1_official_safetensors",
                "loaded_keys": len(state),
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
                "mismatched_keys": mismatched,
                "partial_loaded": [],
            }

        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        backbone_state = self.backbone.state_dict()
        filtered_state = {
            key: value
            for key, value in state.items()
            if key in backbone_state and tuple(value.shape) == tuple(backbone_state[key].shape)
        }
        mismatched = [
            key
            for key, value in state.items()
            if key in backbone_state and tuple(value.shape) != tuple(backbone_state[key].shape)
        ]
        result = self.backbone.load_state_dict(filtered_state, strict=False)
        self.remove_hmpc()
        return {
            "source_type": "infiniteworld_ckpt",
            "loaded_keys": len(filtered_state),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "mismatched_keys": mismatched,
            "partial_loaded": [],
        }

    def remove_hmpc(self) -> None:
        self.backbone.latent_encoder = nn.Identity()
        self.backbone.use_convenc = False
        if self.variant == "dit_condition":
            # B 完全删除 latent history；文本与 Register 在 condition token
            # 维拼接，所以为新增 Register 预留长度。
            self.backbone.model_max_length += self.register_memory.num_registers
        self.hmpc_removed = True

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.register_memory.requires_grad_(True)
        self.action_interface.requires_grad_(True)
        for module in [
            self.current_move_embedding,
            self.current_view_embedding,
            self.action_out_norm,
            self.move_head,
            self.view_head,
        ]:
            module.requires_grad_(True)
        self.current_action_type.requires_grad_(True)
        self.action_query.requires_grad_(True)
        self.action_query_type.requires_grad_(True)

    def build_current_action_condition(
        self,
        move: torch.Tensor,
        view: torch.Tensor,
    ) -> torch.Tensor:
        """构造 `A_cur` DiT condition token。

        `A_cur` 是 video generation condition：它不占用主 video/action token
        序列，而是追加到 Wan block 的 cross-attention context 中。
        """

        dtype = self.action_query.dtype
        move_ids = move.clamp_min(0).clamp_max(self.action_vocab_size - 1)
        view_ids = view.clamp_min(0).clamp_max(self.action_vocab_size - 1)
        cur = (
            self.current_move_embedding(move_ids).mean(dim=1)
            + self.current_view_embedding(view_ids).mean(dim=1)
            + self.current_action_type[0]
        )
        return cur[:, None].to(dtype=dtype)

    def build_shared_action_tokens(
        self,
        batch: int,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, int]:
        """构造 `A_query` shared action tokens。

        `A_query` 是 policy/action output 的待预测变量，进入主 Wan/DiT token
        stream，并从 shared action hidden 解码出 `A_out`。
        """

        query = self.action_query[None].expand(batch, -1, -1) + self.action_query_type[None]
        return query.to(device=device, dtype=self.action_query.dtype), self.action_horizon

    def decode_action_hidden(
        self,
        shared_action_hidden: torch.Tensor | None,
    ) -> dict[str, torch.Tensor] | None:
        if shared_action_hidden is None:
            return None
        action_hidden = shared_action_hidden[:, :self.action_horizon]
        action_hidden = self.action_out_norm(action_hidden.to(self.action_out_norm.weight.dtype))
        return {
            "action_hidden": action_hidden,
            "move_logits": self.move_head(action_hidden),
            "view_logits": self.view_head(action_hidden),
        }

    def shared_action_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "current_move_embedding.weight": self.current_move_embedding.weight,
            "current_view_embedding.weight": self.current_view_embedding.weight,
            "current_action_type": self.current_action_type,
            "action_query": self.action_query,
            "action_query_type": self.action_query_type,
            "action_out_norm.weight": self.action_out_norm.weight,
            "action_out_norm.bias": self.action_out_norm.bias,
            "move_head.weight": self.move_head.weight,
            "move_head.bias": self.move_head.bias,
            "view_head.weight": self.view_head.weight,
            "view_head.bias": self.view_head.bias,
        }

    def load_shared_action_state_dict(self, state: dict[str, torch.Tensor]) -> dict[str, list[str]]:
        current = self.shared_action_state_dict()
        missing = []
        unexpected = []
        with torch.no_grad():
            for key, value in state.items():
                if key not in current:
                    unexpected.append(key)
                    continue
                if tuple(value.shape) != tuple(current[key].shape):
                    unexpected.append(key)
                    continue
                current[key].copy_(value)
            for key in current:
                if key not in state:
                    missing.append(key)
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def prepare_conditions(
        self,
        registers: torch.Tensor,
        local_latent: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.variant == "latent_prefix":
            # Register 是固定 T=4 的 latent-like state；local memory 由
            # WanModel 的显式 local_memory 参数只追加一次。
            return registers, y, y_mask

        register_condition = self.register_memory.as_dit_condition(registers)
        # [B,1,L_text,4096] + [B,1,R,4096]；RE10K 无 caption 时，
        # y 是 UMT5 对空字符串的真实编码。
        y = torch.cat([y, register_condition[:, None]], dim=2)
        register_mask = torch.ones(
            y_mask.shape[0],
            register_condition.shape[1],
            device=y_mask.device,
            dtype=y_mask.dtype,
        )
        y_mask = torch.cat([y_mask, register_mask], dim=1)
        # history T=0；WanModel 只追加一次显式 local memory。
        empty_memory = local_latent[:, :, :0]
        return empty_memory, y, y_mask

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        registers: torch.Tensor,
        local_latent: torch.Tensor,
        move: torch.Tensor,
        view: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if not self.hmpc_removed:
            raise RuntimeError("必须先加载 checkpoint 并移除 HPMC")
        image_cond, y, y_mask = self.prepare_conditions(
            registers, local_latent, y, y_mask
        )
        if self.variant == "dit_condition":
            assert image_cond.shape[2] == 0
            assert y.shape[2] == (
                self.backbone.model_max_length
            )
            assert local_latent.shape[2] == 1
        return_shared_action_tokens = bool(kwargs.pop("return_shared_action_tokens", False))
        current_action_condition = self.build_current_action_condition(
            move,
            view,
        )
        shared_action_tokens, _ = self.build_shared_action_tokens(
            move.shape[0],
            device=move.device,
        )
        native_noop_move = torch.zeros_like(move)
        native_noop_view = torch.zeros_like(view)
        out = self.backbone(
            x=x,
            t=t,
            y=y,
            y_mask=y_mask,
            image_cond=image_cond,
            memory_is_precomputed=True,
            local_memory=local_latent,
            # 正式 V1 中 A_cur 只通过 DiT condition/context 进入 backbone；
            # 原 InfiniteWorld action_encoder 保持 no-op，避免 action 双路注入。
            move=native_noop_move,
            view=native_noop_view,
            shared_condition_tokens=current_action_condition,
            shared_action_tokens=shared_action_tokens,
            disable_native_action_embedding=True,
            return_shared_action_tokens=return_shared_action_tokens,
            **kwargs,
        )
        if return_shared_action_tokens:
            action_outputs = self.decode_action_hidden(
                out.get("shared_action_hidden"),
            )
            out["action_outputs"] = action_outputs
        return out

    @torch.no_grad()
    def policy_forward(
        self,
        *,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        registers: torch.Tensor,
        local_latent: torch.Tensor,
        dtype: torch.dtype | None = None,
        action_timestep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Policy/action-only path through the same Wan/DiT backbone.

        该路径不输入 future noisy video，不输入当前 action condition；只用
        Register/local/text prefix + A_query tokens 跑 shared backbone，再从
        action token hidden 预测动作 logits。
        """

        if not self.hmpc_removed:
            raise RuntimeError("必须先加载 checkpoint 并移除 HPMC")
        image_cond, y, y_mask = self.prepare_conditions(
            registers, local_latent, y, y_mask
        )
        device = local_latent.device
        dtype = dtype or local_latent.dtype
        batch, channels, _, height, width = local_latent.shape
        empty_future = torch.empty(
            batch,
            channels,
            0,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        timestep = torch.zeros(batch, device=device, dtype=torch.float32)
        if action_timestep is None:
            action_timestep = torch.full(
                (batch,),
                1000.0,
                device=device,
                dtype=torch.float32,
            )
        else:
            action_timestep = action_timestep.to(device=device, dtype=torch.float32)
        noop = torch.zeros(batch, self.action_horizon, device=device, dtype=torch.long)
        shared_action_tokens, _ = self.build_shared_action_tokens(
            batch,
            device=device,
        )
        out = self.backbone(
            x=empty_future,
            t=timestep,
            y=y,
            y_mask=y_mask,
            image_cond=image_cond,
            memory_is_precomputed=True,
            local_memory=local_latent,
            move=noop,
            view=noop,
            shared_action_tokens=shared_action_tokens,
            disable_native_action_embedding=True,
            action_timestep=action_timestep,
            return_shared_action_tokens=True,
            policy_only=True,
        )
        action_outputs = self.decode_action_hidden(
            out.get("shared_action_hidden"),
        )
        assert action_outputs is not None
        return action_outputs
