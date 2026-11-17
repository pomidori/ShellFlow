"""Riemannian flow-matching DiT with L-GATr-slim blocks (ShellGATr).

Select this model with ``backbone: lgatr_slim`` in the model config.  The attention
blocks of ``riemannian_DiTs_model.RiemannianDiTsModel`` are replaced by the
self-attention, cross-attention, and MLP layers of L-GATr-slim (Petitjean, Plehn,
Spinner, Koethe, arXiv:2512.17011; package ``lgatr``, github.com/heidelberg-hepml/lgatr,
commit 86d3fb8, installed by ``uv sync --extra gatr``).  The flow matching, the
manifold, the time sampling, the aux head, and the quality weighting are inherited
unchanged.

Every token is a pair (v, s): v_channels Lorentz vectors and s_channels scalars.
Between the blocks the pair is stored flat, h = cat(v.flatten(-2), s), with
4 v_channels + s_channels = d_model, so the block output has the shape (B, N, d_model)
of the original blocks.

What is equivariant.  The blocks are Lorentz-equivariant: scalars only ever multiply
vectors, and vectors get no bias.  The conditioning (time, charge, extras) is scalar:
adaLN-Zero writes gains and gates on v, and gains, shifts, and gates on s.  The network
as a whole is not: the time and beam axes enter as reference vectors, as in L-GATr
event generation, which leaves the rotation about the beam.  The input scalars and the
S2 velocity, the spatial part of an output vector, are frame-dependent too.

Input vectors, per token: p^mu/p_scale, the time axis, and the beam axis, all
(E, px, py, pz).  The MET vector is (0, mx, my, 0)/p_scale, in the same units.
Massless types have beta = 1, so a muon enters lightlike.  Input scalars, per token:
the coordinates z[0] (log E or chi), z[3] (n_z), and z[4] (log m).  The decoder of the
manifold is not used for the input: its muon branch has a sqrt with an infinite slope
at E = m_rest, which gives NaN gradients through the Euler steps.

Precision.  A pair mass is a small inner product of two large vectors, so one bf16
rounding of v destroys it.  Under autocast the vector stream stays in fp32, as inside
the lgatr layers: bmm, matmul, and linear on v run in fp32.  So does all the
conditioning, because it writes per-channel gains on v, and a bf16 rounding of a gain
distorts v like a rounding of v itself.  The layers of the scalar stream stay on
autocast.  TF32 matmuls would round the fp32 vector stream as well, so the training
script turns TF32 off for this backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from lgatr.layers.slim_layers import (
        SlimCrossAttention,
        SlimMLP,
        SlimRMSNorm,
        SlimSelfAttention,
    )
    from lgatr.utils.autocast import minimum_autocast_precision
except ModuleNotFoundError as err:
    if err.name != "lgatr":
        raise
    raise ModuleNotFoundError(
        "backbone 'lgatr_slim' needs the lgatr package; install it with `uv sync --extra gatr`",
        name="lgatr",
    ) from err

from shellflow.models.riemannian_DiTs_model import (
    NUM_PARTICLE_TYPES,
    RiemannianDiTsModel,
    _grouped_linear,
)


def split_tokens(h: torch.Tensor, v_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat token (..., d_model) -> vectors (..., 4, v_channels), scalars (..., s_channels)."""
    v = h[..., : 4 * v_channels].unflatten(-1, (4, v_channels))
    return v, h[..., 4 * v_channels :]


def join_tokens(v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Vectors (..., 4, v_channels), scalars (..., s_channels) -> flat token (..., d_model)."""
    return torch.cat([v.flatten(-2), s], dim=-1)


@minimum_autocast_precision(torch.float32, output="high")
def _vector_linear(w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-token linear, weights (..., d, c) and vectors (..., 4, c) -> (..., 4, d).

    w is fp32, so the output is fp32 too.
    """
    return torch.einsum("...dc,...ic->...id", w, v)


@minimum_autocast_precision(torch.float32, output=None)
def _fp32_call(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """module(x) in fp32 under autocast.

    Output None: x may be bf16, and output="high" would return bf16 then.
    """
    return module(x)


def _grouped_vector_linear(n: int, out_f: int, in_f: int) -> nn.Parameter:
    """Stacked weights of ``n`` bias-free vector linears, shape (n, out_f, in_f).

    See ``_grouped_linear``.
    """
    return nn.Parameter(
        torch.stack(
            [
                nn.Linear(in_f, out_f, bias=False).weight.detach().clone()
                for _ in range(n)
            ]
        )
    )


class SlimDiTBlock(nn.Module):
    """adaLN-Zero transformer block on L-GATr-slim layers.

    The same sub-layers in the same order as ``AdaLNZeroDiTBlock``: self-attention, MET
    cross-attention, and MLP, on the flat token.  Each sub-layer has a gain on v and on
    s, each output gate acts on v and on s, and the shifts act on s only.  All gates
    start at zero, so every block starts as the identity.
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.v_channels, self.s_channels = v_channels, s_channels
        self.norm1 = SlimRMSNorm(v_channels, s_channels, elementwise_affine=False)
        self.attn = SlimSelfAttention(v_channels, s_channels, num_heads)

        self.met_cross_attention = True
        self.cross_norm = SlimRMSNorm(v_channels, s_channels, elementwise_affine=False)
        self.cross_attn = SlimCrossAttention(
            v_channels, 1, s_channels, s_channels, num_heads
        )

        self.norm2 = SlimRMSNorm(v_channels, s_channels, elementwise_affine=False)
        self.mlp = SlimMLP(
            v_channels=v_channels, s_channels=s_channels, mlp_ratio=mlp_ratio
        )

        v, s = v_channels, s_channels
        self.sizes = (s, s, v, s, v, s, v, s, v, s, s, v, s, v)
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, sum(self.sizes)))
        nn.init.zeros_(self.adaln_mlp[-1].weight)
        nn.init.zeros_(self.adaln_mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        met_context: tuple[torch.Tensor, torch.Tensor],
        cond: torch.Tensor,
    ) -> torch.Tensor:
        gs1, bs1, gv1, as1, av1, gsc, gvc, asc, avc, gs2, bs2, gv2, as2, av2 = (
            _fp32_call(self.adaln_mlp, time_emb.unsqueeze(1) + cond).split(
                self.sizes, dim=-1
            )
        )
        # every token attends to the active tokens
        keys = (attention_mask > 0.5)[:, None, None, :]
        v, s = split_tokens(x, self.v_channels)

        n_v, n_s = self.norm1(v, s)
        o_v, o_s = self.attn(
            n_v * (1 + gv1).unsqueeze(-2), n_s * (1 + gs1) + bs1, attn_mask=keys
        )
        v, s = v + av1.unsqueeze(-2) * o_v, s + as1 * o_s

        n_v, n_s = self.cross_norm(v, s)
        o_v, o_s = self.cross_attn(
            n_v * (1 + gvc).unsqueeze(-2),
            met_context[0],
            n_s * (1 + gsc),
            met_context[1],
        )
        v, s = v + avc.unsqueeze(-2) * o_v, s + asc * o_s

        n_v, n_s = self.norm2(v, s)
        o_v, o_s = self.mlp(n_v * (1 + gv2).unsqueeze(-2), n_s * (1 + gs2) + bs2)
        return join_tokens(v + av2.unsqueeze(-2) * o_v, s + as2 * o_s)


class RiemannianSlimDiTsModel(RiemannianDiTsModel):
    """``RiemannianDiTsModel`` with L-GATr-slim blocks; see the module docstring.

    Padded forward only.  The slim blocks have no dropout, so ``dropout`` must be 0.
    """

    def __init__(
        self,
        v_channels: int = 16,
        s_channels: int = 64,
        p_scale: float = 50.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        d_model = self.d_model
        if 4 * v_channels + s_channels != d_model:
            raise ValueError(
                f"4 v_channels + s_channels = {4 * v_channels + s_channels}, d_model = {d_model}"
            )
        if not self.per_particle_adaln or self.num_met_tokens == 0:
            raise ValueError("the slim blocks need per_particle_adaln and MET tokens")
        if kwargs.get("dropout", 0.0) != 0.0:
            raise ValueError(
                f"the slim blocks have no dropout, got {kwargs['dropout']}"
            )
        if self.chart != "onshell":
            raise ValueError(
                f"the slim input vectors need chart='onshell', got {self.chart!r}"
            )
        self.v_channels, self.s_channels, self.p_scale = v_channels, s_channels, p_scale
        v, s = v_channels, s_channels

        self.in_w, self.in_b = _grouped_linear(NUM_PARTICLE_TYPES, s, 3)
        self.in_v_w = _grouped_vector_linear(NUM_PARTICLE_TYPES, v, 3)
        identity_dim = self.identity_film[0].in_features
        self.identity_film = nn.Sequential(
            nn.Linear(identity_dim, d_model), nn.SiLU(), nn.Linear(d_model, 2 * s + v)
        )
        self.met_kv_proj = nn.Sequential(
            nn.Linear(1, kwargs.get("met_embed_dim", 64)),
            nn.SiLU(),
            nn.Linear(kwargs.get("met_embed_dim", 64), self.num_met_tokens * s),
        )
        num_heads, num_layers = kwargs.get("num_heads", 8), kwargs.get("num_layers", 6)
        mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        if mlp_ratio != int(mlp_ratio):
            raise ValueError(
                f"the slim MLP needs an integer mlp_ratio, got {mlp_ratio}"
            )
        mlp_ratio = int(mlp_ratio)
        self.blocks = nn.ModuleList(
            [
                SlimDiTBlock(v, s, num_heads, d_model, mlp_ratio)
                for _ in range(num_layers)
            ]
        )

        self.out_norm = SlimRMSNorm(v, s, elementwise_affine=False)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * s + v))
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)

        # Output: coordinates 0 and 4 from the scalars, coordinates 1:4 from the spatial
        # part of one output vector.
        self.out_w, self.out_b = _grouped_linear(NUM_PARTICLE_TYPES, 2, s)
        self.out_v_w = _grouped_vector_linear(NUM_PARTICLE_TYPES, 1, v)
        if self.use_aux_head:
            nn.init.zeros_(self.out_w)
            nn.init.zeros_(self.out_b)
            nn.init.zeros_(self.out_v_w)
            self.aux_w, self.aux_b = _grouped_linear(2, 2, s)
            self.aux_v_w = _grouped_vector_linear(2, 1, v)

        self.register_buffer(
            "_reference_vectors",
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            persistent=False,
        )

    def _input_vectors(
        self, z: torch.Tensor, type_idx: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        """(..., 4, 3): channels p^mu/p_scale, time axis, and beam axis, in (E, px, py, pz).

        The gradients are finite everywhere.
        """
        n_hat = F.normalize(z[..., 1:4], dim=-1, eps=1e-6)
        massive = self._is_massive_lut[type_idx]
        log_e = torch.log(self._e_ref_lut[type_idx]) + z[..., 0].clamp(-10, 10)
        beta = torch.ones_like(log_e)
        chi = (
            z[..., 0] * self._chi_sigma_lut[type_idx].clamp(min=0.01)
            + self._chi_mu_lut[type_idx]
        ).clamp(0, 12)
        log_m = (
            z[..., 4] * self._logm_sigma_lut[type_idx].clamp(min=0.01)
            + self._logm_mu_lut[type_idx]
        ).clamp(-10, 10)
        # log E = log m + log cosh(chi), and
        # log cosh(chi) = chi + log(1 + exp(-2 chi)) - log 2 has no overflow
        log_e = torch.where(
            massive, log_m + chi + torch.log1p(torch.exp(-2 * chi)) - 0.6931472, log_e
        )
        beta = torch.where(massive, torch.tanh(chi), beta)
        u = torch.cat(
            [torch.ones_like(beta).unsqueeze(-1), beta.unsqueeze(-1) * n_hat], dim=-1
        )
        p = u * torch.exp(log_e).unsqueeze(-1) / self.p_scale
        refs = self._reference_vectors.expand(*p.shape[:-1], 2, 4)
        # (..., 3, 4): channels, components
        vectors = torch.cat([p.unsqueeze(-2), refs], dim=-2)
        return (vectors * active[..., None, None].to(vectors.dtype)).transpose(-1, -2)

    def _embed_input(self, z_t, particle_type, particle_charge, particle_extra, active):
        type_idx = self.map_type_to_idx(particle_type)
        w, b = self.in_w[type_idx], self.in_b[type_idx]
        s = torch.einsum("...di,...i->...d", w, z_t[..., [0, 3, 4]]) + b
        v = _vector_linear(
            self.in_v_w[type_idx], self._input_vectors(z_t, type_idx, active)
        )

        identity = torch.cat(
            [
                self.charge_embed(particle_charge),
                particle_extra[..., : self.num_extra_features],
            ],
            -1,
        )
        gamma_s, beta_s, gamma_v = _fp32_call(self.identity_film, identity).split(
            (self.s_channels, self.s_channels, self.v_channels), dim=-1
        )
        s = (1 + gamma_s) * s + beta_s
        v = (1 + gamma_v).unsqueeze(-2) * v
        cond = _fp32_call(self.cond_proj, identity) * active.unsqueeze(-1).float()
        return join_tokens(v, s) * active.unsqueeze(-1).float(), type_idx, cond

    def _met_context(
        self, met_vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MET tokens: one spatial vector (0, mx, my, 0)/p_scale per token, scalars of |MET|."""
        zero = torch.zeros_like(met_vector[..., :1])
        v = torch.cat([zero, met_vector, zero], dim=-1) / self.p_scale
        v = v[:, None, :, None].expand(-1, self.num_met_tokens, -1, -1)
        s = self.met_kv_proj(torch.log1p(met_vector.norm(dim=-1, keepdim=True)))
        s = s.view(len(met_vector), self.num_met_tokens, self.s_channels)
        return v, s

    def _out_proj(self, h: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
        v, s = split_tokens(h, self.v_channels)
        out_s = (
            torch.einsum("...od,...d->...o", self.out_w[type_idx], s)
            + self.out_b[type_idx]
        )
        out_v = _vector_linear(self.out_v_w[type_idx], v).squeeze(-1)
        return torch.cat([out_s[..., :1], out_v[..., 1:4], out_s[..., 1:]], dim=-1)

    def _out_proj_aux(self, h: torch.Tensor, sector_idx: torch.Tensor) -> torch.Tensor:
        v, s = split_tokens(h, self.v_channels)
        out_s = (
            torch.einsum("...od,...d->...o", self.aux_w[sector_idx], s)
            + self.aux_b[sector_idx]
        )
        out_v = _vector_linear(self.aux_v_w[sector_idx], v).squeeze(-1)
        return torch.cat([out_s[..., :1], out_v[..., 1:4], out_s[..., 1:]], dim=-1)

    def forward(
        self,
        z_t,
        particle_type,
        particle_charge,
        particle_extra,
        t,
        particle_mask,
        met_vector=None,
        return_aux: bool = False,
    ):
        if met_vector is None:
            raise ValueError("the slim model needs met_vector")
        active = particle_mask > 0.5
        h, type_idx, cond = self._embed_input(
            z_t, particle_type, particle_charge, particle_extra, active
        )
        time_emb = _fp32_call(self.time_embed, t * 1000.0)
        met_context = self._met_context(met_vector)
        for block in self.blocks:
            h = block(
                h,
                time_emb,
                attention_mask=particle_mask,
                met_context=met_context,
                cond=cond,
            )

        gamma_s, beta_s, gamma_v = _fp32_call(
            self.final_adaln, time_emb.unsqueeze(1) + cond
        ).split((self.s_channels, self.s_channels, self.v_channels), dim=-1)
        v, s = self.out_norm(*split_tokens(h, self.v_channels))
        h_modulated = join_tokens(
            v * (1 + gamma_v).unsqueeze(-2), s * (1 + gamma_s) + beta_s
        )
        raw = self._postprocess_velocity(
            self._out_proj(h_modulated, type_idx), z_t, type_idx, particle_mask
        )

        raw_aux = None
        if self.use_aux_head:
            sector_idx = self._is_massive_lut[type_idx].long()
            raw_aux = self._postprocess_velocity(
                self._out_proj_aux(h_modulated, sector_idx),
                z_t,
                type_idx,
                particle_mask,
            )
        return (raw, raw_aux) if return_aux else raw

    def forward_geometric(self, *args, **kwargs):
        raise NotImplementedError("the slim model has the padded forward only")

    forward_flat = forward_njt = forward_geometric


def create_riemannian_dits_slim(config: dict) -> RiemannianSlimDiTsModel:
    """Build the slim model from a model config.

    Reads the keys of ``create_riemannian_dits`` plus v_channels, s_channels, and p_scale.
    """
    v_channels, s_channels = config.get("v_channels", 16), config.get("s_channels", 64)
    return RiemannianSlimDiTsModel(
        v_channels=v_channels,
        s_channels=s_channels,
        p_scale=config.get("p_scale", 50.0),
        d_model=config.get("d_model", 4 * v_channels + s_channels),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 6),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        dropout=config.get("dropout", 0.0),
        met_embed_dim=config.get("met_embed_dim", 64),
        num_met_tokens=config.get("num_met_tokens", 4),
        num_extra_features=config.get("num_extra_features", 13),
        manifold_stats=config.get("manifold_stats"),
        direction_geometry=config.get("direction_geometry", "S2"),
        chart=config.get("chart", "onshell"),
        aux_head_config=config.get("aux_head"),
        per_particle_adaln=config.get("per_particle_adaln", False),
        quality_weighting_config=config.get("quality_weighting"),
    )
