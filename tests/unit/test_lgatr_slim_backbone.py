"""Unit tests for the L-GATr-slim backbone (ShellGATr, ``backbone: lgatr_slim``).

Validates:
1. A SlimDiTBlock is Lorentz-equivariant: transforming every input vector by a
   random Lorentz transformation transforms the output vectors the same way and
   leaves the output scalars unchanged (float64, tolerance 1e-12).
2. Padded tokens do not change the output of the active tokens.
3. Every parameter gets a finite gradient, so DDP static_graph sees all of them.
4. The released ShellGATr checkpoint loads strictly (skips if the file is absent).
5. Without lgatr, importing the model names the gatr extra.

The tests need the gatr extra (``uv sync --extra gatr``) and skip without it.

Run with: uv run pytest tests/unit/test_lgatr_slim_backbone.py -v
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("lgatr")

from shellflow.models.riemannian_DiTs_model_slimLGATr import (  # noqa: E402
    RiemannianSlimDiTsModel,
    SlimDiTBlock,
    join_tokens,
    split_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path(
    os.environ.get(
        "SHELLGATR_CHECKPOINT",
        REPO_ROOT / "checkpoints" / "shellgatr_2to4lep" / "model.ckpt",
    )
)
METRIC = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64))

# --- Fixtures --------------------------------------------------------------


def _build_model(aux_head: bool = True) -> RiemannianSlimDiTsModel:
    """Small slim model: 4 vectors and 16 scalars per token, 2 blocks."""
    torch.manual_seed(42)
    return RiemannianSlimDiTsModel(
        v_channels=4,
        s_channels=16,
        d_model=32,
        num_heads=2,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_met_tokens=2,
        num_extra_features=13,
        per_particle_adaln=True,
        aux_head_config={"enabled": aux_head},
    )


def _randomize(module: torch.nn.Module, seed: int = 0) -> None:
    """Add noise to every parameter, so no zero-init gate hides a path."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))


def _make_padded_batch(B: int = 3, N: int = 7, seed: int = 0) -> dict:
    """Padded batch with every particle type; the last two tokens are padding."""
    g = torch.Generator().manual_seed(seed)
    z_t = torch.randn(B, N, 5, generator=g)
    z_t[..., 1:4] = torch.nn.functional.normalize(z_t[..., 1:4], dim=-1)
    p_type = torch.tensor([11, 13, 1, 22, 15, 0, 0]).expand(B, N).clone()
    p_type[0, 3] = 10  # one large-R jet
    mask = (p_type > 0).float()
    return {
        "z_t": z_t,
        "particle_type": p_type,
        "particle_charge": torch.randint(0, 3, (B, N), generator=g) * mask.long(),
        "particle_extra": torch.randn(B, N, 13, generator=g) * mask.unsqueeze(-1),
        "t": torch.rand(B, generator=g) * 0.98 + 0.01,
        "particle_mask": mask,
        "met_vector": torch.randn(B, 2, generator=g) * 20.0,
    }


def _forward(model, batch: dict):
    return model(
        batch["z_t"],
        batch["particle_type"],
        batch["particle_charge"],
        batch["particle_extra"],
        batch["t"],
        batch["particle_mask"],
        met_vector=batch["met_vector"],
        return_aux=True,
    )


def _random_lorentz(seed: int) -> torch.Tensor:
    """Random proper orthochronous Lorentz transformation in (E, px, py, pz)."""
    g = torch.Generator().manual_seed(seed)
    rapidity = 0.2 + 0.8 * torch.rand((), generator=g, dtype=torch.float64)
    n = torch.nn.functional.normalize(
        torch.randn(3, generator=g, dtype=torch.float64), dim=0
    )
    boost = torch.eye(4, dtype=torch.float64)
    boost[0, 0] = torch.cosh(rapidity)
    boost[0, 1:] = boost[1:, 0] = torch.sinh(rapidity) * n
    boost[1:, 1:] += (torch.cosh(rapidity) - 1) * torch.outer(n, n)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    rotation = torch.eye(4, dtype=torch.float64)
    rotation[1:, 1:] = q
    lam = boost @ rotation
    torch.testing.assert_close(lam.T @ METRIC @ lam, METRIC, rtol=0, atol=1e-12)
    return lam


# --- Tests -----------------------------------------------------------------


class TestSlimBlockLorentzEquivariance:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_block_is_lorentz_equivariant(self, seed):
        """block(Lambda v, s) = (Lambda v_out, s_out) for a random Lorentz Lambda."""
        v_ch, s_ch, cond_dim, B, N, T = 4, 16, 32, 2, 5, 3
        torch.manual_seed(seed)
        block = SlimDiTBlock(v_ch, s_ch, num_heads=2, cond_dim=cond_dim, mlp_ratio=2)
        block = block.double().eval()
        _randomize(block, seed)

        g = torch.Generator().manual_seed(100 + seed)
        v = torch.randn(B, N, 4, v_ch, generator=g, dtype=torch.float64)
        s = torch.randn(B, N, s_ch, generator=g, dtype=torch.float64)
        met_v = torch.randn(B, T, 4, 1, generator=g, dtype=torch.float64)
        met_s = torch.randn(B, T, s_ch, generator=g, dtype=torch.float64)
        time_emb = torch.randn(B, cond_dim, generator=g, dtype=torch.float64)
        cond = torch.randn(B, N, cond_dim, generator=g, dtype=torch.float64)
        mask = torch.ones(B, N, dtype=torch.float64)
        mask[1, -2:] = 0.0

        lam = _random_lorentz(seed)

        def transform(x):
            return torch.einsum("ij,...jc->...ic", lam, x)

        def run(v_in, met_v_in):
            return block(
                join_tokens(v_in, s),
                time_emb,
                attention_mask=mask,
                met_context=(met_v_in, met_s),
                cond=cond,
            )

        with torch.no_grad():
            v_out, s_out = split_tokens(run(v, met_v), v_ch)
            v_out_t, s_out_t = split_tokens(run(transform(v), transform(met_v)), v_ch)

        assert (v_out - v).abs().max() > 1e-3, "the block acts as the identity"
        torch.testing.assert_close(v_out_t, transform(v_out), rtol=0, atol=1e-12)
        torch.testing.assert_close(s_out_t, s_out, rtol=0, atol=1e-12)


class TestSlimModelPadding:
    def test_padding_does_not_leak(self):
        """Changing every input of the padded tokens leaves the active outputs unchanged."""
        model = _build_model().double().eval()
        _randomize(model)
        batch = _make_padded_batch()
        batch = {
            k: v.double() if v.is_floating_point() else v for k, v in batch.items()
        }
        pad = batch["particle_mask"] < 0.5

        changed = {k: v.clone() for k, v in batch.items()}
        changed["z_t"][pad] = torch.randn_like(changed["z_t"][pad]) * 5.0
        changed["particle_type"][pad] = 13
        changed["particle_charge"][pad] = 2
        changed["particle_extra"][pad] = 7.0

        with torch.no_grad():
            out, out_aux = _forward(model, batch)
            out_c, out_aux_c = _forward(model, changed)

        active = ~pad
        assert out[active].abs().max() > 1e-3
        torch.testing.assert_close(out_c[active], out[active], rtol=0, atol=1e-12)
        torch.testing.assert_close(
            out_aux_c[active], out_aux[active], rtol=0, atol=1e-12
        )


class TestSlimModelGradients:
    def test_every_parameter_gets_a_gradient(self):
        """Every parameter is in the graph and gets a finite gradient."""
        model = _build_model().train()
        _randomize(model)
        batch = _make_padded_batch()
        out, out_aux = _forward(model, batch)
        active = batch["particle_mask"] > 0.5
        loss = out[active].pow(2).sum() + out_aux[active].pow(2).sum()
        loss.backward()

        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, f"parameters without a gradient: {missing}"
        not_finite = [
            n for n, p in model.named_parameters() if not torch.isfinite(p.grad).all()
        ]
        assert not not_finite, f"non-finite gradients: {not_finite}"
        zero = [n for n, p in model.named_parameters() if p.grad.abs().max() == 0]
        assert not zero, f"parameters with a zero gradient: {zero}"


class TestSlimModelChecks:
    def test_rejects_width_mismatch(self):
        with pytest.raises(ValueError, match="d_model"):
            RiemannianSlimDiTsModel(
                v_channels=4, s_channels=16, d_model=64, per_particle_adaln=True
            )

    def test_rejects_free_chart(self):
        with pytest.raises(ValueError, match="chart"):
            RiemannianSlimDiTsModel(
                v_channels=4,
                s_channels=16,
                d_model=32,
                num_heads=2,
                dropout=0.0,
                per_particle_adaln=True,
                chart="free",
                direction_geometry="R3",
            )

    def test_missing_lgatr_names_the_extra(self, monkeypatch):
        class _NoLgatr:
            """Import finder that behaves as if lgatr were not installed."""

            def find_spec(self, fullname, path=None, target=None):
                if fullname == "lgatr" or fullname.startswith("lgatr."):
                    raise ModuleNotFoundError(
                        f"No module named {fullname!r}", name=fullname
                    )
                return None

        for name in [m for m in sys.modules if m == "lgatr" or m.startswith("lgatr.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, "meta_path", [_NoLgatr(), *sys.meta_path])
        monkeypatch.delitem(sys.modules, "shellflow.models.riemannian_DiTs_model_slimLGATr")
        with pytest.raises(ModuleNotFoundError, match="uv sync --extra gatr"):
            importlib.import_module("shellflow.models.riemannian_DiTs_model_slimLGATr")


@pytest.mark.requires_data
@pytest.mark.skipif(
    not CHECKPOINT.exists(), reason=f"ShellGATr checkpoint not found at {CHECKPOINT}"
)
def test_release_checkpoint_loads_strict():
    """The released ShellGATr checkpoint loads strictly into the slim model."""
    from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule

    module = RiemannianDiTsModule.load_from_checkpoint(
        CHECKPOINT, map_location="cpu", strict=True
    )
    assert module.hparams["model_config"]["backbone"] == "lgatr_slim"
    for model in (module.model, module.ema_model):
        assert isinstance(model, RiemannianSlimDiTsModel)
        assert sum(p.numel() for p in model.parameters()) == 3_087_498
