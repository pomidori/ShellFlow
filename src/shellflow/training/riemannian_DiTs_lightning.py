"""
PyTorch Lightning module for Riemannian Diffusion Transformer training.

Riemannian Flow Matching (RFM) on unified product manifold R × S² (× R):
  - Massless (e, μ, γ): (logE_norm, n̂, 0) on R × S²
  - Massive (τ, jet, large-R jet): (χ_norm, n̂, logm_norm) on R × S² × R  (Log H³_1)

Linear + SLERP interpolant → velocity regression → Euler ODE generation.
"""

import copy
import math
from dataclasses import asdict
from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from shellflow.config.training_config import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from shellflow.data.tuple_categories import tuple_keys
from shellflow.models.riemannian_DiTs_geometry import (
    conditional_velocity_S2,
    exp_map_sphere_stable,
    geodesic_interpolant_S2,
    sample_uniform_sphere,
)
from shellflow.models.riemannian_DiTs_model import (
    create_riemannian_dits,
)
from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)


_TYPE_NAMES = {
    11: "electron",
    13: "muon",
    22: "photon",
    15: "tau",
    1: "jet",
    10: "largeRjet",
}


# ---------------------------------------------------------------------------
# Per-particle 4-momentum decode helpers (used by the K-body aux losses)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _decode_particle_4mom(
    x: torch.Tensor,
    e_ref: torch.Tensor,
    is_massless: torch.Tensor,
    chi_mu: torch.Tensor,
    chi_sigma: torch.Tensor,
    logm_mu: torch.Tensor,
    logm_sigma: torch.Tensor,
    m_rest: torch.Tensor,
    free_chart: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode manifold representation to (E, p⃗) for both massless and massive.

    free_chart=True (Euclidean ablation): E = e_ref·exp(s_0), |p| = e_ref·exp(s_4)
    for all types — independent coordinates, no on-shell tie.

    Legacy chart (pre-bounded-pT) + muon-on-shell decode:
      Massless: E = e_ref · exp(s_0);
                |p⃗| = sqrt(E² − m_rest²);  p⃗ = |p⃗| · n̂
        m_rest is the per-type fundamental rest mass (m_μ for muon, 0 for
        photon and all composite types). At m_rest = 0 this collapses to the
        lightlike form |p⃗| = E. With m_rest > 0 the on-shell condition
        E² − |p⃗|² = m_rest² holds by construction, eliminating dimuon
        kinematic-threshold leakage below 2·m_μ.
      Massive : χ = s_0·σ_χ + μ_χ;        m = exp(s_4·σ_logm + μ_logm);
                E = m·cosh(χ);            |p⃗| = m·sinh(χ);   p⃗ = |p⃗|·n̂

    Args:
        x: (..., 5) manifold coords
        e_ref: (...) energy reference (massless only)
        is_massless: (...) bool
        chi_mu, chi_sigma, logm_mu, logm_sigma: (...) massive normalization params
        m_rest: (...) per-type fundamental rest mass (massless-branch only)

    Returns:
        E: (...) energy
        p3: (..., 3) 3-momentum
    """
    n_hat = F.normalize(x[..., 1:4], dim=-1, eps=1e-6)

    if free_chart:
        E_f = e_ref * torch.exp(x[..., 0].clamp(-10, 10))
        p_mag_f = e_ref * torch.exp(x[..., 4].clamp(-10, 10))
        return E_f, p_mag_f.unsqueeze(-1) * n_hat

    # Massless: E = e_ref · exp(s_0); |p| = sqrt(E² − m_rest²); p⃗ = |p| · n̂.
    E_ml = e_ref * torch.exp(x[..., 0].clamp(-10, 10))
    p_mag_ml = (E_ml * E_ml - m_rest * m_rest).clamp(min=0.0).sqrt()
    p3_ml = p_mag_ml.unsqueeze(-1) * n_hat

    # Massive: χ = χ_norm * σ + μ, m = exp(logm_norm * σ + μ), E = m*cosh(χ), |p| = m*sinh(χ)
    chi = x[..., 0] * chi_sigma.clamp(min=0.01) + chi_mu
    logm_raw = x[..., 4] * logm_sigma.clamp(min=0.01) + logm_mu
    m = torch.exp(logm_raw.clamp(-10, 10))
    E_mv = m * torch.cosh(chi.clamp(0, 12))
    p_mag = m * torch.sinh(chi.clamp(0, 12))
    p3_mv = p_mag.unsqueeze(-1) * n_hat

    # Select by particle type
    ml = is_massless.unsqueeze(-1)
    E = torch.where(is_massless, E_ml, E_mv)
    p3 = torch.where(ml, p3_ml, p3_mv)

    return E, p3


def _decode_particle_4mom_with_velocity(
    x: torch.Tensor,
    v: torch.Tensor,
    e_ref: torch.Tensor,
    is_massless: torch.Tensor,
    chi_mu: torch.Tensor,
    chi_sigma: torch.Tensor,
    logm_mu: torch.Tensor,
    logm_sigma: torch.Tensor,
    m_rest: torch.Tensor,
    free_chart: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode manifold state+velocity to (E, p⃗, dE/dt, dp⃗/dt) for any type.

    free_chart=True (Euclidean ablation), all types:
        E = e_ref·exp(x[0]);   |p| = e_ref·exp(x[4]);   p⃗ = |p|·n̂
        dE/dt = E·v[0];        d|p|/dt = |p|·v[4]
        dp⃗/dt = d|p|/dt·n̂ + |p|·dn̂/dt  (dn̂/dt as below)

    Legacy chart (pre-bounded-pT) + muon-on-shell decode. Handles both
    massless and massive via torch.where on is_massless.

    Massless (with per-type fundamental rest mass m_rest, const in t):
        E = e_ref · exp(x[0])
        |p| = sqrt(E² − m_rest²);   p⃗ = |p| · n̂
        dE/dt = E · v[0]
        d|p|/dt = E · dE/dt / |p|   (chain rule of |p|² = E² − m_rest²)
        dn̂/dt = (v[1:4] − (v[1:4]·n̂)n̂) / |r|
        dp⃗/dt = d|p|/dt · n̂ + |p| · dn̂/dt
        At m_rest = 0: |p| = E, d|p|/dt = dE/dt — collapses to lightlike form.

    Massive:
        χ = x[0]·σ_χ + μ_χ;  log m = x[4]·σ_logm + μ_logm;  m = exp(log m)
        E = m·cosh(χ);   |p| = m·sinh(χ);   p⃗ = |p|·n̂
        dχ/dt = v[0]·σ_χ;    d log m/dt = v[4]·σ_logm;  dm/dt = m·(d log m/dt)
        dE/dt = dm/dt·cosh χ + m·sinh χ · dχ/dt
        d|p|/dt = dm/dt·sinh χ + m·cosh χ · dχ/dt
        dp⃗/dt = d|p|/dt·n̂ + |p|·dn̂/dt
    """
    # Direction (shared)
    r = x[..., 1:4]
    r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    n_hat = r / r_norm

    v_r = v[..., 1:4]
    proj = (v_r * n_hat).sum(dim=-1, keepdim=True) * n_hat
    dn_dt = (v_r - proj) / r_norm  # (..., 3)

    if free_chart:
        E_f = e_ref * torch.exp(x[..., 0].clamp(-10, 10))
        p_mag_f = e_ref * torch.exp(x[..., 4].clamp(-10, 10))
        p3_f = p_mag_f.unsqueeze(-1) * n_hat
        dE_f = E_f * v[..., 0]
        dpmag_f = p_mag_f * v[..., 4]
        dp3_f = dpmag_f.unsqueeze(-1) * n_hat + p_mag_f.unsqueeze(-1) * dn_dt
        return E_f, p3_f, dE_f, dp3_f

    # Massless path: E from manifold, |p| via on-shell relation.
    s0_ml = x[..., 0].clamp(-10, 10)
    E_ml = e_ref * torch.exp(s0_ml)
    p_mag_ml = (E_ml * E_ml - m_rest * m_rest).clamp(min=1e-12).sqrt()
    p3_ml = p_mag_ml.unsqueeze(-1) * n_hat
    dE_ml = E_ml * v[..., 0]
    # d|p|/dt = E·dE/dt / |p| (chain rule). At m_rest = 0, |p| = E so this
    # collapses to dE_ml — bit-identical to the pre-mu-onshell form.
    # Floor the denominator at a physical scale (GeV; real |p| ≥ ~1 GeV via
    # E_MIN), not 1e-12: a sub-eV denominator would spike the velocity gradient
    # by ~1e12 at a degenerate low-E config. Never binds for legitimate momenta.
    dpmag_ml = E_ml * dE_ml / p_mag_ml.clamp(min=1e-3)
    dp3_ml = dpmag_ml.unsqueeze(-1) * n_hat + p_mag_ml.unsqueeze(-1) * dn_dt

    # Massive path
    chi_s = chi_sigma.clamp(min=0.01)
    logm_s = logm_sigma.clamp(min=0.01)
    chi = (x[..., 0] * chi_s + chi_mu).clamp(0, 12)
    logm_raw = (x[..., 4] * logm_s + logm_mu).clamp(-10, 10)
    m = torch.exp(logm_raw)
    cosh_chi = torch.cosh(chi)
    sinh_chi = torch.sinh(chi)
    E_mv = m * cosh_chi
    p_mag = m * sinh_chi
    p3_mv = p_mag.unsqueeze(-1) * n_hat

    dchi_dt = v[..., 0] * chi_s
    dlogm_dt = v[..., 4] * logm_s
    dm_dt = m * dlogm_dt
    dE_mv = dm_dt * cosh_chi + m * sinh_chi * dchi_dt
    dpmag_dt = dm_dt * sinh_chi + m * cosh_chi * dchi_dt
    dp3_mv = dpmag_dt.unsqueeze(-1) * n_hat + p_mag.unsqueeze(-1) * dn_dt

    # Select by particle type
    ml = is_massless.unsqueeze(-1)
    E = torch.where(is_massless, E_ml, E_mv)
    p3 = torch.where(ml, p3_ml, p3_mv)
    dE = torch.where(is_massless, dE_ml, dE_mv)
    dp3 = torch.where(ml, dp3_ml, dp3_mv)

    return E, p3, dE, dp3


class RiemannianDiTsModule(L.LightningModule):
    """Lightning module for Riemannian Flow Matching on product manifolds."""

    def __init__(
        self,
        model_config: dict,
        optimizer_config: OptimizerConfig | dict,
        scheduler_config: SchedulerConfig | dict,
        training_config: TrainingConfig | dict,
        total_steps: Optional[int] = None,
        auxiliary_loss_config: dict | None = None,
        quality_weighting_config: dict | None = None,
    ):
        super().__init__()

        if isinstance(optimizer_config, dict):
            opt_d = dict(optimizer_config)
            if "betas" in opt_d and isinstance(opt_d["betas"], list):
                opt_d["betas"] = tuple(opt_d["betas"])
            optimizer_config = OptimizerConfig(**opt_d)
        if isinstance(scheduler_config, dict):
            scheduler_config = SchedulerConfig(**scheduler_config)
        if isinstance(training_config, dict):
            training_config = TrainingConfig(**training_config)

        self.model_config = model_config
        self.direction_geometry = model_config.get("direction_geometry", "S2")
        # Chart: "onshell" (paper model) or "free" (Euclidean ablation — logE
        # and log|p| as independent coordinates, no mass-shell tie). Validated
        # against direction_geometry in the model ctor.
        self.chart = model_config.get("chart", "onshell")
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.training_config = training_config
        self.total_steps = total_steps

        self.save_hyperparameters(
            {
                "model_config": model_config,
                "optimizer_config": asdict(optimizer_config),
                "scheduler_config": asdict(scheduler_config),
                "training_config": asdict(training_config),
                "total_steps": total_steps,
                "quality_weighting_config": quality_weighting_config,
            }
        )

        # Create model — inject quality_weighting into model_config so the
        # factory can wire the (single) QualityScorer at construction time.
        model_config_with_aux = dict(model_config)
        if quality_weighting_config is not None:
            model_config_with_aux["quality_weighting"] = quality_weighting_config
        # architecture="legacy_type_film" selects the paper architecture
        # (type embedding in FiLM/adaLN conditioning, per-sector projections)
        # used by the released checkpoints; see riemannian_DiTs_model_legacy.
        # backbone="lgatr_slim" selects the L-GATr-slim blocks (ShellGATr); see
        # riemannian_DiTs_model_slimLGATr. It needs the gatr extra.
        backbone = model_config.get("backbone")
        if backbone not in (None, "lgatr_slim"):
            raise ValueError(f"unknown backbone {backbone!r}; known: 'lgatr_slim'")
        if backbone == "lgatr_slim":
            if model_config.get("architecture") is not None:
                raise ValueError("backbone 'lgatr_slim' takes no architecture key")
            from shellflow.models.riemannian_DiTs_model_slimLGATr import (
                create_riemannian_dits_slim,
            )

            self.model = create_riemannian_dits_slim(model_config_with_aux)
        elif model_config.get("architecture") == "legacy_type_film":
            from shellflow.models.riemannian_DiTs_model_legacy import (
                create_riemannian_dits as create_legacy_riemannian_dits,
            )

            self.model = create_legacy_riemannian_dits(model_config_with_aux)
        else:
            self.model = create_riemannian_dits(model_config_with_aux)

        self.sigma_prior = model_config.get("sigma_prior", 1.0)

        # Time sampling: method dispatch + distribution params
        self.t_sample_method = model_config.get("t_sample_method", "logit_normal")
        self.t_min = float(model_config.get("t_min", 0.001))
        self.t_max = float(model_config.get("t_max", 0.9995))
        self.t_sample_mu = float(model_config.get("t_sample_mu", 0.0))
        self.t_sample_sigma = float(model_config.get("t_sample_sigma", 1.0))
        self.t_power_exponent = float(model_config.get("t_power_exponent", 1.5))

        # Loss weighting by time: weight CFM loss by per-bin loss EMA
        # (stochastic interpolant-style optimal weighting)
        self.loss_weighting_t = bool(model_config.get("loss_weighting_t", False))
        self.loss_weighting_t_warmup = int(
            model_config.get("loss_weighting_t_warmup", 2000)
        )
        self.loss_weighting_t_alpha = float(
            model_config.get("loss_weighting_t_alpha", 0.01)
        )
        self.loss_weighting_t_temp = float(
            model_config.get("loss_weighting_t_temp", 1.0)
        )
        self.n_loss_weight_bins = 0

        if self.loss_weighting_t:
            n_bins = int(model_config.get("loss_weighting_t_bins", 16))
            self.n_loss_weight_bins = n_bins
            bin_edges_t = torch.linspace(self.t_min, self.t_max, n_bins + 1)
            self.register_buffer("_at_bin_edges", bin_edges_t)
            self.register_buffer("_at_loss_ema", torch.ones(n_bins))

        # Huber loss delta (controls transition from quadratic to linear)
        self.huber_delta = float(model_config.get("huber_delta", 1.0))

        # Aux head config (separate per-sector output projection aux_w/aux_b
        # for amplified log M supervision)
        ahc = model_config.get("aux_head", {})
        self.use_aux_head = ahc.get("enabled", False)
        self.aux_head_lambda_pos = float(ahc.get("lambda_pos", 1.0))
        self.aux_head_lambda_vel = float(ahc.get("lambda_vel", 1.0))
        self.aux_head_pos_t_gate = float(ahc.get("pos_t_gate", 0.3))
        # K-body extension: default k_max=2 preserves current behavior.
        # Per-K weight dict (multiplier applied on top of lambda_pos/lambda_vel).
        self.aux_head_k_max = int(ahc.get("k_max", 2))
        _kw = ahc.get("k_weights", {})
        self.aux_head_k_weights = {
            K: float(_kw.get(str(K), _kw.get(K, 1.0 if K == 2 else 0.5)))
            for K in range(2, self.aux_head_k_max + 1)
        }
        # Optional separate per-K velocity weights — defaults to k_weights so
        # existing configs are unchanged. The hierarchical doc recommends
        # tapering K>2 velocity (1.0/0.5/0.5) to reflect diminishing marginal
        # information beyond what K=2 velocity + K>2 position already provide.
        _kw_vel = ahc.get("k_weights_vel", _kw)
        self.aux_head_k_weights_vel = {
            K: float(_kw_vel.get(str(K), _kw_vel.get(K, self.aux_head_k_weights[K])))
            for K in range(2, self.aux_head_k_max + 1)
        }

        # Hierarchical aux head: replaces flat per-K-tuple mass loss with
        # decay-tree structure. Position-level uses learned heads (PairRepHead
        # → TripletRepHead → QuadRepHead → MassProj); velocity-level uses
        # analytical 4-momentum reuse (cache pair sums, add for K>2). Default
        # OFF — enables backward-compatible flat behavior.
        hier_cfg = ahc.get("hierarchical", {})
        self.use_aux_head_hierarchical = bool(hier_cfg.get("enabled", False))
        if self.use_aux_head_hierarchical and not self.use_aux_head:
            raise ValueError(
                "aux_head.hierarchical.enabled=true requires aux_head.enabled=true"
            )

        # Quality weighting is consolidated at the top-level `quality_weighting:`
        # config block (parsed below). Per-consumer toggles are gone — when the
        # top-level scorer is enabled, it is applied automatically to the
        # aux_head K-body loss and to the CFM primary loss with `cfm_q_floor`
        # softening. The legacy per-consumer `aux_head.quality_weighting` block
        # is rejected by the model __init__. `self.use_aux_head_quality` is
        # derived below (see "Quality weighting" section).

        # Reject the old config blocks explicitly — silent fall-through would
        # leave the loss machinery unweighted with no indication. Users must
        # migrate to the new `distribution_matching:` block.
        for old_key in ("density_aware", "inverse_density"):
            if old_key in ahc:
                raise ValueError(
                    f"aux_head.{old_key} block is removed. Migrate to "
                    "aux_head.distribution_matching: (see plan "
                    "ticklish-painting-sketch.md for schema)."
                )

        # ─── Distribution-matching weight w* ──────────────────────────────
        # Replaces the old M (density_aware) and w (inverse_density). A
        # single signed importance-ratio weight per K-tuple per event:
        #   w*(m_truth) = (p̂_truth / ((1-η)·p̂_gen + η·p̂_truth))^α
        # The η truth-prior mix regularizes the denominator (bounded above by
        # (1/η)^α, smooth, no 1/p̂_gen blow-up) — replaces the old clamp_min/max.
        # Truth-side: KDE built once at __init__ from training data (same
        # as before, in `_build_density_aware_buffers`).
        # Gen-side:   KDE refreshed in-place by SamplePlotCallback every
        # `every_n_epochs` after each generation pass. Initialized to a
        # clone of the truth grid so w* ≡ 1 until the first refresh.
        #
        # Applied identically on BOTH position-level and velocity-level
        # K-body aux losses with NO time-gating (w* is noise-free truth-
        # only). pos_t_gate now lives only on the position-level Huber's
        # pos_mask_f_k (its true scope).
        #
        # Config block: aux_head.distribution_matching.{enabled, alpha, eta,
        #                                                bandwidth, grid_*,
        #                                                max_events,
        #                                                max_kde_samples,
        #                                                dataloader_config}
        # Python attribute prefix: `aux_head_dm_*`
        dm_cfg = ahc.get("distribution_matching", {})
        self.use_aux_head_distribution_matching = bool(dm_cfg.get("enabled", False))
        if self.use_aux_head_distribution_matching and not self.use_aux_head:
            raise ValueError(
                "aux_head.distribution_matching.enabled=true requires "
                "aux_head.enabled=true"
            )
        self.aux_head_dm_alpha = float(dm_cfg.get("alpha", 1.0))
        # Truth-prior mix that regularizes the w* denominator (replaces the old
        # clamp_min/clamp_max): w* = (p̂_truth / ((1-η)·p̂_gen + η·p̂_truth))^α,
        # bounded above by (1/η)^α (η=0.2, α=1 → ceiling 5.0), smoothly, with no
        # 1/p̂_gen blow-up and no lower clamp.
        # eta is the truth-prior mix that caps w* at (1/eta)^alpha; eta=0 would
        # reintroduce the 1/p̂_gen blow-up the formula exists to avoid. Floor it.
        self.aux_head_dm_eta = max(float(dm_cfg.get("eta", 0.2)), 1e-6)
        # sharpness k: exponent on r = p̂_gen/p̂_truth. 1.0 = original curve;
        # k>1 steepens the w* sigmoid around r=1 (ceiling/floor/match unchanged).
        self.aux_head_dm_sharpness = float(dm_cfg.get("sharpness", 1.0))
        self.aux_head_dm_bandwidth = float(dm_cfg.get("bandwidth", 0.01))
        self.aux_head_dm_grid_size = int(dm_cfg.get("grid_size", 4096))
        self.aux_head_dm_grid_lo = float(dm_cfg.get("grid_lo", -3.0))
        self.aux_head_dm_grid_hi = float(dm_cfg.get("grid_hi", 6.0))
        self.aux_head_dm_max_events = int(dm_cfg.get("max_events", 1_000_000))
        self.aux_head_dm_max_kde_samples = int(dm_cfg.get("max_kde_samples", 200_000))
        self.aux_head_dm_dataloader_config = dm_cfg.get("dataloader_config")
        # Category-conditioned w*: each tuple keyed by type-composition +
        # charge-sign; w* uses that category's own KDE. Categories discovered
        # dynamically at build time (keys with ≥ category_min_samples get a
        # KDE; rarer keys fall back to w*=1). See tuple_categories.py.
        self.aux_head_dm_category_scheme = dm_cfg.get("category_scheme", "type_charge")
        self.aux_head_dm_category_min_samples = int(
            dm_cfg.get("category_min_samples", 2000)
        )

        # Legacy `auxiliary_loss` (K-body loss on PRIMARY v_theta) is removed.
        # The kwarg is kept ONLY so load_from_checkpoint on pre-removal
        # checkpoints (whose hparams include auxiliary_loss_config) still
        # constructs; a config that actually enables it must fail loudly.
        if auxiliary_loss_config and auxiliary_loss_config.get("enabled", False):
            raise ValueError(
                "auxiliary_loss (K-body loss on primary v_theta) has been "
                "removed. Use aux_head with hierarchical + "
                "distribution_matching instead."
            )

        # ─── Quality weighting (consolidated, single source of truth) ─────
        # Top-level `quality_weighting:` block holds the per-type sigmoid
        # params shared by ALL consumers. Master flag `enabled` controls
        # whether the scorer is instantiated at all. When enabled, the scorer
        # is applied automatically to:
        #   - the aux_head K-body loss — per-K-tuple q_tuple = ∏ q_i
        #   - CFM primary loss — per-particle q_eff = cfm_q_floor +
        #     (1 − cfm_q_floor) · q_i, multiplied into per_particle Huber.
        # The cfm_q_floor sub-knob softens raw q for CFM only (the K-body
        # consumer always uses raw q via the tuple product).
        qw_cfg = quality_weighting_config or {}
        self.use_quality_weighting = bool(qw_cfg.get("enabled", False))
        self.cfm_q_floor = float(qw_cfg.get("cfm_q_floor", 0.5))

        # Derived flag: scorer applies to the K-body consumer when it is on.
        self.use_aux_head_quality = self.use_quality_weighting and self.use_aux_head
        # CFM particleness: always on when the master flag is on (CFM primary
        # is always part of training, so no separate enable toggle is needed
        # — `cfm_q_floor=1.0` is the way to silence it without disabling the
        # K-body consumer's quality weighting).
        self.use_cfm_particleness = self.use_quality_weighting

        # K-body buffers (aux_head).
        _kbody_k_max = self.aux_head_k_max if self.use_aux_head else 0
        # Precompute fixed upper-triangle pair indices for max_particles
        if self.use_aux_head:
            N = model_config.get("max_particles", 10)
            ii, jj = torch.triu_indices(N, N, offset=1)
            self.register_buffer("_pair_i", ii)
            self.register_buffer("_pair_j", jj)
            self._n_pair_slots = ii.shape[0]  # N*(N-1)/2

        # K-body combination indices (K=3..k_max). K=2 uses _pair_i/_pair_j above.
        if self.use_aux_head:
            N = model_config.get("max_particles", 10)
            for K in range(3, _kbody_k_max + 1):
                combs = torch.combinations(torch.arange(N), r=K)
                self.register_buffer(f"_comb_{K}", combs)

        # Hierarchical pair-index buffers (only when hierarchical is enabled).
        # The hierarchy is purely analytical — pair sums of per-particle
        # 4-momenta computed once from the existing aux head output and
        # reused at K=3 (pair + single) and K=4 (canonical pair + pair).
        # For each triplet (i, j, k) with i<j<k: canonical pair = (i, j),
        # third particle = k.
        # For each quad (i, j, k, l) with i<j<k<l: canonical partition =
        # ((i, j), (k, l)). The other two partitions {(ik)(jl), (il)(jk)}
        # would produce IDENTICAL 4-momentum sums (sum is permutation-
        # invariant in the K particles), so we only need ONE per quad —
        # mathematically equivalent to the flat per-K-tuple computation.
        _needs_hierarchical = self.use_aux_head and self.use_aux_head_hierarchical
        if _needs_hierarchical:
            N = model_config.get("max_particles", 10)

            def _pair_idx(i: int, j: int) -> int:
                """Upper-triangle index for pair (i, j) with i < j."""
                return i * N - i * (i + 1) // 2 + j - i - 1

            if _kbody_k_max >= 3:
                combs_3 = self._comb_3  # (C_3, 3) sorted
                trip_pair = torch.tensor(
                    [_pair_idx(int(t[0]), int(t[1])) for t in combs_3], dtype=torch.long
                )
                self.register_buffer("_triplet_pair_idx", trip_pair)
                self.register_buffer("_triplet_third_idx", combs_3[:, 2].clone().long())

            if _kbody_k_max >= 4:
                combs_4 = self._comb_4  # (C_4, 4) sorted
                quad_pair1, quad_pair2 = [], []
                for q in combs_4:
                    i, j, k, ell = (int(x) for x in q.tolist())
                    quad_pair1.append(_pair_idx(i, j))
                    quad_pair2.append(_pair_idx(k, ell))
                self.register_buffer(
                    "_quad_pair1_idx", torch.tensor(quad_pair1, dtype=torch.long)
                )
                self.register_buffer(
                    "_quad_pair2_idx", torch.tensor(quad_pair2, dtype=torch.long)
                )

        # KDE buffers (per K). Built inline by walking the training set
        # once at module init when aux_head distribution_matching is enabled.
        # Buffers registered: `_da_log_m_grid_k{K}` (mass grid),
        # `_da_log_p_grid_k{K}` (truth log density on grid),
        # `_da_log_p_floor_k{K}` (defensive floor on log density),
        # `_da_log_p_gen_k{K}` (gen log density, initialized to truth
        # clone, updated by SamplePlotCallback).
        # Saved into checkpoints so resume / generation skip the compute.
        self._aux_head_needs_kde = self.use_aux_head_distribution_matching
        if self._aux_head_needs_kde:
            self._build_density_aware_buffers()

        # EMA — created on all ranks (required for DDP parameter broadcast),
        # but only updated on rank 0 (see on_train_batch_end)
        self.ema_decay = model_config.get("ema_decay", None)
        self.ema_model = None
        if self.ema_decay is not None:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.requires_grad_(False)
            self._ema_params = list(self.ema_model.parameters())
            self._model_params = list(self.model.parameters())

        # Type code tensor for compile-friendly loss (avoids torch.tensor() in forward)
        self.register_buffer(
            "_tc_tensor",
            torch.tensor(list(_TYPE_NAMES.keys()), dtype=torch.long),
            persistent=False,
        )

        # Tracking
        self.total_events_seen = 0
        self.events_this_epoch = 0
        self.validation_step_outputs = []

    # ------------------------------------------------------------------
    # torch.compile checkpoint compatibility
    # ------------------------------------------------------------------
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Strip _orig_mod. prefix from compiled model checkpoints."""
        sd = checkpoint.get("state_dict", {})
        fixed = {}
        needs_fix = False
        for k, v in sd.items():
            if "._orig_mod." in k:
                fixed[k.replace("._orig_mod.", ".")] = v
                needs_fix = True
            else:
                fixed[k] = v
        if needs_fix:
            checkpoint["state_dict"] = fixed

    # ------------------------------------------------------------------
    # torch.compile safety
    # ------------------------------------------------------------------
    def on_validation_start(self) -> None:
        orig = getattr(self.model, "_orig_mod", None)
        if orig is not None:
            self._compiled_model_stash = self.model
            self.model = orig

    def on_validation_end(self) -> None:
        compiled = getattr(self, "_compiled_model_stash", None)
        if compiled is not None:
            self.model = compiled
            del self._compiled_model_stash

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_time(self, B: int, device: torch.device) -> torch.Tensor:
        """Sample training time t, clamped to [t_min, t_max].

        ``t_sample_method`` selects: logit_normal | power_law | uniform.
        """
        if self.t_sample_method == "power_law":
            alpha = self.t_power_exponent
            exp = alpha + 1.0
            a = self.t_min**exp
            b = self.t_max**exp
            u = torch.rand(B, device=device)
            t = (u * (b - a) + a).pow(1.0 / exp)
        elif self.t_sample_method == "uniform":
            t = torch.rand(B, device=device) * (self.t_max - self.t_min) + self.t_min
        else:  # logit_normal (default)
            z = torch.randn(B, device=device)
            t = torch.sigmoid(self.t_sample_mu + self.t_sample_sigma * z)
        return t.clamp(self.t_min, self.t_max)

    def _compute_loss_weights(self, t: torch.Tensor) -> torch.Tensor:
        """Per-sample loss weights from EMA bins. Compilable.

        Returns weights normalized so mean = 1.  During warmup the EMA
        buffer is all-ones, so weights are naturally uniform.
        """
        bin_idx = torch.bucketize(t, self._at_bin_edges) - 1
        bin_idx = bin_idx.clamp(0, self.n_loss_weight_bins - 1)
        raw_w = self._at_loss_ema[bin_idx].pow(self.loss_weighting_t_temp)
        w = raw_w / raw_w.mean().clamp(min=1e-8)
        return w.clamp(max=5.0)

    @torch.no_grad()
    def _update_loss_weight_ema(
        self,
        t: torch.Tensor,
        per_event_loss: torch.Tensor,
    ):
        """Update EMA loss bins for loss weighting. Fully vectorized."""
        if not self.loss_weighting_t:
            return
        if self.global_step < self.loss_weighting_t_warmup:
            return
        alpha = self.loss_weighting_t_alpha

        # Assign events to bins (bucketize returns index into sorted edges)
        bin_idx = torch.bucketize(t, self._at_bin_edges) - 1
        bin_idx = bin_idx.clamp(0, self.n_loss_weight_bins - 1)

        # Per-bin mean loss via one-hot scatter
        arange = torch.arange(self.n_loss_weight_bins, device=t.device)
        one_hot = (bin_idx.unsqueeze(-1) == arange).float()  # (B, n_bins)
        counts = one_hot.sum(dim=0)  # (n_bins,)
        sums = (per_event_loss.unsqueeze(-1) * one_hot).sum(dim=0)
        has_samples = counts > 0.5
        bin_means = sums / counts.clamp(min=1)

        # EMA update only for bins that received samples
        self._at_loss_ema = torch.where(
            has_samples,
            (1 - alpha) * self._at_loss_ema + alpha * bin_means,
            self._at_loss_ema,
        )

    def _precompute_type_info(
        self,
        p_type: torch.Tensor,
        p_mask: torch.Tensor | None,
    ):
        """Compute all type-dependent lookups once per step."""
        type_idx = self.model.map_type_to_idx(p_type)
        return {
            "type_idx": type_idx,
            "is_massless": self.model._is_massless_lut[type_idx],
            "is_massive": self.model._is_massive_lut[type_idx],
            "e_ref": self.model._e_ref_lut[type_idx],
            "mass_floor": self.model._mass_floor_lut[type_idx],
            "chi_mu": self.model._chi_mu_lut[type_idx],
            "chi_sigma": self.model._chi_sigma_lut[type_idx],
            "logm_mu": self.model._logm_mu_lut[type_idx],
            "logm_sigma": self.model._logm_sigma_lut[type_idx],
            "m_rest": self.model._m_rest_lut[type_idx],
            "active": p_mask > 0.5
            if p_mask is not None
            else torch.ones_like(p_type, dtype=torch.bool),
        }

    # ------------------------------------------------------------------
    # RFM: geodesic interpolant + target velocity
    # ------------------------------------------------------------------
    def _rfm_interpolant(
        self,
        x_data: torch.Tensor,
        t: torch.Tensor,
        ti: dict,
        batch_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """RFM interpolant and target velocity on unified R × S² (× R).

        Both massless and massive share the same structure:
          dim 0: scalar (logE or χ) — linear interpolant on R
          dims 1:4: direction (n̂) — SLERP interpolant on S²
          dim 4: scalar (0 or logm) — linear interpolant on R

        Args:
            x_data: (..., 5) manifold data.
            t: (B,) time.
            ti: precomputed type info dict.
            batch_idx: (T_total,) event index per particle (geometric mode),
                       or None (padded mode).

        Returns:
            (psi_t, u_t, x_0) each (..., 5)
        """
        is_massless = ti["is_massless"]
        active = ti["active"]

        if batch_idx is not None:
            # Geometric mode: x_data is (T, 5), t is (B,)
            noise_shape = x_data.shape[:-1]  # (T,)
            t_exp = t[batch_idx]  # (T,)
        else:
            # Padded mode: x_data is (B, N, 5), t is (B,)
            B, N, _ = x_data.shape
            noise_shape = (B, N)
            t_exp = t.unsqueeze(-1).expand(B, N)  # (B, N)

        # Prior: s0 ~ N(0, σ), n ~ Uniform(S²), s4 ~ N(0, σ)
        s0_prior = (
            torch.randn(*noise_shape, device=x_data.device, dtype=x_data.dtype)
            * self.sigma_prior
        )
        n_prior = sample_uniform_sphere(x_data[..., 1:4])
        s4_prior = (
            torch.randn(*noise_shape, device=x_data.device, dtype=x_data.dtype)
            * self.sigma_prior
        )

        s0_data = x_data[..., 0]  # (B, N)
        n_data = x_data[..., 1:4]  # (B, N, 3)
        s4_data = x_data[..., 4]  # (B, N)

        # Interpolant: linear on R, SLERP on S² (or linear in R³), linear on R
        s0_psi = (1.0 - t_exp) * s0_prior + t_exp * s0_data
        if self.direction_geometry == "S2":
            n_psi = geodesic_interpolant_S2(n_prior, n_data, t_exp)
        else:
            t3 = t_exp.unsqueeze(-1)  # (..., 1)
            n_psi = (1.0 - t3) * n_prior + t3 * n_data
        s4_psi = (1.0 - t_exp) * s4_prior + t_exp * s4_data

        # Target velocity: constant on R, conditional on S² (or constant in R³), constant on R
        s0_vel = s0_data - s0_prior
        if self.direction_geometry == "S2":
            n_vel = conditional_velocity_S2(n_psi, n_data, t_exp)
        else:
            n_vel = n_data - n_prior  # constant OT velocity in R³
        s4_vel = s4_data - s4_prior

        # Zero dim 4 for massless particles (they have no logm).
        # Free chart: s4 = log|p| is live for ALL types — no zeroing.
        if self.chart == "free":
            s4_prior_z = s4_prior
        else:
            ml = is_massless  # (B, N)
            s4_psi = torch.where(ml, torch.zeros_like(s4_psi), s4_psi)
            s4_vel = torch.where(ml, torch.zeros_like(s4_vel), s4_vel)
            s4_prior_z = torch.where(ml, torch.zeros_like(s4_prior), s4_prior)

        # Pack into 5-dim and mask inactive particles
        act = active.unsqueeze(-1)  # (B, N, 1)
        psi_t = (
            torch.cat([s0_psi.unsqueeze(-1), n_psi, s4_psi.unsqueeze(-1)], dim=-1) * act
        )
        u_t = (
            torch.cat([s0_vel.unsqueeze(-1), n_vel, s4_vel.unsqueeze(-1)], dim=-1) * act
        )
        x_0 = (
            torch.cat(
                [s0_prior.unsqueeze(-1), n_prior, s4_prior_z.unsqueeze(-1)], dim=-1
            )
            * act
        )

        return psi_t, u_t, x_0

    # ------------------------------------------------------------------
    # Density-aware multiplier (aux head): inline KDE build + lookup
    # ------------------------------------------------------------------
    def _build_density_aware_buffers(self) -> None:
        """Walk the training set, fit per-K KDE on log m, register buffers.

        Called once during __init__ when aux_head distribution_matching is
        enabled. KDE settings (bandwidth, grid, dataloader) come from the
        `aux_head.distribution_matching` sub-block. The compute is
        deterministic (fixed seed, fixed dataset ordering) so all DDP ranks
        produce identical buffers in parallel.
        """
        from shellflow.data.aux_mass_kde import build_per_k_category_kde

        cfg_path = self.aux_head_dm_dataloader_config
        bandwidth = self.aux_head_dm_bandwidth
        grid_size = self.aux_head_dm_grid_size
        grid_lo = self.aux_head_dm_grid_lo
        grid_hi = self.aux_head_dm_grid_hi
        max_events = self.aux_head_dm_max_events
        max_kde_samples = self.aux_head_dm_max_kde_samples
        category_scheme = self.aux_head_dm_category_scheme
        category_min_samples = self.aux_head_dm_category_min_samples

        if cfg_path is None:
            raise ValueError(
                "distribution_matching requires "
                "distribution_matching.dataloader_config to point at the "
                "dataset config used for KDE estimation."
            )

        K_max_eff = self.aux_head_k_max

        # Category-conditioned KDE: one truth KDE per (K, category) where the
        # category = type-composition + charge-sign of the tuple. Categories
        # are discovered dynamically (keys with ≥ category_min_samples).
        self._dm_scheme = category_scheme
        self._dm_n_cat: dict[int, int] = {}

        per_k = build_per_k_category_kde(
            config_path=cfg_path,
            K_max=K_max_eff,
            bandwidth=bandwidth,
            grid_size=grid_size,
            grid_lo=grid_lo,
            grid_hi=grid_hi,
            max_events=max_events,
            max_kde_samples=max_kde_samples,
            category_min_samples=category_min_samples,
            max_particles=self.model_config.get("max_particles", 10),
        )
        for K in range(2, K_max_eff + 1):
            if K not in per_k:
                raise RuntimeError(
                    f"distribution_matching: KDE build returned no result for K={K}"
                )
            r = per_k[K]
            cats = r["categories"]  # list of {key, log_p_grid, log_p_floor, ...}
            n_cat = len(cats)
            self._dm_n_cat[K] = n_cat
            # Shared log-mass grid (x-axis) across this K's categories.
            self.register_buffer(f"_da_log_m_grid_k{K}", r["log_m_grid"])
            # Sorted int64 keys → category index = position. Used at train/gen
            # time to map a tuple's key to its buffer index (searchsorted +
            # exact-match). Empty (no category cleared threshold) → all w*=1.
            keys = torch.tensor(
                [c["key"] for c in cats], dtype=torch.int64
            )  # already ascending (build sorts by key)
            self.register_buffer(f"_da_keys_k{K}", keys)
            for c, cat in enumerate(cats):
                self.register_buffer(f"_da_log_p_grid_k{K}_c{c}", cat["log_p_grid"])
                self.register_buffer(
                    f"_da_log_p_floor_k{K}_c{c}",
                    torch.tensor(cat["log_p_floor"], dtype=torch.float32),
                )
                # Gen-side log-density (w* = (p̂_truth/p̂_gen)^α). Init = clone
                # of truth so w* ≡ 1 until the first SamplePlotCallback refresh.
                self.register_buffer(
                    f"_da_log_p_gen_k{K}_c{c}", cat["log_p_grid"].clone()
                )

    def _density_aware_log_p(
        self,
        log_m: torch.Tensor,
        K: int,
        side: str = "truth",
        cat: int = 0,
    ) -> torch.Tensor:
        """Linear interp lookup of log p̂(log m) for category `cat` of K-body mass.

        `side` selects which density grid to read:
          - "truth": `_da_log_p_grid_k{K}_c{cat}` (training data, at __init__)
          - "gen":   `_da_log_p_gen_k{K}_c{cat}`  (SamplePlotCallback refresh)

        The log-mass grid `_da_log_m_grid_k{K}` is shared across categories.

        Defensive: log_m is clamped to the grid range before searchsorted so
        out-of-range predictions return the boundary log-p value (no
        unbounded extrapolation that could produce huge multipliers).
        """
        grid = getattr(self, f"_da_log_m_grid_k{K}")
        log_p_name = (
            f"_da_log_p_grid_k{K}_c{cat}"
            if side == "truth"
            else f"_da_log_p_gen_k{K}_c{cat}"
        )
        log_p = getattr(self, log_p_name)
        floor = getattr(self, f"_da_log_p_floor_k{K}_c{cat}")
        # Clamp to grid range so out-of-range values get boundary log_p (not
        # extrapolation). Predictions can drift far during early training;
        # without this clamp, |Δ log p| could be unbounded.
        log_m_c = log_m.clamp(min=grid[0], max=grid[-1])
        idx = torch.searchsorted(grid, log_m_c).clamp(1, len(grid) - 1)
        x0 = grid[idx - 1]
        x1 = grid[idx]
        y0 = log_p[idx - 1]
        y1 = log_p[idx]
        denom = (x1 - x0).clamp(min=1e-8)
        out = y0 + (y1 - y0) * (log_m_c - x0) / denom
        return torch.maximum(out, floor)

    def _category_ids(self, keys: torch.Tensor, K: int) -> torch.Tensor:
        """Map per-tuple category keys → category index (buffer slot).

        `_da_keys_k{K}` is the sorted int64 buffer of kept category keys; the
        index of a key in that buffer is its category slot. Keys not present
        (rare compositions below the build-time threshold) get sentinel -1
        (→ w*=1 fallback at the weight site).

        Returns (same shape as `keys`) long.
        """
        kept = getattr(self, f"_da_keys_k{K}")  # (n_cat,) ascending int64
        if kept.numel() == 0:
            return torch.full_like(keys, -1)
        pos = torch.searchsorted(kept, keys).clamp(max=kept.numel() - 1)
        matched = kept[pos] == keys
        return pos.masked_fill(~matched, -1)

    def _distribution_matching_weight_categorized(
        self,
        log_m_true: torch.Tensor,
        cat_ids: torch.Tensor,
        K: int,
        *,
        alpha: float,
        eta: float,
        sharpness: float = 1.0,
    ) -> torch.Tensor:
        """Category-conditioned w*(r) = ((1-η)·r^k + η)^(-α),  r = p̂_gen/p̂_truth.

        Each tuple's w* is computed against its OWN category's truth/gen KDE
        (category = type-composition + charge-sign). Tuples whose category was
        not built (cat_id == -1, rare composition below the build threshold)
        get w* = 1 (vanilla loss).

        The denominator is regularized by mixing in η (a truth prior), which
        replaces the old hard clamp_min/clamp_max. The three anchors are fixed by
        η and α and are INDEPENDENT of the sharpness k:
          - deficit (r → 0):    w* → (1/η)^α  (bounded ceiling, no 1/p̂_gen
                                  blow-up; η=0.2, α=1 ⇒ ceiling 5.0)
          - excess  (r → ∞):    w* → 0        (smoothly down-weighted; no floor)
          - match   (r = 1):    w* = 1        (fixed point preserved)
        `sharpness` k is an exponent on the ratio r = p̂_gen/p̂_truth: in log r it
        is a logistic of slope k centred at r=1. k=1 is the original curve;
        k>1 concentrates the transition around r=1 (steeper S-curve) WITHOUT
        moving the ceiling, floor, or matched point. The bound is reached by a
        smooth saturating curve, so there is no hard clamp / clip jitter.

        Runs eagerly (the K-body loss is NOT inside the compiled core), so the
        fixed Python loop over categories is fine. n_cat is small (~tens).
        """
        w = torch.ones_like(log_m_true)
        n_cat = self._dm_n_cat.get(K, 0)
        for c in range(n_cat):
            sel = cat_ids == c
            if not bool(sel.any()):
                continue
            lp_t = self._density_aware_log_p(log_m_true, K, side="truth", cat=c)
            lp_g = self._density_aware_log_p(log_m_true, K, side="gen", cat=c)
            # w* = ((1-η)·r^k + η)^(-α),  r = p̂_gen/p̂_truth,  computed via the
            # gen/truth log-difference for stability:  r^k = exp(k·(log p̂_gen -
            # log p̂_truth)). The cap avoids fp overflow in the excess limit
            # (w* → 0 there). k=1 ⇒ identical to the original (un-sharpened) form.
            d = (sharpness * (lp_g - lp_t)).clamp(max=70.0)
            w_c = ((1.0 - eta) * torch.exp(d) + eta).pow(-alpha)
            w = torch.where(sel, w_c, w)
        return w

    def _compute_aux_head_loss(
        self,
        v_theta_aux: torch.Tensor,
        u_t: torch.Tensor,
        psi_t: torch.Tensor,
        x_data: torch.Tensor,
        t: torch.Tensor,
        ti: dict,
        p_mask: torch.Tensor,
        p_type: torch.Tensor | None = None,
        p_extra: torch.Tensor | None = None,
        p_charge: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict] | None:
        """Auxiliary K-body loss applied to v_theta_aux (separate per-sector aux head, aux_w/aux_b).

        Thin wrapper over _compute_kbody_loss using aux_head config attributes.
        """
        return self._compute_kbody_loss(
            v_pred=v_theta_aux,
            u_t=u_t,
            psi_t=psi_t,
            x_data=x_data,
            t=t,
            ti=ti,
            p_mask=p_mask,
            lambda_pos=self.aux_head_lambda_pos,
            lambda_vel=self.aux_head_lambda_vel,
            pos_t_gate=self.aux_head_pos_t_gate,
            k_max=self.aux_head_k_max,
            k_weights=self.aux_head_k_weights,
            k_weights_vel=self.aux_head_k_weights_vel,
            quality_scorer=(
                self.model.quality_scorer if self.use_aux_head_quality else None
            ),
            p_type=p_type,
            p_extra=p_extra,
            p_charge=p_charge,
            use_hierarchical=self.use_aux_head_hierarchical,
            use_distribution_matching=self.use_aux_head_distribution_matching,
            dm_alpha=self.aux_head_dm_alpha,
            dm_eta=self.aux_head_dm_eta,
            dm_sharpness=self.aux_head_dm_sharpness,
            metric_prefix="aux_head",
        )

    def _compute_kbody_loss(
        self,
        v_pred: torch.Tensor,
        u_t: torch.Tensor,
        psi_t: torch.Tensor,
        x_data: torch.Tensor,
        t: torch.Tensor,
        ti: dict,
        p_mask: torch.Tensor,
        *,
        lambda_pos: float,
        lambda_vel: float,
        pos_t_gate: float,
        k_max: int,
        k_weights: dict,
        k_weights_vel: dict,
        quality_scorer: nn.Module | None = None,
        p_type: torch.Tensor | None = None,
        p_extra: torch.Tensor | None = None,
        p_charge: torch.Tensor | None = None,
        use_hierarchical: bool = True,
        use_distribution_matching: bool = False,
        dm_alpha: float = 1.0,
        dm_eta: float = 0.2,
        dm_sharpness: float = 1.0,
        metric_prefix: str = "aux_head",
    ) -> tuple[torch.Tensor, dict] | None:
        """Hierarchical K-body Huber loss on log m_K and ds/dt (position + velocity level).

        Uses per-particle aux head velocity to:
        1. Position-level: extrapolate x_0_pred, compute K-body log m, Huber vs truth
        2. Velocity-level: compute d(log s)/dt from velocity via chain rule,
           compare to target d(log s)/dt derived from u_t

        Applied to ALL active K-tuples (massless + massive), no OSSF filtering.
        Uses full Lorentz-invariant formula: m² = (E_sum)² - |p⃗_sum|².
        Angular amplification reaches the backbone via the aux head but NOT
        the primary per-type out_w/out_b projections.

        Optional extensions (gated by config):
        - Hierarchical K-body sums (no extra params): pair sums computed once
          and reused for K=3 (pair + single) and K=4 (canonical pair + pair).
          Mathematically identical to the flat per-K-tuple sum — both produce
          the same K-body invariant mass and ds/dt. The reorganization makes
          pair-level intermediates first-class (useful for future per-level
          knobs) and saves a small amount of compute by avoiding re-summation.
        - Quality weighting (p_type + p_extra required): multiplicative
          per-particle weight q_i ∈ [0, 1] from detector features; per-tuple
          weight = ∏ q_k folds into the active-pair mask softly.
        - Distribution-matching weight w* (truth-only-keyed importance ratio):
          w*(m_truth) = (p̂_truth / ((1-η)·p̂_gen + η·p̂_truth))^α  (η = dm_eta).
          Applied IDENTICALLY to position and velocity branches with no time
          gating. Replaces the old density_aware (M) and inverse_density (w)
          mechanisms. Has the fixed-point property: at convergence
          (p̂_gen → p̂_truth), w* → 1 and the loss reduces to the unweighted
          K-body Huber. See `_distribution_matching_weight_categorized` for
          details.
        """
        # Pair mask: both active (all particle types)
        active = p_mask > 0.5
        pair_mask = active[:, self._pair_i] & active[:, self._pair_j]  # (B, P)
        if not pair_mask.any():
            # Return zero-loss with grad link to v_pred so the aux head
            # (aux_w/aux_b) is still recognized as "used" by DDP on this batch.
            return 0.0 * v_pred.sum(), {}

        is_massless = ti["is_massless"]  # (B, N)
        # Legacy chart LUTs (massless: e_ref for E = e_ref·exp(s_0);
        # massive: chi_mu/chi_sigma for χ = s_0·σ_χ + μ_χ).
        e_ref = ti["e_ref"]  # (B, N)
        chi_mu = ti["chi_mu"]  # (B, N)
        chi_sigma = ti["chi_sigma"]  # (B, N)
        logm_mu = ti["logm_mu"]  # (B, N)
        logm_sigma = ti["logm_sigma"]  # (B, N)
        m_rest = ti["m_rest"]  # (B, N) — per-type fundamental rest mass (muon-onshell)

        # Validate optional inputs against config flags
        if (quality_scorer is not None) and (p_type is None or p_extra is None):
            raise ValueError("quality_weighting requires p_type + p_extra")

        metrics = {}
        # Phantom term: 0 * v_pred.sum() ensures the aux head (aux_w/aux_b) is
        # ALWAYS in the autograd graph, regardless of whether per-K iterations
        # skip via the `if mask_f_k.sum() < 0.5: continue` guard. Without this,
        # when quality weighting reduces the soft mask sum below the threshold
        # for all K, total_loss stays at the initial scalar zero (no v_pred
        # dependency), and DDP flags aux_w/aux_b as unused → either crashes
        # (default) or dilutes its gradient via find_unused_parameters=True
        # (slow training). Zero contribution to the loss VALUE, just keeps the
        # gradient pathway alive.
        total_loss = 0.0 * v_pred.sum()
        has_vel_loss = lambda_vel > 0
        has_pos_loss = lambda_pos > 0

        # ========================================================================
        # SHARED DECODING — computed once and reused for all K values.
        # ========================================================================
        # Velocity-level: decode (E, p⃗, dE/dt, dp⃗/dt) from psi_t + velocity
        if has_vel_loss:
            E_p, p3_p, dE_p, dp3_p = _decode_particle_4mom_with_velocity(
                psi_t,
                v_pred,
                e_ref,
                is_massless,
                chi_mu,
                chi_sigma,
                logm_mu,
                logm_sigma,
                m_rest,
                free_chart=self.chart == "free",
            )
            with torch.no_grad():
                _, _, dE_tt, dp3_tt = _decode_particle_4mom_with_velocity(
                    psi_t,
                    u_t,
                    e_ref,
                    is_massless,
                    chi_mu,
                    chi_sigma,
                    logm_mu,
                    logm_sigma,
                    m_rest,
                    free_chart=self.chart == "free",
                )

        # Position-level: extrapolate x_0_pred = psi_t + (1-t)*v_pred
        # (linear in R components, exp_map on S² for direction) then decode.
        # t_gate (B, 1): only apply position loss when t > aux_head_pos_t_gate.
        # The decode is needed for BOTH flat and hierarchical paths — the
        # hierarchy is just a reorganization of how K-tuple sums are computed
        # FROM these per-particle 4-momenta.
        if has_pos_loss:
            dt_scalar = (1.0 - t).unsqueeze(-1)  # (B, 1)
            s0_pred = psi_t[..., 0] + dt_scalar * v_pred[..., 0]
            s4_pred = psi_t[..., 4] + dt_scalar * v_pred[..., 4]
            B_sh, N_sh = psi_t.shape[0], psi_t.shape[1]
            dir_flat = psi_t[..., 1:4].reshape(-1, 3)
            vel_flat = (dt_scalar.unsqueeze(-1) * v_pred[..., 1:4]).reshape(-1, 3)
            if self.direction_geometry == "S2":
                n_pred_flat = exp_map_sphere_stable(dir_flat, vel_flat)
            else:
                # R³: linear extrapolation, consistent with the flat interpolant
                n_pred_flat = dir_flat + vel_flat
            n_pred = n_pred_flat.view(B_sh, N_sh, 3)
            x_0_pred = torch.cat(
                [s0_pred.unsqueeze(-1), n_pred, s4_pred.unsqueeze(-1)],
                dim=-1,
            )
            E_pos, p3_pos = _decode_particle_4mom(
                x_0_pred,
                e_ref,
                is_massless,
                chi_mu,
                chi_sigma,
                logm_mu,
                logm_sigma,
                m_rest,
                free_chart=self.chart == "free",
            )
            with torch.no_grad():
                E_true, p3_true = _decode_particle_4mom(
                    x_data,
                    e_ref,
                    is_massless,
                    chi_mu,
                    chi_sigma,
                    logm_mu,
                    logm_sigma,
                    m_rest,
                    free_chart=self.chart == "free",
                )
            pos_t_gate = (t > pos_t_gate).float().unsqueeze(-1)  # (B, 1)

        # Hierarchical analytical pair sums — computed ONCE per side
        # (predicted, truth, target-velocity) and reused at K=3, K=4.
        # All quantities are exact tensor reductions, no learned parameters.
        if use_hierarchical:
            if has_vel_loss:
                # Velocity-level pair sums (predicted side)
                E_pair_p = E_p[:, self._pair_i] + E_p[:, self._pair_j]  # (B, C_2)
                p3_pair_p = p3_p[:, self._pair_i] + p3_p[:, self._pair_j]
                dE_pair_p = dE_p[:, self._pair_i] + dE_p[:, self._pair_j]
                dp3_pair_p = dp3_p[:, self._pair_i] + dp3_p[:, self._pair_j]
                # Truth-side velocity sums
                with torch.no_grad():
                    dE_pair_tt = dE_tt[:, self._pair_i] + dE_tt[:, self._pair_j]
                    dp3_pair_tt = dp3_tt[:, self._pair_i] + dp3_tt[:, self._pair_j]
            if has_pos_loss:
                # Position-level pair sums
                E_pair_pos = E_pos[:, self._pair_i] + E_pos[:, self._pair_j]
                p3_pair_pos = p3_pos[:, self._pair_i] + p3_pos[:, self._pair_j]
                with torch.no_grad():
                    E_pair_true = E_true[:, self._pair_i] + E_true[:, self._pair_j]
                    p3_pair_true = p3_true[:, self._pair_i] + p3_true[:, self._pair_j]

        # Per-particle quality scores for tuple weighting. q ∈ [0, 1], q=1 for
        # inactive / unknown types so the active-particle mask handles exclusion.
        # Computed once from truth p3 (no_grad) so the weight is stable across
        # training and doesn't couple with model predictions.
        q_per_particle = None
        if quality_scorer is not None:
            # Reuse the truth decode from the position-level block when available
            # to avoid a redundant decode call.
            if has_pos_loss:
                p3_true_q = p3_true
            else:
                with torch.no_grad():
                    _, p3_true_q = _decode_particle_4mom(
                        x_data,
                        e_ref,
                        is_massless,
                        chi_mu,
                        chi_sigma,
                        logm_mu,
                        logm_sigma,
                        m_rest,
                        free_chart=self.chart == "free",
                    )
            q_per_particle = quality_scorer(
                p_type, p_extra, p3_true_q, active
            )  # (B, N)

        # ========================================================================
        # K-body loop: K=2, 3, ..., k_max, all using the shared decoded tensors.
        # Each K computes:
        #   m² = (E_sum)² - |p_sum|²                           (position-level)
        #   ds/dt = 2·E_sum·dE_sum/dt - 2·p_sum·dp_sum/dt      (velocity-level)
        # Applied to ALL K-tuples of active particles (no type/charge filter).
        # ========================================================================
        for K in range(2, k_max + 1):
            wK_pos = k_weights.get(K, 1.0 if K == 2 else 0.5)
            wK_vel = k_weights_vel.get(K, wK_pos)

            # Build K-tuple index tensor and active-mask for this K
            if K == 2:
                # Use precomputed _pair_i, _pair_j (upper-triangle pairs)
                # Flatten as (C_2, 2) to match combs format
                combs = torch.stack([self._pair_i, self._pair_j], dim=-1)  # (C_2, 2)
            else:
                combs = getattr(self, f"_comb_{K}")  # (C_K, K)

            # Active mask: all K particles in the tuple must be active
            pair_mask_k = active[:, combs].all(dim=-1)  # (B, C_K)
            mask_f_k = pair_mask_k.float()

            # Quality weighting: multiply mask by per-tuple product of qualities.
            # q = 1 for inactive particles so the active-mask above remains the
            # only HARD exclusion; quality weighting is purely soft.
            if q_per_particle is not None:
                q_tuple_k = q_per_particle[:, combs].prod(dim=-1)  # (B, C_K)
                mask_f_k = mask_f_k * q_tuple_k

            if mask_f_k.sum() < 0.5:
                continue

            # ---- Distribution-matching weight w* — computed once per K ----
            # Single signed importance ratio applied identically on both
            # branches with NO time gating:
            #   w*(m_truth) = (p̂_truth / ((1-η)·p̂_gen + η·p̂_truth))^α
            # Replaces the old M (density_aware) drift penalty and w
            # (inverse_density) truth-rebalancer. At convergence (p̂_gen →
            # p̂_truth), w* → 1 and the loss reduces to the unweighted
            # K-body Huber (fixed-point property).
            w_star_k = None
            if use_distribution_matching:
                # Compute truth-side log m_K once. Reuse hierarchical pair
                # sums when available (mirrors the position branch's
                # hierarchical dispatch below); otherwise sum per K-tuple.
                with torch.no_grad():
                    if use_hierarchical and K == 2:
                        E_sum_kt_dm = E_pair_true
                        p3_sum_kt_dm = p3_pair_true
                    elif use_hierarchical and K == 3:
                        E_sum_kt_dm = (
                            E_pair_true[:, self._triplet_pair_idx]
                            + E_true[:, self._triplet_third_idx]
                        )
                        p3_sum_kt_dm = (
                            p3_pair_true[:, self._triplet_pair_idx]
                            + p3_true[:, self._triplet_third_idx]
                        )
                    elif use_hierarchical and K == 4:
                        E_sum_kt_dm = (
                            E_pair_true[:, self._quad_pair1_idx]
                            + E_pair_true[:, self._quad_pair2_idx]
                        )
                        p3_sum_kt_dm = (
                            p3_pair_true[:, self._quad_pair1_idx]
                            + p3_pair_true[:, self._quad_pair2_idx]
                        )
                    else:
                        E_sum_kt_dm = E_true[:, combs].sum(dim=-1)
                        p3_sum_kt_dm = p3_true[:, combs].sum(dim=-2)
                    m2_true_k_dm = E_sum_kt_dm.pow(2) - (
                        p3_sum_kt_dm * p3_sum_kt_dm
                    ).sum(dim=-1)
                    logm_true_k_dm = 0.5 * torch.log(m2_true_k_dm.clamp(min=1e-4))
                    # Category-conditioned w*: each tuple keyed by its
                    # type-composition + charge-sign, weighted against that
                    # category's own truth/gen KDE. Requires p_type + p_charge.
                    if p_type is None or p_charge is None:
                        raise ValueError(
                            "distribution_matching requires p_type and p_charge "
                            "(category keys are computed from type + charge)."
                        )
                    keys_k = tuple_keys(
                        p_type[:, combs], p_charge[:, combs], K
                    )  # (B, C_K)
                    cat_ids_k = self._category_ids(keys_k, K)  # (B, C_K)
                    w_star_k = self._distribution_matching_weight_categorized(
                        logm_true_k_dm,
                        cat_ids_k,
                        K,
                        alpha=dm_alpha,
                        eta=dm_eta,
                        sharpness=dm_sharpness,
                    )

            # ---- Velocity-level K-body loss ----
            if has_vel_loss:
                # Hierarchical: reuse pair sums for K=3, K=4 (mathematically
                # identical to the flat per-K-tuple sum).
                if use_hierarchical and K == 3:
                    E_sum_k = (
                        E_pair_p[:, self._triplet_pair_idx]
                        + E_p[:, self._triplet_third_idx]
                    )
                    p3_sum_k = (
                        p3_pair_p[:, self._triplet_pair_idx]
                        + p3_p[:, self._triplet_third_idx]
                    )
                    dE_sum_k = (
                        dE_pair_p[:, self._triplet_pair_idx]
                        + dE_p[:, self._triplet_third_idx]
                    )
                    dp3_sum_k = (
                        dp3_pair_p[:, self._triplet_pair_idx]
                        + dp3_p[:, self._triplet_third_idx]
                    )
                    with torch.no_grad():
                        dE_sum_tk = (
                            dE_pair_tt[:, self._triplet_pair_idx]
                            + dE_tt[:, self._triplet_third_idx]
                        )
                        dp3_sum_tk = (
                            dp3_pair_tt[:, self._triplet_pair_idx]
                            + dp3_tt[:, self._triplet_third_idx]
                        )
                elif use_hierarchical and K == 4:
                    E_sum_k = (
                        E_pair_p[:, self._quad_pair1_idx]
                        + E_pair_p[:, self._quad_pair2_idx]
                    )
                    p3_sum_k = (
                        p3_pair_p[:, self._quad_pair1_idx]
                        + p3_pair_p[:, self._quad_pair2_idx]
                    )
                    dE_sum_k = (
                        dE_pair_p[:, self._quad_pair1_idx]
                        + dE_pair_p[:, self._quad_pair2_idx]
                    )
                    dp3_sum_k = (
                        dp3_pair_p[:, self._quad_pair1_idx]
                        + dp3_pair_p[:, self._quad_pair2_idx]
                    )
                    with torch.no_grad():
                        dE_sum_tk = (
                            dE_pair_tt[:, self._quad_pair1_idx]
                            + dE_pair_tt[:, self._quad_pair2_idx]
                        )
                        dp3_sum_tk = (
                            dp3_pair_tt[:, self._quad_pair1_idx]
                            + dp3_pair_tt[:, self._quad_pair2_idx]
                        )
                elif use_hierarchical and K == 2:
                    # Pair sums are exactly the K=2 K-body sums.
                    E_sum_k = E_pair_p
                    p3_sum_k = p3_pair_p
                    dE_sum_k = dE_pair_p
                    dp3_sum_k = dp3_pair_p
                    with torch.no_grad():
                        dE_sum_tk = dE_pair_tt
                        dp3_sum_tk = dp3_pair_tt
                else:
                    # Flat per-K-tuple computation (existing).
                    E_sum_k = E_p[:, combs].sum(dim=-1)  # (B, C_K)
                    p3_sum_k = p3_p[:, combs].sum(dim=-2)
                    dE_sum_k = dE_p[:, combs].sum(dim=-1)
                    dp3_sum_k = dp3_p[:, combs].sum(dim=-2)
                    with torch.no_grad():
                        dE_sum_tk = dE_tt[:, combs].sum(dim=-1)
                        dp3_sum_tk = dp3_tt[:, combs].sum(dim=-2)

                s_k = E_sum_k.pow(2) - (p3_sum_k * p3_sum_k).sum(dim=-1)
                s_safe_k = s_k.clamp(min=1e-4)
                ds_dt_p_k = 2.0 * E_sum_k * dE_sum_k - 2.0 * (p3_sum_k * dp3_sum_k).sum(
                    dim=-1
                )
                with torch.no_grad():
                    ds_dt_t_k = 2.0 * E_sum_k * dE_sum_tk - 2.0 * (
                        p3_sum_k * dp3_sum_tk
                    ).sum(dim=-1)

                dlog_s_pred_k = ds_dt_p_k / s_safe_k
                dlog_s_true_k = ds_dt_t_k / s_safe_k.detach()

                per_k_vel = (
                    F.huber_loss(
                        dlog_s_pred_k,
                        dlog_s_true_k.detach(),
                        reduction="none",
                        delta=self.huber_delta,
                    ).clamp(max=10.0)
                    * mask_f_k
                )

                # Apply w* (distribution-matching weight) — no time gating.
                # w* depends only on truth-side m_K (both p̂_truth and p̂_gen
                # are evaluated at m_K_truth), so it's noise-free at every t.
                # `pos_t_gate` does NOT appear on the velocity branch in the
                # new design — its sole remaining use is gating per_k_pos
                # via pos_mask_f_k on the position branch below.
                if w_star_k is not None:
                    per_k_vel = per_k_vel * w_star_k

                ev_cnt_k = mask_f_k.sum(dim=1).clamp(min=1)
                ev_mean_k = per_k_vel.sum(dim=1) / ev_cnt_k
                has_k = mask_f_k.sum(dim=1) > 0.5
                if has_k.any():
                    vel_loss_k = ev_mean_k[has_k].sum() / has_k.sum().clamp(min=1)
                    total_loss = total_loss + wK_vel * lambda_vel * vel_loss_k
                    if K == 2:
                        metrics[f"loss/{metric_prefix}_vel"] = vel_loss_k.detach()

            # ---- Position-level K-body loss ----
            if has_pos_loss:
                pos_mask_f_k = mask_f_k * pos_t_gate
                if pos_mask_f_k.sum() >= 0.5:
                    # Compute K-tuple 4-momentum sums.
                    # Hierarchical: reuse pair sums (mathematically identical
                    # to flat per-K-tuple sums; just less recomputation).
                    if use_hierarchical and K == 2:
                        E_sum_kp = E_pair_pos
                        p3_sum_kp = p3_pair_pos
                        with torch.no_grad():
                            E_sum_kt = E_pair_true
                            p3_sum_kt = p3_pair_true
                    elif use_hierarchical and K == 3:
                        E_sum_kp = (
                            E_pair_pos[:, self._triplet_pair_idx]
                            + E_pos[:, self._triplet_third_idx]
                        )
                        p3_sum_kp = (
                            p3_pair_pos[:, self._triplet_pair_idx]
                            + p3_pos[:, self._triplet_third_idx]
                        )
                        with torch.no_grad():
                            E_sum_kt = (
                                E_pair_true[:, self._triplet_pair_idx]
                                + E_true[:, self._triplet_third_idx]
                            )
                            p3_sum_kt = (
                                p3_pair_true[:, self._triplet_pair_idx]
                                + p3_true[:, self._triplet_third_idx]
                            )
                    elif use_hierarchical and K == 4:
                        E_sum_kp = (
                            E_pair_pos[:, self._quad_pair1_idx]
                            + E_pair_pos[:, self._quad_pair2_idx]
                        )
                        p3_sum_kp = (
                            p3_pair_pos[:, self._quad_pair1_idx]
                            + p3_pair_pos[:, self._quad_pair2_idx]
                        )
                        with torch.no_grad():
                            E_sum_kt = (
                                E_pair_true[:, self._quad_pair1_idx]
                                + E_pair_true[:, self._quad_pair2_idx]
                            )
                            p3_sum_kt = (
                                p3_pair_true[:, self._quad_pair1_idx]
                                + p3_pair_true[:, self._quad_pair2_idx]
                            )
                    else:
                        # Flat per-K-tuple computation (existing).
                        E_sum_kp = E_pos[:, combs].sum(dim=-1)
                        p3_sum_kp = p3_pos[:, combs].sum(dim=-2)
                        with torch.no_grad():
                            E_sum_kt = E_true[:, combs].sum(dim=-1)
                            p3_sum_kt = p3_true[:, combs].sum(dim=-2)

                    m2_pred_k = E_sum_kp.pow(2) - (p3_sum_kp * p3_sum_kp).sum(dim=-1)
                    logm_pred_k = 0.5 * torch.log(m2_pred_k.clamp(min=1e-4))
                    with torch.no_grad():
                        m2_true_k = E_sum_kt.pow(2) - (p3_sum_kt * p3_sum_kt).sum(
                            dim=-1
                        )
                        logm_true_k = 0.5 * torch.log(m2_true_k.clamp(min=1e-4))

                    per_k_pos = F.huber_loss(
                        logm_pred_k,
                        logm_true_k,
                        reduction="none",
                        delta=self.huber_delta,
                    )

                    # Apply w* (distribution-matching weight) — same factor
                    # as the velocity branch above, applied identically here.
                    # Note: w_star_k is None iff distribution_matching is
                    # disabled.
                    if w_star_k is not None:
                        per_k_pos = per_k_pos * w_star_k

                    per_k_pos = per_k_pos * pos_mask_f_k

                    ev_cnt_kp = pos_mask_f_k.sum(dim=1).clamp(min=1)
                    ev_mean_kp = per_k_pos.sum(dim=1) / ev_cnt_kp
                    has_kp = pos_mask_f_k.sum(dim=1) > 0.5
                    if has_kp.any():
                        pos_loss_k = ev_mean_kp[has_kp].sum() / has_kp.sum().clamp(
                            min=1
                        )
                        total_loss = total_loss + wK_pos * lambda_pos * pos_loss_k
                        if K == 2:
                            metrics[f"loss/{metric_prefix}_pos"] = pos_loss_k.detach()

        return total_loss, metrics

    def _compute_cfm_loss(
        self,
        v_theta: torch.Tensor,
        u_t: torch.Tensor,
        ti: dict,
        p_type: torch.Tensor,
        t: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
        B: int | None = None,
        kin: torch.Tensor | None = None,
        p_extra: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute CFM velocity Huber loss — unified for R × S² (× R).

        All particles: Huber on dim 0 + per-component Huber on dims 1:4 (S²)
        Massive only: + Huber on dim 4 (logm)
        Massless dim 4 is zero in both prediction and target.

        When `quality_weighting.enabled=true` and `kin`/`p_extra` are passed,
        applies CFM particleness weighting: per-particle Huber is multiplied
        by `q_eff = cfm_q_floor + (1 − cfm_q_floor) · q_i`, where q_i is the
        per-particle quality score from QualityScorer. `cfm_q_floor=1.0`
        silences this without disabling the K-body consumer's quality weight.
        """
        loss_weight = None
        if t is not None and self.loss_weighting_t and self.training:
            if batch_idx is not None:
                t_expanded = t[batch_idx]  # (B,) → (T,)
            else:
                t_expanded = t  # (B,) for padded mode
            w = self._compute_loss_weights(t_expanded)
            if batch_idx is None:
                w = w.unsqueeze(-1)  # (B, 1) for broadcast over (B, N)
            loss_weight = w

        B_ev = B if B is not None else v_theta.shape[0]
        loss, type_means, type_counts = self._compute_cfm_loss_core(
            v_theta,
            u_t,
            ti,
            p_type,
            loss_weight=loss_weight,
            batch_idx=batch_idx,
            B=B_ev,
            kin=kin,
            p_extra=p_extra,
        )

        # Extract metrics outside the compilable core (has .item()-like accesses)
        metrics = self._extract_loss_metrics(loss, type_means, type_counts, ti)
        return loss, metrics

    def _compute_cfm_loss_core(
        self,
        v_theta: torch.Tensor,
        u_t: torch.Tensor,
        ti: dict,
        p_type: torch.Tensor,
        loss_weight: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
        B: int | None = None,
        kin: torch.Tensor | None = None,
        p_extra: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compilable core of CFM loss with per-event normalisation.

        Per-particle Huber losses are averaged within each event first,
        then averaged across events per type, then across present types.
        Each event contributes equally regardless of particle multiplicity.
        Compilable: no .item(), no Python loops, no asserts.
        """
        is_massless = ti["is_massless"]
        active = ti["active"]

        _HUBER_DELTA = self.huber_delta
        dv_s0 = F.huber_loss(
            v_theta[..., 0], u_t[..., 0], reduction="none", delta=_HUBER_DELTA
        )
        dv_s2 = F.huber_loss(
            v_theta[..., 1:4], u_t[..., 1:4], reduction="none", delta=_HUBER_DELTA
        ).sum(dim=-1)
        dv_s4 = F.huber_loss(
            v_theta[..., 4], u_t[..., 4], reduction="none", delta=_HUBER_DELTA
        )
        # Free chart: s4 = log|p| is a live coordinate for all types — keep its
        # loss term. On-shell chart: massless types have no s4, zero it out.
        if self.chart != "free":
            dv_s4 = torch.where(is_massless, torch.zeros_like(dv_s4), dv_s4)

        per_particle = torch.where(
            active, dv_s0 + dv_s2 + dv_s4, torch.zeros_like(dv_s0)
        )
        per_particle = torch.nan_to_num(per_particle, nan=0.0, posinf=100.0, neginf=0.0)
        per_particle = per_particle.clamp(max=100.0)

        # Stochastic interpolant loss weighting: scale per-particle loss by w(t)
        if loss_weight is not None:
            per_particle = per_particle * loss_weight

        # CFM particleness: soften per-particle Huber by detector quality.
        # q_eff = cfm_q_floor + (1 − cfm_q_floor) · q_i ∈ [cfm_q_floor, 1].
        # Low-quality particles (failing iso/IP/JVT/etc.) get down-weighted in
        # the primary CFM loss so the model is steered toward producing the
        # detector-grade particles seen by quality-cut analyses.
        if (
            self.use_cfm_particleness
            and self.model.quality_scorer is not None
            and kin is not None
            and p_extra is not None
        ):
            with torch.no_grad():
                p3_true = kin[..., :3]
                q_per_particle = self.model.quality_scorer(
                    p_type, p_extra, p3_true, active
                )
                q_eff = self.cfm_q_floor + (1.0 - self.cfm_q_floor) * q_per_particle
            per_particle = per_particle * q_eff

        # Per-event, per-type aggregation
        N_types = self._tc_tensor.shape[0]
        type_match = (
            (p_type.unsqueeze(-1) == self._tc_tensor) & active.unsqueeze(-1)
        ).float()

        if batch_idx is not None:
            # Flat (T,) tensors — scatter into (B, N_types)
            ev_type_sum = torch.zeros(
                B, N_types, device=per_particle.device, dtype=per_particle.dtype
            )
            ev_type_cnt = torch.zeros(
                B, N_types, device=per_particle.device, dtype=per_particle.dtype
            )
            expand_idx = batch_idx.unsqueeze(-1).expand_as(type_match)
            ev_type_sum.scatter_add_(
                0, expand_idx, per_particle.unsqueeze(-1) * type_match
            )
            ev_type_cnt.scatter_add_(0, expand_idx, type_match)
        else:
            # Padded (B, N) tensors — sum over particle dim
            ev_type_sum = (per_particle.unsqueeze(-1) * type_match).sum(
                dim=1
            )  # (B, N_types)
            ev_type_cnt = type_match.sum(dim=1)  # (B, N_types)

        # Per-event mean, then average across events per type
        ev_type_mean = ev_type_sum / ev_type_cnt.clamp(min=1)  # (B, N_types)
        ev_has_type = (ev_type_cnt > 0.5).float()  # (B, N_types)
        type_means = (ev_type_mean * ev_has_type).sum(dim=0) / ev_has_type.sum(
            dim=0
        ).clamp(min=1)
        type_counts = ev_type_cnt.sum(dim=0)  # total particles per type (for logging)

        present = (type_counts > 0.5).float()
        n_present = present.sum().clamp(min=1)

        loss = (type_means * present).sum() / n_present

        return loss, type_means, type_counts

    def _extract_loss_metrics(
        self,
        loss: torch.Tensor,
        type_means: torch.Tensor,
        type_counts: torch.Tensor,
        ti: dict,
    ) -> dict:
        """Extract logging metrics — not compiled, runs after the graph."""
        type_names = list(_TYPE_NAMES.values())
        present = type_counts > 0.5

        metrics: dict = {
            "n_active": ti["active"].sum().clamp(min=1).float().detach(),
            "loss": loss.detach(),
        }
        for i, name in enumerate(type_names):
            if present[i]:
                metrics[f"loss/{name}"] = type_means[i].detach()

        return metrics

    def _single_step_endpoint(
        self,
        psi_t: torch.Tensor,
        v_theta: torch.Tensor,
        t: torch.Tensor,
        ti: dict,
        batch_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Recover approximate x̂₁ via single-step extrapolation on R × S² (× R).

        Unified for both massless and massive particles.
        Works on both (B, N, 5) padded and (T, 5) geometric shapes.
        """
        if batch_idx is not None:
            # Geometric: t is (B,), expand to per-particle (T,)
            dt_s = (1.0 - t[batch_idx]).clamp(min=0.0)  # (T,)
            dt_n = dt_s.unsqueeze(-1)  # (T, 1) for (T, 3) broadcast
        else:
            # Padded: t is (B,), need (B, 1) for (B, N) broadcast
            dt_s = (1.0 - t).unsqueeze(-1).clamp(min=0.0)  # (B, 1)
            dt_n = dt_s.unsqueeze(-1)  # (B, 1, 1) for (B, N, 3) broadcast

        s0_1 = psi_t[..., 0] + dt_s * v_theta[..., 0]
        if self.direction_geometry == "S2":
            n_1 = exp_map_sphere_stable(psi_t[..., 1:4], dt_n * v_theta[..., 1:4])
        else:
            n_1 = psi_t[..., 1:4] + dt_n * v_theta[..., 1:4]
        s4_1 = psi_t[..., 4] + dt_s * v_theta[..., 4]

        x1_hat = torch.cat([s0_1.unsqueeze(-1), n_1, s4_1.unsqueeze(-1)], dim=-1)
        # Zero out inactive particles
        x1_hat = x1_hat * ti["active"].unsqueeze(-1)
        return x1_hat

    def _riemannian_euler_step(
        self,
        phi: torch.Tensor,
        v: torch.Tensor,
        dt: float | torch.Tensor,
        ti: dict,
    ) -> torch.Tensor:
        """One Euler step on R × S² (× R): flat + S² exp_map + flat."""
        # dt can be scalar or (B,) or (B, 1)
        if isinstance(dt, (int, float)):
            dt_s = dt  # scalar for dim 0 and 4
            dt_n = dt  # scalar for S² tangent
        else:
            dt_s = dt if dt.dim() <= 1 else dt.squeeze(-1)  # (B,) or scalar
            if dt_s.dim() == 1:
                dt_s = dt_s.unsqueeze(-1)  # (B, 1)
            dt_n = dt_s.unsqueeze(-1)  # (B, 1, 1)

        s0_next = phi[..., 0] + dt_s * v[..., 0]
        if self.direction_geometry == "S2":
            n_next = exp_map_sphere_stable(phi[..., 1:4], dt_n * v[..., 1:4])
        else:
            n_next = phi[..., 1:4] + dt_n * v[..., 1:4]
        s4_next = phi[..., 4] + dt_s * v[..., 4]

        phi_next = torch.cat(
            [s0_next.unsqueeze(-1), n_next, s4_next.unsqueeze(-1)], dim=-1
        )
        # Inactive particles keep original state
        act = ti["active"].unsqueeze(-1)
        return torch.where(act, phi_next, phi)

    def _pbfm_unroll_endpoint(
        self,
        psi_t: torch.Tensor,
        t: torch.Tensor,
        ti: dict,
        K: int,
        **model_kwargs,
    ) -> torch.Tensor:
        """PBFM-style K-step ODE unrolling from ψ_t to approximate x₁.

        Only the final velocity evaluation carries gradient.
        All intermediate states and velocities are detached.
        """
        p_type = model_kwargs["p_type"]
        p_charge = model_kwargs["p_charge"]
        p_extra = model_kwargs["p_extra"]
        p_mask = model_kwargs["p_mask"]
        met_vector = model_kwargs.get("met_vector")

        phi = psi_t.detach()

        for k in range(K):
            # Time for this sub-step
            tau_k = t + k * (1.0 - t) / K  # (B,)
            dt_k = (1.0 - t) / K  # (B,) step size

            if k < K - 1:
                # Intermediate: fully detached (no gradient)
                with torch.no_grad():
                    v_k = self.model(
                        phi,
                        p_type,
                        p_charge,
                        p_extra,
                        tau_k,
                        p_mask,
                        met_vector=met_vector,
                    ).float()
                    phi = self._riemannian_euler_step(
                        phi,
                        v_k,
                        dt_k.unsqueeze(-1),
                        ti,
                    )
            else:
                # Final step: gradient flows through v_θ
                phi = phi.detach()
                v_k = self.model(
                    phi,
                    p_type,
                    p_charge,
                    p_extra,
                    tau_k,
                    p_mask,
                    met_vector=met_vector,
                ).float()
                phi = self._riemannian_euler_step(
                    phi,
                    v_k,
                    dt_k.unsqueeze(-1),
                    ti,
                )

        return phi

    # ------------------------------------------------------------------
    # Kinematic diagnostics (no grad, every step)
    # ------------------------------------------------------------------

    def _decode_manifold_to_physical(
        self,
        pred: torch.Tensor,
        ti: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode manifold representation to physical (px, py, pz, E) in GeV.

        Legacy chart:
          Massless: (logE_norm, nx, ny, nz, 0) — R × S²; E = E_ref·exp(s_0); p⃗ = E·n̂.
          Massive : (χ_norm, nx, ny, nz, logm_norm) — R × S² × R (Log H³_1);
                    E = m·cosh χ; |p⃗| = m·sinh χ.
        """
        is_massless = ti["is_massless"]
        e_ref = ti["e_ref"]
        m_rest = ti["m_rest"]  # (B, N) per-type fundamental rest mass

        n_pred = F.normalize(pred[..., 1:4], dim=-1, eps=1e-6)

        if self.chart == "free":
            # Free Euclidean chart: E and |p| independent, no on-shell tie.
            E_free = e_ref * torch.exp(pred[..., 0].clamp(-10, 10))
            p_mag_free = e_ref * torch.exp(pred[..., 4].clamp(-10, 10))
            px = p_mag_free * n_pred[..., 0]
            py = p_mag_free * n_pred[..., 1]
            pz = p_mag_free * n_pred[..., 2]
            return px, py, pz, E_free

        # Massless (with muon-onshell): E = E_ref·exp(s_0); |p| = sqrt(E²−m_rest²);
        # p⃗ = |p|·n̂. m_rest = 0 for photon/composite collapses to lightlike form.
        E_ml = e_ref * torch.exp(pred[..., 0].clamp(-10, 10))
        p_mag_ml = (E_ml * E_ml - m_rest * m_rest).clamp(min=0.0).sqrt()
        px_ml = p_mag_ml * n_pred[..., 0]
        py_ml = p_mag_ml * n_pred[..., 1]
        pz_ml = p_mag_ml * n_pred[..., 2]

        # Massive: Log H³_1 → E = m*cosh(χ), |p| = m*sinh(χ), p⃗ = |p|*n̂
        chi = pred[..., 0] * ti["chi_sigma"].clamp(min=0.01) + ti["chi_mu"]
        logm_raw = pred[..., 4] * ti["logm_sigma"].clamp(min=0.01) + ti["logm_mu"]
        m_phys = torch.exp(logm_raw.clamp(-10, 10))
        E_mv = m_phys * torch.cosh(chi.clamp(0, 12))
        p_mag = m_phys * torch.sinh(chi.clamp(0, 12))
        px_mv = p_mag * n_pred[..., 0]
        py_mv = p_mag * n_pred[..., 1]
        pz_mv = p_mag * n_pred[..., 2]

        px = torch.where(is_massless, px_ml, px_mv)
        py = torch.where(is_massless, py_ml, py_mv)
        pz = torch.where(is_massless, pz_ml, pz_mv)
        E = torch.where(is_massless, E_ml, E_mv)
        return px, py, pz, E

    @staticmethod
    def _bc_torch(p: torch.Tensor, t: torch.Tensor, n_bins: int = 100) -> torch.Tensor:
        """Bray-Curtis dissimilarity between two 1-D sample tensors.

            BC = Σ_b |c_p(b) − c_t(b)| / (N_p + N_t)

        Range [0, 1]. Captures count-mismatch that W1 normalises away.
        Matches the formula in callback._compute_bray_curtis (NumPy).
        """
        if p.numel() == 0 or t.numel() == 0:
            return torch.tensor(float("nan"), device=p.device)
        lo = float(torch.minimum(p.min(), t.min()).item())
        hi = float(torch.maximum(p.max(), t.max()).item())
        if hi <= lo:
            return torch.zeros((), device=p.device)
        p_hist = torch.histc(p, bins=n_bins, min=lo, max=hi)
        t_hist = torch.histc(t, bins=n_bins, min=lo, max=hi)
        denom = (p_hist.sum() + t_hist.sum()).clamp(min=1.0)
        return (p_hist - t_hist).abs().sum() / denom

    @torch.no_grad()
    def _compute_kinematic_metrics(
        self,
        pred: torch.Tensor,
        x_data: torch.Tensor,
        ti: dict,
        p_type: torch.Tensor,
        p_charge: torch.Tensor,
        met_vector: torch.Tensor | None,
        batch_idx: torch.Tensor | None = None,
        ptr: torch.Tensor | None = None,
    ) -> dict:
        """Per-particle resolution/shape + event-level OSSF metrics.

        Works on both padded (B, N, ...) and geometric (T_total, ...) tensors.

        Naming: {particle_type}_{observable}_{metric}
        """
        active = ti["active"]
        geometric = batch_idx is not None

        if geometric:
            B = ptr.shape[0] - 1
        else:
            B, N = p_type.shape

        # ── Decode PRED and TRUTH → (px, py, pz, E) in GeV ──
        px_p, py_p, pz_p, E_p = self._decode_manifold_to_physical(pred, ti)
        px_t, py_t, pz_t, E_t = self._decode_manifold_to_physical(x_data, ti)

        # ── Derived observables ──
        pT_p = torch.sqrt(px_p**2 + py_p**2 + 1e-8)
        pT_t = torch.sqrt(px_t**2 + py_t**2 + 1e-8)
        logpT_p = torch.log(pT_p.clamp(min=0.1))
        logpT_t = torch.log(pT_t.clamp(min=0.1))
        phi_p = torch.atan2(py_p, px_p)
        phi_t = torch.atan2(py_t, px_t)
        eta_p = torch.asinh(pz_p / pT_p)
        eta_t = torch.asinh(pz_t / pT_t)
        logE_p = torch.log(E_p.clamp(min=0.1))
        logE_t = torch.log(E_t.clamp(min=0.1))

        metrics: dict = {}
        agg_pT_p, agg_pT_t = [], []
        agg_eta_p, agg_eta_t = [], []
        agg_phi_p, agg_phi_t = [], []
        agg_E_p, agg_E_t = [], []

        # Precompute type masks and counts — single GPU→CPU transfer
        type_codes_list = list(_TYPE_NAMES.keys())
        type_names_list = list(_TYPE_NAMES.values())
        tc_t = self._tc_tensor  # registered buffer, avoids torch.tensor() each call
        type_oh = (p_type.unsqueeze(-1) == tc_t) & active.unsqueeze(-1)
        sum_dims = tuple(range(type_oh.dim() - 1))
        type_counts = type_oh.sum(dim=sum_dims)
        # Single DtoH transfer for all counts, then iterate on CPU
        counts_list = type_counts.tolist()

        # Precompute all per-type masks at once (one kernel per type, not per observable)
        type_masks = [type_oh[..., idx].bool() for idx in range(len(type_codes_list))]

        for idx, (tcode, name) in enumerate(zip(type_codes_list, type_names_list)):
            if counts_list[idx] < 2:
                continue
            tmask = type_masks[idx]

            lpt_p, lpt_t = logpT_p[tmask], logpT_t[tmask]
            e_p_m, e_t_m = eta_p[tmask], eta_t[tmask]
            phi_p_m, phi_t_m = phi_p[tmask], phi_t[tmask]
            le_p_m, le_t_m = logE_p[tmask], logE_t[tmask]

            metrics[f"1-particle/{name}_pT_w1"] = (
                (lpt_p.sort().values - lpt_t.sort().values).abs().mean()
            )
            metrics[f"1-particle/{name}_pT_bc"] = self._bc_torch(lpt_p, lpt_t)
            metrics[f"1-particle/{name}_eta_w1"] = (
                (e_p_m.sort().values - e_t_m.sort().values).abs().mean()
            )
            metrics[f"1-particle/{name}_eta_bc"] = self._bc_torch(e_p_m, e_t_m)
            metrics[f"1-particle/{name}_phi_w1"] = (
                (phi_p_m.sort().values - phi_t_m.sort().values).abs().mean()
            )
            metrics[f"1-particle/{name}_phi_bc"] = self._bc_torch(phi_p_m, phi_t_m)
            metrics[f"1-particle/{name}_E_w1"] = (
                (le_p_m.sort().values - le_t_m.sort().values).abs().mean()
            )
            metrics[f"1-particle/{name}_E_bc"] = self._bc_torch(le_p_m, le_t_m)

            agg_pT_p.append(lpt_p)
            agg_pT_t.append(lpt_t)
            agg_eta_p.append(e_p_m)
            agg_eta_t.append(e_t_m)
            agg_phi_p.append(phi_p_m)
            agg_phi_t.append(phi_t_m)
            agg_E_p.append(le_p_m)
            agg_E_t.append(le_t_m)

        # Aggregate W1 across all types
        if agg_pT_p:
            all_pT_p = torch.cat(agg_pT_p).sort().values
            all_pT_t = torch.cat(agg_pT_t).sort().values
            metrics["1-particle/all_pT_w1"] = (all_pT_p - all_pT_t).abs().mean()
            metrics["1-particle/all_pT_bc"] = self._bc_torch(all_pT_p, all_pT_t)
            all_eta_p = torch.cat(agg_eta_p).sort().values
            all_eta_t = torch.cat(agg_eta_t).sort().values
            metrics["1-particle/all_eta_w1"] = (all_eta_p - all_eta_t).abs().mean()
            metrics["1-particle/all_eta_bc"] = self._bc_torch(all_eta_p, all_eta_t)
            all_phi_p = torch.cat(agg_phi_p).sort().values
            all_phi_t = torch.cat(agg_phi_t).sort().values
            metrics["1-particle/all_phi_w1"] = (all_phi_p - all_phi_t).abs().mean()
            metrics["1-particle/all_phi_bc"] = self._bc_torch(all_phi_p, all_phi_t)
            all_E_p = torch.cat(agg_E_p).sort().values
            all_E_t = torch.cat(agg_E_t).sort().values
            metrics["1-particle/all_E_w1"] = (all_E_p - all_E_t).abs().mean()
            metrics["1-particle/all_E_bc"] = self._bc_torch(all_E_p, all_E_t)

        # Event-level scalar sum pT, leading/subleading pT
        if geometric:
            # scatter_add for per-event sums
            scalar_sum_pT_p = torch.zeros(B, device=pT_p.device)
            scalar_sum_pT_t = torch.zeros(B, device=pT_t.device)
            scalar_sum_pT_p.scatter_add_(0, batch_idx, pT_p)
            scalar_sum_pT_t.scatter_add_(0, batch_idx, pT_t)

            # Leading/subleading via scatter_max (no Python loop)
            # scatter_max gives per-event maximum; for subleading, mask out the lead
            lead_p = torch.zeros(B, device=pT_p.device)
            lead_t = torch.zeros(B, device=pT_t.device)
            lead_p.scatter_reduce_(0, batch_idx, pT_p, reduce="amax")
            lead_t.scatter_reduce_(0, batch_idx, pT_t, reduce="amax")

            # Subleading: mask out leading particle, then scatter_max again
            is_lead_p = (pT_p == lead_p[batch_idx]) & active
            pT_p_sub = torch.where(is_lead_p, torch.zeros_like(pT_p), pT_p)
            pT_t_sub = torch.where(
                (pT_t == lead_t[batch_idx]) & active, torch.zeros_like(pT_t), pT_t
            )
            sub_p = torch.zeros(B, device=pT_p.device)
            sub_t = torch.zeros(B, device=pT_t.device)
            sub_p.scatter_reduce_(0, batch_idx, pT_p_sub, reduce="amax")
            sub_t.scatter_reduce_(0, batch_idx, pT_t_sub, reduce="amax")
        else:
            pT_p_masked = pT_p.clone()
            pT_t_masked = pT_t.clone()
            pT_p_masked[~active] = 0.0
            pT_t_masked[~active] = 0.0
            scalar_sum_pT_p = pT_p_masked.sum(dim=-1)
            scalar_sum_pT_t = pT_t_masked.sum(dim=-1)

            big_neg = -1e9
            pT_p_sort = pT_p_masked.clone()
            pT_t_sort = pT_t_masked.clone()
            pT_p_sort[~active] = big_neg
            pT_t_sort[~active] = big_neg
            pT_p_sorted = pT_p_sort.sort(dim=-1, descending=True).values
            pT_t_sorted = pT_t_sort.sort(dim=-1, descending=True).values
            lead_p, lead_t = pT_p_sorted[:, 0], pT_t_sorted[:, 0]
            sub_p = pT_p_sorted[:, 1] if pT_p_sorted.shape[1] > 1 else None
            sub_t = pT_t_sorted[:, 1] if pT_t_sorted.shape[1] > 1 else None

        metrics["n-particle/scalar_sum_pT_w1"] = (
            (scalar_sum_pT_p.sort().values - scalar_sum_pT_t.sort().values).abs().mean()
        )
        metrics["n-particle/scalar_sum_pT_bc"] = self._bc_torch(
            scalar_sum_pT_p, scalar_sum_pT_t
        )
        metrics["1-particle/leading_pT_w1"] = (
            (lead_p.sort().values - lead_t.sort().values).abs().mean()
        )
        metrics["1-particle/leading_pT_bc"] = self._bc_torch(lead_p, lead_t)
        if sub_p is not None and sub_t is not None:
            valid_sub = (sub_p > 0) & (sub_t > 0)
            if valid_sub.sum() > 1:
                metrics["1-particle/subleading_pT_w1"] = (
                    (sub_p[valid_sub].sort().values - sub_t[valid_sub].sort().values)
                    .abs()
                    .mean()
                )
                metrics["1-particle/subleading_pT_bc"] = self._bc_torch(
                    sub_p[valid_sub], sub_t[valid_sub]
                )

        # ── Event-level OSSF dilepton + MET ──
        if met_vector is not None:
            ossf = self._compute_ossf_metrics(
                px_p,
                py_p,
                pz_p,
                E_p,
                px_t,
                py_t,
                pz_t,
                E_t,
                p_type,
                p_charge,
                active,
                met_vector,
                batch_idx=batch_idx,
                ptr=ptr,
            )
            metrics.update(ossf)

        # Filter out NaN/Inf metrics
        clean = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
                if v.isfinite():
                    clean[k] = v
            else:
                clean[k] = v
        return clean

    @torch.no_grad()
    def _compute_ossf_metrics(
        self,
        px_p,
        py_p,
        pz_p,
        E_p,
        px_t,
        py_t,
        pz_t,
        E_t,
        p_type,
        p_charge,
        active,
        met_vector,
        batch_idx=None,
        ptr=None,
    ) -> dict:
        """OSSF dilepton ΔR, m(ℓℓ), Δφ(ℓℓ,MET), mT(ℓℓ,MET).

        Works on both padded (B, N) and geometric (T_total,) tensors.
        """
        device = p_type.device
        geometric = batch_idx is not None

        if geometric:
            B = ptr.shape[0] - 1
        else:
            B, N = p_type.shape

        is_lepton = ((p_type == 11) | (p_type == 13)) & active
        q_signed = p_charge.float() - 1.0

        # Find first OSSF pair per event — vectorized
        # Use pairwise comparison on the lepton subset
        lep_pos = is_lepton.nonzero(as_tuple=True)[0] if geometric else None

        if geometric and lep_pos is not None and lep_pos.shape[0] >= 2:
            lep_ev = batch_idx[lep_pos]
            lep_typ = p_type[lep_pos]
            lep_q = q_signed[lep_pos]
            L = lep_pos.shape[0]
            same_ev = lep_ev.unsqueeze(0) == lep_ev.unsqueeze(1)
            same_fl = lep_typ.unsqueeze(0) == lep_typ.unsqueeze(1)
            opp_q = (lep_q.unsqueeze(0) * lep_q.unsqueeze(1)) < -0.5
            triu_m = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
            )
            valid = same_ev & same_fl & opp_q & triu_m
            if valid.any():
                vi, vj = valid.nonzero(as_tuple=True)
                all_ii = lep_pos[vi]
                all_jj = lep_pos[vj]
                all_ev = lep_ev[vi]
                # Take first pair per event: unique on event index
                _, _first_idx = (
                    torch.unique(
                        all_ev, return_inverse=False, sorted=True, return_counts=False
                    ),
                    None,
                )
                # Actually, torch.unique with return_inverse doesn't give first occurrence.
                # Use a scatter trick: for each event, keep the first pair (lowest index)
                uniq_events, inv = torch.unique(all_ev, return_inverse=True)
                # First occurrence per unique event = argmin of pair index within each event
                # First occurrence per event: scatter trick
                # For sorted events, the first occurrence is where inv differs from previous
                first_mask = torch.ones(
                    all_ev.shape[0], dtype=torch.bool, device=device
                )
                first_mask[1:] = inv[1:] != inv[:-1]
                ii_t = all_ii[first_mask]
                jj_t = all_jj[first_mask]
                ev_idx_t = all_ev[first_mask]
            else:
                return {}
        elif not geometric:
            # Padded path: use existing vectorized selection
            from shellflow.training.riemannian_DiTs_loss import _select_all_ossf_leptons

            p_mask_float = active.float()
            sel = _select_all_ossf_leptons(p_type, p_charge, p_mask_float)
            if sel is None or sel[0].shape[0] < 2:
                return {}
            all_b, all_ii, all_jj = sel
            uniq_events, inv = torch.unique(all_b, return_inverse=True)
            first_mask = torch.zeros(all_b.shape[0], dtype=torch.bool, device=device)
            seen = torch.zeros(uniq_events.shape[0], dtype=torch.bool, device=device)
            for k in range(all_b.shape[0]):
                if not seen[inv[k]]:
                    first_mask[k] = True
                    seen[inv[k]] = True
            ev_idx_t = all_b[first_mask]
            ii_t = all_ii[first_mask]
            jj_t = all_jj[first_mask]
        else:
            return {}

        n_valid = ev_idx_t.shape[0]
        if n_valid < 2:
            return {}

        # Gather pair kinematics
        if geometric:
            # ii_t, jj_t are global flat indices
            pxi_p, pyi_p, pzi_p, Ei_p = px_p[ii_t], py_p[ii_t], pz_p[ii_t], E_p[ii_t]
            pxj_p, pyj_p, pzj_p, Ej_p = px_p[jj_t], py_p[jj_t], pz_p[jj_t], E_p[jj_t]
            pxi_t, pyi_t, pzi_t, Ei_t = px_t[ii_t], py_t[ii_t], pz_t[ii_t], E_t[ii_t]
            pxj_t, pyj_t, pzj_t, Ej_t = px_t[jj_t], py_t[jj_t], pz_t[jj_t], E_t[jj_t]
        else:
            # ii_t, jj_t are per-event N-indices; ev_idx_t selects the event
            pxi_p = px_p[ev_idx_t, ii_t]
            pyi_p = py_p[ev_idx_t, ii_t]
            pzi_p = pz_p[ev_idx_t, ii_t]
            Ei_p = E_p[ev_idx_t, ii_t]
            pxj_p = px_p[ev_idx_t, jj_t]
            pyj_p = py_p[ev_idx_t, jj_t]
            pzj_p = pz_p[ev_idx_t, jj_t]
            Ej_p = E_p[ev_idx_t, jj_t]
            pxi_t = px_t[ev_idx_t, ii_t]
            pyi_t = py_t[ev_idx_t, ii_t]
            pzi_t = pz_t[ev_idx_t, ii_t]
            Ei_t = E_t[ev_idx_t, ii_t]
            pxj_t = px_t[ev_idx_t, jj_t]
            pyj_t = py_t[ev_idx_t, jj_t]
            pzj_t = pz_t[ev_idx_t, jj_t]
            Ej_t = E_t[ev_idx_t, jj_t]

        # Dilepton 4-vector
        dll_px_p, dll_py_p = pxi_p + pxj_p, pyi_p + pyj_p
        dll_pz_p, dll_E_p = pzi_p + pzj_p, Ei_p + Ej_p
        dll_px_t, dll_py_t = pxi_t + pxj_t, pyi_t + pyj_t
        dll_pz_t, dll_E_t = pzi_t + pzj_t, Ei_t + Ej_t

        # m(ℓℓ)
        mll_p = torch.sqrt(
            (dll_E_p**2 - dll_px_p**2 - dll_py_p**2 - dll_pz_p**2).clamp(min=0) + 1e-8
        )
        mll_t = torch.sqrt(
            (dll_E_t**2 - dll_px_t**2 - dll_py_t**2 - dll_pz_t**2).clamp(min=0) + 1e-8
        )

        def _eta_phi(px, py, pz):
            p = torch.sqrt(px**2 + py**2 + pz**2 + 1e-8)
            return (
                torch.arctanh((pz / p).clamp(-1 + 1e-6, 1 - 1e-6)),
                torch.atan2(py, px),
            )

        def _wrap(d):
            return d - 2 * math.pi * torch.round(d / (2 * math.pi))

        # ΔR(ℓ₁, ℓ₂)
        ei_p, pi_p = _eta_phi(pxi_p, pyi_p, pzi_p)
        ej_p, pj_p = _eta_phi(pxj_p, pyj_p, pzj_p)
        dR_p = torch.sqrt((ei_p - ej_p) ** 2 + _wrap(pi_p - pj_p) ** 2 + 1e-8)
        ei_t, pi_t = _eta_phi(pxi_t, pyi_t, pzi_t)
        ej_t, pj_t = _eta_phi(pxj_t, pyj_t, pzj_t)
        dR_t = torch.sqrt((ei_t - ej_t) ** 2 + _wrap(pi_t - pj_t) ** 2 + 1e-8)

        # Δφ(dilepton, MET)
        met_px = met_vector[ev_idx_t, 0]
        met_py = met_vector[ev_idx_t, 1]
        phi_met = torch.atan2(met_py, met_px)
        dphi_met_p = _wrap(torch.atan2(dll_py_p, dll_px_p) - phi_met)
        dphi_met_t = _wrap(torch.atan2(dll_py_t, dll_px_t) - phi_met)

        # mT(dilepton, MET)
        pT_dll_p = torch.sqrt(dll_px_p**2 + dll_py_p**2 + 1e-8)
        pT_dll_t = torch.sqrt(dll_px_t**2 + dll_py_t**2 + 1e-8)
        met_mag = torch.sqrt(met_px**2 + met_py**2 + 1e-8)
        mT_p = torch.sqrt(
            (2 * pT_dll_p * met_mag * (1 - torch.cos(dphi_met_p))).clamp(min=0) + 1e-8
        )
        mT_t = torch.sqrt(
            (2 * pT_dll_t * met_mag * (1 - torch.cos(dphi_met_t))).clamp(min=0) + 1e-8
        )

        metrics: dict = {}

        if n_valid >= 4:
            log_mll_p = torch.log(mll_p.clamp(min=1e-4))
            log_mll_t = torch.log(mll_t.clamp(min=1e-4))
            metrics["2-particle/ossf_log_mll_w1"] = (
                (log_mll_p.sort().values - log_mll_t.sort().values).abs().mean()
            )
            metrics["2-particle/ossf_log_mll_bc"] = self._bc_torch(log_mll_p, log_mll_t)
            metrics["2-particle/ossf_dR_w1"] = (
                (dR_p.sort().values - dR_t.sort().values).abs().mean()
            )
            metrics["2-particle/ossf_dR_bc"] = self._bc_torch(dR_p, dR_t)
            metrics["2-particle/ossf_dphi_w1"] = (
                (dphi_met_p.sort().values - dphi_met_t.sort().values).abs().mean()
            )
            metrics["2-particle/ossf_dphi_bc"] = self._bc_torch(dphi_met_p, dphi_met_t)
            metrics["2-particle/ossf_met_w1"] = (
                (mT_p.sort().values - mT_t.sort().values).abs().mean()
            )
            metrics["2-particle/ossf_met_bc"] = self._bc_torch(mT_p, mT_t)

        return metrics

    # ------------------------------------------------------------------
    # Compiled flat forward core
    # ------------------------------------------------------------------
    def _flat_compiled_core(
        self,
        kin: torch.Tensor,
        p_type: torch.Tensor,
        p_charge: torch.Tensor,
        p_extra: torch.Tensor,
        batch_idx: torch.Tensor,
        ptr: torch.Tensor,
        met_vector: torch.Tensor | None,
        max_seqlen: int,
        T_real: int,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fully compilable forward: encode → interpolant → model → loss.

        No .item() calls, no Python loops, no data-dependent branching.
        T_real is a Python int from the collator — no GPU sync.
        t is sampled outside to avoid graph breaks.
        """
        kin_f32 = kin.float()
        B = ptr.shape[0] - 1

        # Type info + encode (flat)
        # Padding tokens from pad_batch_to_static have p_type=0 — mark inactive
        p_mask = (p_type > 0).float()
        ti = self._precompute_type_info(p_type, p_mask)
        x_data = self.model.encode_to_manifold(kin_f32, p_type, p_mask)

        # Interpolant — PyTorch SLERP path (compilable by torch.compile)
        psi_t, u_t, x_0 = self._rfm_interpolant(
            x_data,
            t,
            ti,
            batch_idx=batch_idx,
        )

        # Model forward (flat) — T_real passed as Python int, no graph break
        v_theta = self.model.forward_flat(
            psi_t,
            p_type,
            p_charge,
            p_extra,
            t,
            batch_idx,
            ptr,
            met_vector=met_vector,
            max_seqlen=max_seqlen,
            T_real=T_real,
        )
        v_theta_f32 = v_theta.float()

        # Loss weighting from EMA bins (compilable)
        loss_weight = None
        if self.loss_weighting_t:
            t_per_particle = t[batch_idx]  # (B,) → (T,)
            loss_weight = self._compute_loss_weights(t_per_particle)

        # Loss (compilable core — per-event normalisation)
        loss, type_means, type_counts = self._compute_cfm_loss_core(
            v_theta_f32,
            u_t,
            ti,
            p_type,
            loss_weight=loss_weight,
            batch_idx=batch_idx,
            B=B,
            kin=kin_f32,
            p_extra=p_extra,
        )
        # Return intermediates for diagnostics/aux loss outside compiled region.
        # Legacy chart: e_ref for massless E; chi_mu/chi_sigma for massive
        # rapidity; logm_mu/logm_sigma for massive log-mass.
        return (
            loss,
            type_means,
            type_counts,
            v_theta_f32,
            psi_t,
            u_t,
            x_data,
            t,
            ti["is_massless"],
            ti["is_massive"],
            ti["e_ref"],
            ti["active"],
            ti["mass_floor"],
            ti["chi_mu"],
            ti["chi_sigma"],
            ti["logm_mu"],
            ti["logm_sigma"],
            ti["m_rest"],
        )

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------
    def _shared_step(self, batch: dict):
        # Detect geometric vs padded mode via duck-typing
        geometric = "batch" in batch

        # Fast path: fully compiled flat pipeline
        compile_val = self.model_config.get("compile_validation", True)
        if (
            geometric
            and self.model_config.get("use_flat_forward", False)
            and (self.training or compile_val)
        ):
            kin = batch["kinematics"]
            p_type = batch["particle_type"]
            p_charge = batch["particle_charge"]
            p_extra = batch["particle_extra"]
            batch_idx = batch["batch"]
            ptr = batch["ptr"]
            met_vector = batch.get("met_vector")
            max_seqlen = batch["max_seqlen"]
            T_real = batch.get("T_real", kin.shape[0])  # Python int from collator
            B = ptr.shape[0] - 1

            # Compile on first call
            if not hasattr(self, "_flat_core_compiled"):
                compile_mode = self.model_config.get(
                    "compile_mode_flat", "max-autotune"
                )
                self._flat_core_compiled = torch.compile(
                    self._flat_compiled_core,
                    mode=compile_mode,
                    dynamic=False,
                )

            # Sample time outside compiled core (method dispatch has Python conditionals)
            t = self._sample_time(B, kin.device)

            # ── Compiled core: encode → interpolant → model → loss ──
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                (
                    loss,
                    type_means,
                    type_counts,
                    v_theta_f32,
                    psi_t,
                    u_t,
                    x_data,
                    t,
                    is_massless,
                    is_massive,
                    e_ref,
                    active,
                    mass_floor,
                    chi_mu,
                    chi_sigma,
                    logm_mu,
                    logm_sigma,
                    m_rest,
                ) = self._flat_core_compiled(
                    kin,
                    p_type,
                    p_charge,
                    p_extra,
                    batch_idx,
                    ptr,
                    met_vector,
                    max_seqlen,
                    T_real,
                    t,
                )

            # ── Everything below runs eagerly outside the compiled graph ──

            # Rebuild ti dict from returned flat tensors (legacy chart).
            ti = {
                "is_massless": is_massless,
                "is_massive": is_massive,
                "e_ref": e_ref,
                "active": active,
                "mass_floor": mass_floor,
                "chi_mu": chi_mu,
                "chi_sigma": chi_sigma,
                "logm_mu": logm_mu,
                "logm_sigma": logm_sigma,
                "m_rest": m_rest,
            }

            # Per-type loss metrics
            metrics = self._extract_loss_metrics(loss, type_means, type_counts, ti)
            metrics["batch_size"] = B

            # Loss weighting EMA: track unweighted per-event loss in bins
            if self.loss_weighting_t and self.training:
                with torch.no_grad():
                    pp = F.huber_loss(
                        v_theta_f32,
                        u_t,
                        reduction="none",
                    ).sum(dim=-1)
                    pp = torch.where(active, pp, torch.zeros_like(pp))
                    ev_sum = torch.zeros(B, device=pp.device, dtype=pp.dtype)
                    ev_cnt = torch.zeros(B, device=pp.device, dtype=pp.dtype)
                    ev_sum.scatter_add_(0, batch_idx, pp)
                    ev_cnt.scatter_add_(0, batch_idx, active.float())
                    per_event_loss = ev_sum / ev_cnt.clamp(min=1)
                self._update_loss_weight_ema(t, per_event_loss)
                if self.global_step >= self.loss_weighting_t_warmup:
                    metrics["loss_weighting_t/ema_ratio"] = (
                        self._at_loss_ema.max()
                        / self._at_loss_ema.min().clamp(min=1e-8)
                    ).item()

            # Kinematic diagnostics (every 50 train steps; always during validation)
            step = self.global_step
            if not self.training or step % 50 == 0:
                with torch.no_grad():
                    x1_hat = self._single_step_endpoint(
                        psi_t,
                        v_theta_f32,
                        t,
                        ti,
                        batch_idx=batch_idx,
                    )
                x_data_f32 = x_data.float()
                kin_metrics = self._compute_kinematic_metrics(
                    x1_hat,
                    x_data_f32,
                    ti,
                    p_type,
                    p_charge,
                    met_vector,
                    batch_idx=batch_idx,
                    ptr=ptr,
                )
                metrics.update(kin_metrics)

            return loss, metrics

        if geometric:
            kin = batch["kinematics"]  # (T_total, 4)
            p_type = batch["particle_type"]  # (T_total,)
            p_charge = batch["particle_charge"]
            p_extra = batch["particle_extra"]
            batch_idx = batch["batch"]  # (T_total,) int64
            ptr = batch["ptr"]  # (B+1,) int32
            met_vector = batch.get("met_vector")
            B = ptr.shape[0] - 1
            # Flat batches from pad_batch_to_static may have padding tokens
            # (p_type=0). Pure geometric batches have all real particles.
            has_padding = self.model_config.get("use_flat_forward", False)
            p_mask = (p_type > 0).float() if has_padding else None
            # max_seqlen computed on CPU by the collator — no GPU sync needed
            max_seqlen = batch["max_seqlen"]

            # Hybrid mode: convert geometric → padded (B, N_max) at entry,
            # then run the entire padded code path. The geometric collator
            # still reduces N from 30 to ~8.
            if self.model_config.get("geometric_hybrid", False):
                from shellflow.data.particle_vector_collator import (
                    geometric_to_dense,
                )

                fixed_N = self.model_config.get("geometric_hybrid_fixed_N", None)
                kin, p_mask = geometric_to_dense(kin, ptr, fixed_N=fixed_N)
                p_type, _ = geometric_to_dense(
                    p_type.unsqueeze(-1).float(), ptr, fixed_N=fixed_N
                )
                p_type = p_type.squeeze(-1).long()
                p_charge, _ = geometric_to_dense(
                    p_charge.unsqueeze(-1).float(), ptr, fixed_N=fixed_N
                )
                p_charge = p_charge.squeeze(-1).long()
                p_extra, _ = geometric_to_dense(p_extra, ptr, fixed_N=fixed_N)
                p_mask = p_mask.float()
                batch_idx = None
                ptr = None
                geometric = False
        else:
            kin = batch["kinematics"]  # (B, N, 4)
            p_type = batch["particle_type"]  # (B, N)
            p_charge = batch["particle_charge"]
            p_extra = batch["particle_extra"]
            p_mask = batch["particle_mask"]  # (B, N)
            met_vector = batch.get("met_vector")
            batch_idx = None
            ptr = None
            B = kin.shape[0]

        # All geometry in float32
        kin_f32 = kin.float()

        # Precompute type info once (avoids repeated map_type_to_idx + LUT lookups)
        ti = self._precompute_type_info(p_type, p_mask)

        # 1. Encode to manifold representation
        x_data = self.model.encode_to_manifold(kin_f32, p_type, p_mask)

        # 2. Sample time
        t = self._sample_time(B, self.device)

        x_data_f32 = x_data.float()

        # RFM: geodesic interpolant + velocity regression
        psi_t, u_t, x_0 = self._rfm_interpolant(
            x_data,
            t,
            ti,
            batch_idx=batch_idx,
        )

        v_theta_aux = None  # set by padded path when aux head is enabled

        if geometric and self.model_config.get("use_flat_forward", False):
            v_theta = self.model.forward_flat(
                psi_t,
                p_type,
                p_charge,
                p_extra,
                t,
                batch_idx,
                ptr,
                met_vector=met_vector,
                max_seqlen=max_seqlen,
            )
        elif geometric and self.model_config.get("use_njt_forward", False):
            v_theta = self.model.forward_njt(
                psi_t,
                p_type,
                p_charge,
                p_extra,
                t,
                batch_idx,
                ptr,
                met_vector=met_vector,
                max_seqlen=max_seqlen,
            )
        elif geometric:
            v_theta = self.model.forward_geometric(
                psi_t,
                p_type,
                p_charge,
                p_extra,
                t,
                batch_idx,
                ptr,
                met_vector=met_vector,
                max_seqlen=max_seqlen,
            )
        else:
            need_aux = self.use_aux_head and self.training
            result = self.model(
                psi_t,
                p_type,
                p_charge,
                p_extra,
                t,
                p_mask,
                met_vector=met_vector,
                return_aux=need_aux,
            )
            if need_aux:
                v_theta, v_theta_aux = result
            else:
                v_theta = result
                v_theta_aux = None
        v_theta_f32 = v_theta.float()

        cfm_loss, metrics = self._compute_cfm_loss(
            v_theta_f32,
            u_t,
            ti,
            p_type,
            t=t,
            batch_idx=batch_idx,
            B=B,
            kin=kin_f32,
            p_extra=p_extra,
        )

        # Aux head loss (padded mode only, training only).
        # p_type/p_extra needed for quality weighting; p_type/p_charge needed
        # for category-conditioned distribution matching. Hierarchical mass is
        # purely analytical (uses the same v_theta_aux as flat).
        aux_head_loss = None
        if self.use_aux_head and self.training and v_theta_aux is not None:
            ah_needs_id = (
                self.use_aux_head_quality or self.use_aux_head_distribution_matching
            )
            aux_result = self._compute_aux_head_loss(
                v_theta_aux.float(),
                u_t.float(),
                psi_t.float(),
                x_data_f32,
                t,
                ti,
                p_mask,
                p_type=p_type if ah_needs_id else None,
                p_extra=p_extra if self.use_aux_head_quality else None,
                p_charge=p_charge if self.use_aux_head_distribution_matching else None,
            )
            if aux_result is not None:
                aux_head_loss, aux_metrics = aux_result
                metrics.update(aux_metrics)

        # Combine CFM + aux head losses
        loss = cfm_loss
        if aux_head_loss is not None:
            loss = loss + aux_head_loss

        # Loss weighting EMA: track unweighted per-event loss in bins
        if self.loss_weighting_t and self.training:
            with torch.no_grad():
                pp = F.huber_loss(
                    v_theta_f32,
                    u_t,
                    reduction="none",
                ).sum(dim=-1)
                act = ti["active"]
                if pp.dim() == 2:
                    # Padded mode: (B, N) → per-event mean
                    act_f = act.float()
                    pp = torch.where(act, pp, torch.zeros_like(pp))
                    per_event_loss = (pp * act_f).sum(dim=-1) / act_f.sum(dim=-1).clamp(
                        min=1
                    )
                else:
                    # Flat/geometric mode: (T,) → scatter
                    pp = torch.where(act, pp, torch.zeros_like(pp))
                    ev_sum = torch.zeros(B, device=pp.device, dtype=pp.dtype)
                    ev_cnt = torch.zeros(B, device=pp.device, dtype=pp.dtype)
                    ev_sum.scatter_add_(0, batch_idx, pp)
                    ev_cnt.scatter_add_(0, batch_idx, act.float())
                    per_event_loss = ev_sum / ev_cnt.clamp(min=1)
            self._update_loss_weight_ema(t, per_event_loss)
            if self.global_step >= self.loss_weighting_t_warmup:
                metrics["loss_weighting_t/ema_ratio"] = (
                    self._at_loss_ema.max() / self._at_loss_ema.min().clamp(min=1e-8)
                ).item()

        # Kinematic diagnostics (every 50 train steps; always during validation)
        step = self.global_step
        if not self.training or step % 50 == 0:
            with torch.no_grad():
                x1_hat = self._single_step_endpoint(
                    psi_t,
                    v_theta_f32,
                    t,
                    ti,
                    batch_idx=batch_idx,
                )
            kin_metrics = self._compute_kinematic_metrics(
                x1_hat,
                x_data_f32,
                ti,
                p_type,
                p_charge,
                met_vector,
                batch_idx=batch_idx,
                ptr=ptr,
            )
            metrics.update(kin_metrics)

        metrics["batch_size"] = B
        return loss, metrics

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _log_metrics(self, loss: torch.Tensor, metrics: dict) -> None:
        """Log loss and per-key metrics to W&B / Lightning loggers."""
        self.total_events_seen += metrics["batch_size"]
        self.events_this_epoch += metrics["batch_size"]

        self.log("train/loss/total", loss.detach(), prog_bar=True, sync_dist=True)

        # NOTE: per-key metric logging deliberately does NOT use sync_dist=True
        # because the keys can be CONDITIONAL on the rank's local batch:
        #   - per-type losses (loss/electron, loss/jet, ...) only present if
        #     that rank's batch contains the type.
        # With sync_dist=True, ranks must call .log() with the SAME key set or
        # the all-reduce hangs (NCCL deadlock — observed in practice). Until
        # the metrics dict is restructured to always include all keys (with
        # NaN/zero for missing ones), train metrics remain rank-0-local for
        # display. This is display-only and does NOT affect training math.
        # (val/* metrics CAN safely use sync_dist=True because validation runs
        # the same dataset shard structure across ranks.)
        _SKIP = {"loss", "batch_size", "n_active"}
        for key, val in metrics.items():
            if key not in _SKIP and isinstance(val, (int, float, torch.Tensor)):
                self.log(f"train/{key}", val, prog_bar=False)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=True)

    def training_step(self, batch: dict, batch_idx: int):
        loss, metrics = self._shared_step(batch)
        self._log_metrics(loss, metrics)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # EMA update runs on ALL ranks. Inputs (self._model_params) are kept
        # identical across ranks by DDP gradient all-reduce in backward();
        # the lerp is deterministic; seed_everything() before module
        # construction guarantees identical initial self.ema_model. So every
        # rank computes the same trained EMA each step — no broadcast or
        # other cross-rank sync needed. Previously gated by is_global_zero,
        # which silently broke any code that consumed pl_module.ema_model on
        # a non-zero rank (notably the multi-rank Higgs callback at
        # _higgs_generate_shard, which would use the stale random-init EMA
        # → 4× over-count in the no-quality-cut m_4ℓ plot).
        if self.ema_model is not None:
            with torch.no_grad():
                torch._foreach_lerp_(
                    self._ema_params, self._model_params, 1.0 - self.ema_decay
                )
                for b_ema, b in zip(self.ema_model.buffers(), self.model.buffers()):
                    b_ema.copy_(b)

    def on_before_optimizer_step(self, optimizer) -> None:
        """DDP-safe non-finite-gradient guard.

        Runs AFTER backward + DDP gradient all-reduce, so every rank sees the
        SAME gradients and makes the SAME decision — no desync (a per-rank
        `return None` skip would itself hang DDP). If any gradient is NaN/Inf,
        zero ALL grads so this optimizer step is a no-op: one poisoned step
        otherwise turns every weight into NaN (grad_clip cannot rescue a
        NaN-norm) and silently corrupts the rest of a multi-day run. Cost: one
        fused grad-norm check per step (negligible).
        """
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if not grads:
            return
        total = torch.stack(torch._foreach_norm(grads)).norm()
        if not torch.isfinite(total):
            self._nonfinite_grad_steps = getattr(self, "_nonfinite_grad_steps", 0) + 1
            for g in grads:
                g.zero_()
            if self._nonfinite_grad_steps <= 5 or self._nonfinite_grad_steps % 100 == 0:
                logger.warning(
                    f"Non-finite gradient at step {self.global_step}; zeroed "
                    f"(total non-finite steps so far: {self._nonfinite_grad_steps})"
                )

    def on_train_epoch_start(self) -> None:
        self.events_this_epoch = 0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch: dict, batch_idx: int):
        loss, metrics = self._shared_step(batch)
        metrics["loss"] = loss.detach()

        cpu_metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }
        self.validation_step_outputs.append(cpu_metrics)
        return cpu_metrics

    def on_validation_epoch_end(self) -> None:
        if not self.validation_step_outputs:
            return
        outputs = self.validation_step_outputs
        total_samples = sum(o["batch_size"] for o in outputs)

        def _wavg(key):
            return sum(o.get(key, 0) * o["batch_size"] for o in outputs) / total_samples

        # NCCL deadlock guard: per-key val metrics are CONDITIONAL on the
        # rank's local batch composition (e.g. `loss/muon` is absent on a
        # rank whose val batches happen to contain no muons). With
        # sync_dist=True, each .log() call triggers a `dist.all_reduce`, and
        # mismatched key counts across ranks → asymmetric all_reduce
        # sequences → NCCL deadlock at the val/epoch boundary. The training
        # step had the same problem and resolved it with sync_dist=False for
        # per-key metrics (see _log_metrics comment). The same constraint
        # applies here — even though val *shards* are identical-shape, the
        # PER-BATCH key composition still varies. val/loss remains
        # sync_dist=True because it is unconditionally present on every rank.
        self.log("val/loss/total", _wavg("loss"), prog_bar=True, sync_dist=True)

        _SKIP = {"loss", "batch_size", "n_active"}
        for key in outputs[0]:
            if key not in _SKIP and isinstance(outputs[0][key], (int, float)):
                self.log(f"val/{key}", _wavg(key), prog_bar=False)

        self.validation_step_outputs.clear()

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt_config = self.optimizer_config
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.betas,
            eps=opt_config.eps,
        )

        sched_config = self.scheduler_config

        if sched_config.name == "cosine":
            if self.total_steps is None:
                raise ValueError("total_steps required for CosineAnnealingLR")
            if sched_config.warmup_steps is not None:
                warmup_steps = min(sched_config.warmup_steps, self.total_steps // 2)
            else:
                warmup_steps = int(sched_config.warmup_pct * self.total_steps)

            cosine_steps = self.total_steps - warmup_steps
            if warmup_steps > 0:
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-8,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_steps,
                    eta_min=sched_config.min_lr,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[warmup_steps],
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.total_steps,
                    eta_min=sched_config.min_lr,
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        elif sched_config.name == "onecycle":
            if self.total_steps is None:
                raise ValueError("total_steps required for OneCycleLR")
            pct_start = (
                min(sched_config.warmup_steps / self.total_steps, 0.5)
                if sched_config.warmup_steps is not None
                else sched_config.warmup_pct
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt_config.lr,
                total_steps=self.total_steps,
                pct_start=pct_start,
                anneal_strategy=sched_config.anneal_strategy,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        elif sched_config.name == "constant":
            return optimizer

        else:
            raise ValueError(f"Unknown scheduler: {sched_config.name}")
