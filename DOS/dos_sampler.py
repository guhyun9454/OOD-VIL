from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# scikit-learn은 환경에 따라 numpy ABI 불일치로 import가 깨질 수 있어,
# 가능하면 사용하되 실패 시 numpy 기반의 간단 KMeans로 fallback 합니다.
try:
    from sklearn.cluster import KMeans as _SKLearnKMeans  # type: ignore
except Exception:  # pragma: no cover
    _SKLearnKMeans = None


@dataclass(frozen=True)
class DOSConfig:
    """Configuration for Diverse Outlier Sampling (DOS).

    Notes:
    - This repo does NOT assume an explicit absent class (K+1). When absent class
      logits are not available, DOS hardness can be approximated via MSP / MaxLogit / Energy.
    """

    num_clusters: int
    n_init: int = 10
    hardness: str = "msp"  # {"msp", "maxlogit", "energy"}
    fill_mode: str = "random"  # {"random", "hard"}
    energy_temperature: float = 1.0
    seed: Optional[int] = None


def _kmeans_fit_predict_numpy(
    x: np.ndarray,
    *,
    n_clusters: int,
    n_init: int = 10,
    max_iter: int = 50,
    seed: Optional[int] = None,
) -> np.ndarray:
    """A small, dependency-free KMeans (Lloyd) for fallback."""

    x = x.astype(np.float32, copy=False)
    N, D = x.shape
    k = int(n_clusters)
    if k <= 0 or k > N:
        raise ValueError(f"[DOS] invalid n_clusters={k} for N={N}")

    rng = np.random.default_rng(seed)
    best_inertia = float("inf")
    best_labels = None

    x_norm2 = (x * x).sum(axis=1, keepdims=True)  # (N, 1)

    for init_id in range(int(max(1, n_init))):
        # random init (good enough for small batches)
        init_idx = rng.choice(N, size=k, replace=False)
        centers = x[init_idx].copy()  # (k, D)

        labels = np.zeros(N, dtype=np.int64)
        for _ in range(int(max_iter)):
            c_norm2 = (centers * centers).sum(axis=1, keepdims=False)[None, :]  # (1, k)
            dist2 = x_norm2 + c_norm2 - 2.0 * (x @ centers.T)  # (N, k)
            new_labels = dist2.argmin(axis=1).astype(np.int64)

            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels

            # recompute centers
            for c in range(k):
                mask = labels == c
                if not np.any(mask):
                    centers[c] = x[rng.integers(0, N)]
                else:
                    centers[c] = x[mask].mean(axis=0)

        # inertia
        c_norm2 = (centers * centers).sum(axis=1, keepdims=False)[None, :]
        dist2 = x_norm2 + c_norm2 - 2.0 * (x @ centers.T)
        inertia = float(dist2[np.arange(N), labels].sum())

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    if best_labels is None:
        raise RuntimeError("[DOS] KMeans fallback failed unexpectedly.")
    return best_labels


def forward_logits_and_feats(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, penultimate_features) for a batch.

    Preferred path (ViT-like models):
      feats = model.forward_head(model.forward_features(x), pre_logits=True)
      logits = model.head(feats)

    Fallback path:
      attach a forward hook to model.head and capture its input as feats.
    """

    if hasattr(model, "forward_features") and hasattr(model, "forward_head") and hasattr(model, "head"):
        feats_tokens = model.forward_features(x)
        feats = model.forward_head(feats_tokens, pre_logits=True)
        logits = model.head(feats)
        return logits, feats

    # --- generic fallback via head hook ---
    if not hasattr(model, "head"):
        raise ValueError("[DOS] model has no `head`; cannot extract penultimate features.")

    feats_holder = {}

    def _hook(_module, inp, _out):
        # inp[0] is the input to the head (penultimate feature)
        feats_holder["feats"] = inp[0]

    handle = model.head.register_forward_hook(_hook)
    try:
        logits = model(x)
    finally:
        handle.remove()

    feats = feats_holder.get("feats", None)
    if feats is None:
        raise RuntimeError("[DOS] Failed to capture features via head hook.")

    return logits, feats


def compute_hardness_from_logits(
    logits: torch.Tensor,
    mode: str = "msp",
    *,
    energy_temperature: float = 1.0,
) -> torch.Tensor:
    """Compute 'hardness' score for candidates.

    Higher hardness == more ID-like / harder outliers.

    - msp: max softmax probability
    - maxlogit: max logit
    - energy: -E(x) where E(x) = -T * logsumexp(logits/T); so hardness = T*logsumexp(logits/T)
    """

    mode = str(mode).lower().strip()
    if mode == "msp":
        prob = torch.softmax(logits, dim=1)
        return prob.max(dim=1).values
    if mode == "maxlogit":
        return logits.max(dim=1).values
    if mode == "energy":
        T = float(energy_temperature)
        return T * torch.logsumexp(logits / T, dim=1)
    raise ValueError(f"[DOS] Unsupported hardness mode: {mode}. Use one of: msp, maxlogit, energy")


def collapse_logits_by_label(
    logits_full: torch.Tensor,
    labels_in_head: np.ndarray,
    num_classes: int,
    override_last_for_labels: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Collapse expanded-head logits into (B, num_classes) by max-pooling nodes per label.

    ICON may append extra nodes for existing labels; `labels_in_head` maps each head node -> label id.
    """

    if logits_full.shape[1] == num_classes:
        return logits_full

    B = logits_full.shape[0]
    device = logits_full.device
    dtype = logits_full.dtype
    pooled = torch.empty((B, int(num_classes)), device=device, dtype=dtype)

    for label in range(int(num_classes)):
        nodes = np.where(labels_in_head == label)[0]
        if nodes.size == 0:
            raise ValueError(f"[DOS] label {label} not found in labels_in_head")
        if nodes.size == 1:
            pooled[:, label] = logits_full[:, int(nodes[0])]
        else:
            idx = torch.tensor(nodes, device=device, dtype=torch.long)
            pooled[:, label] = torch.max(logits_full.index_select(1, idx), dim=1).values

    if override_last_for_labels is not None:
        for label in override_last_for_labels:
            nodes = np.where(labels_in_head == int(label))[0]
            if nodes.size > 0:
                pooled[:, int(label)] = logits_full[:, int(nodes[-1])]

    return pooled


def dos_select_indices(
    feats_norm: np.ndarray,
    hardness: np.ndarray,
    *,
    num_select: int,
    cfg: DOSConfig,
) -> np.ndarray:
    """Select indices using normalized feature clustering + per-cluster hard sampling."""

    if feats_norm.ndim != 2:
        raise ValueError(f"[DOS] feats_norm must be 2D (N,D). got: {feats_norm.shape}")
    if hardness.ndim != 1:
        raise ValueError(f"[DOS] hardness must be 1D (N,). got: {hardness.shape}")
    if feats_norm.shape[0] != hardness.shape[0]:
        raise ValueError("[DOS] feats_norm and hardness must have same N")

    N = int(feats_norm.shape[0])
    if N == 0:
        return np.array([], dtype=np.int64)

    num_select = int(num_select)
    num_select = max(1, min(num_select, N))

    k = int(cfg.num_clusters)
    k = max(1, min(k, num_select, N))

    fill_mode = str(cfg.fill_mode).lower().strip()
    if fill_mode not in {"random", "hard"}:
        raise ValueError(f"[DOS] Unsupported fill_mode: {cfg.fill_mode}")

    rng = np.random.default_rng(cfg.seed)

    # If clustering is degenerate, fall back to top-hard selection.
    if k <= 1 or N <= 1:
        order = np.argsort(-hardness)  # desc
        return order[:num_select].astype(np.int64)

    # KMeans clustering (use sklearn if available; otherwise fallback)
    feats_norm = feats_norm.astype(np.float32, copy=False)
    if _SKLearnKMeans is not None:
        random_state = None if cfg.seed is None else int(cfg.seed)
        kmeans = _SKLearnKMeans(n_clusters=k, n_init=int(cfg.n_init), random_state=random_state)
        labels = kmeans.fit_predict(feats_norm)
    else:
        labels = _kmeans_fit_predict_numpy(
            feats_norm,
            n_clusters=k,
            n_init=int(cfg.n_init),
            max_iter=50,
            seed=cfg.seed,
        )

    base = num_select // k
    rem = num_select % k

    selected: list[int] = []
    selected_mask = np.zeros(N, dtype=bool)

    for c in range(k):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            continue

        take = base + (1 if c < rem else 0)
        if take <= 0:
            continue

        # hard-first within each cluster
        idx_sorted = idx_c[np.argsort(-hardness[idx_c])]
        chosen = idx_sorted[:take]

        selected.extend(chosen.tolist())
        selected_mask[chosen] = True

    if len(selected) < num_select:
        remain = np.where(~selected_mask)[0]
        need = num_select - len(selected)
        if remain.size > 0:
            if fill_mode == "random":
                fill = rng.choice(remain, size=min(need, remain.size), replace=False)
            else:
                fill = remain[np.argsort(-hardness[remain])][:need]
            selected.extend(fill.tolist())

    return np.asarray(selected[:num_select], dtype=np.int64)


@torch.no_grad()
def dos_select_batch(
    model: torch.nn.Module,
    x_candidates: torch.Tensor,
    *,
    num_select: int,
    cfg: DOSConfig,
    labels_in_head: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    override_last_for_labels: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """High-level helper: run model to get logits+feats, then return selected batch tensor."""

    model_was_training = model.training
    model.eval()
    try:
        logits_full, feats = forward_logits_and_feats(model, x_candidates)

        # Optionally collapse expanded head logits (e.g., ICON)
        logits_for_hardness = logits_full
        if labels_in_head is not None and num_classes is not None:
            logits_for_hardness = collapse_logits_by_label(
                logits_full,
                labels_in_head=labels_in_head,
                num_classes=int(num_classes),
                override_last_for_labels=override_last_for_labels,
            )

        hard = compute_hardness_from_logits(
            logits_for_hardness,
            mode=cfg.hardness,
            energy_temperature=cfg.energy_temperature,
        )

        feats_norm = F.normalize(feats, dim=-1).detach().cpu().numpy()
        hard_np = hard.detach().cpu().numpy()

        idx = dos_select_indices(feats_norm, hard_np, num_select=int(num_select), cfg=cfg)
        idx_t = torch.as_tensor(idx, device=x_candidates.device, dtype=torch.long)
        return x_candidates.index_select(0, idx_t)
    finally:
        model.train(model_was_training)

