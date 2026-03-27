"""
Option 2: Graded-Weight NT-Xent Loss

Batch layout (produced by tricontext_collate):
    [tumor_0 … tumor_{B-1} | peri_0 … peri_{B-1} | par_0 … par_{B-1}]
    total = 3B

Asymmetric same-lesion treatment (per center-type pair):

    Center pair        | same-lesion weight  | meaning
    -------------------|---------------------|---------------------------------
    tumor  ↔ peri      | peri_weight  (0.4)  | soft denominator contribution
    tumor  ↔ par       | 0.0                 | fully masked (hard ignore)
    peri   ↔ par       | 0.0                 | fully masked (hard ignore)

Cross-lesion pairs (any center types, different lesion index):
    weight = 1.0  — standard repulsion in denominator

Loss:
    L_i = -log [ exp(sim(z1_i, z2_i)/τ) / Σ_j w_{i,j} · exp(sim(z1_i, z2_j)/τ) ]

where w_{i,j}:
    1.0          if i == j            (positive, diagonal — but not summed explicitly)
    1.0          if j%B != i%B        (different lesion — standard negative)
    peri_weight  if same-lesion & centre-pair ∈ {tumor↔peri}
    0.0          if same-lesion & centre-pair ∈ {tumor↔par, peri↔par}

Biological motivation:
    Peritumoral micro-environment shares immune infiltration patterns with tumor core
    (relevant for PDL1 and survival), but is not identical.  Soft down-weighting
    (rather than hard collapse) preserves the distinction while reducing spurious
    repulsion between spatially adjacent patches of the same lesion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradedContextNTXentLoss(nn.Module):
    """
    NT-Xent loss with per-center-pair denominator weights for 3-context batches.

    Args:
        temperature:  softmax temperature (default 0.1)
        peri_weight:  denominator weight for same-lesion tumor↔peri pairs.
                      0.0 = fully ignored, 1.0 = standard NT-Xent, (0,1) = soft.
                      Default 0.4.
    """

    def __init__(self, temperature: float = 0.1, peri_weight: float = 0.4):
        super().__init__()
        if abs(temperature) < 1e-8:
            raise ValueError(f"temperature must be non-zero, got {temperature}")
        if not (0.0 <= peri_weight <= 1.0):
            raise ValueError(f"peri_weight must be in [0, 1], got {peri_weight}")
        self.temperature = temperature
        self.peri_weight = peri_weight

        # 3×3 center-type weight matrix for SAME-LESION off-diagonal pairs
        # center indices: 0=tumor, 1=peri, 2=par
        cw = torch.zeros(3, 3)
        cw[0, 1] = peri_weight   # tumor → peri
        cw[1, 0] = peri_weight   # peri  → tumor
        # all others remain 0.0 (masked)
        self.register_buffer("center_weights", cw)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: [3B, D]  projection-head outputs (un-normalised)
                    order: [tumor×B | peri×B | par×B]

        Returns:
            Scalar graded NT-Xent loss.
        """
        n = z1.shape[0]
        assert n % 3 == 0, f"Batch size must be divisible by 3, got {n}"
        B = n // 3
        device = z1.device

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Pairwise logits [3B, 3B]
        logits = torch.mm(z1, z2.t()) / self.temperature

        # Index helpers
        idx = torch.arange(n, device=device)
        lesion_idx = idx % B             # lesion identity:  0…B-1
        center_idx = idx // B            # center type:      0=tumor,1=peri,2=par

        # Same-lesion mask (excludes diagonal implicitly handled below)
        same_lesion = lesion_idx.unsqueeze(0) == lesion_idx.unsqueeze(1)   # [3B,3B]
        diag = torch.eye(n, device=device, dtype=torch.bool)

        # Build weight matrix
        # Default: 1.0 everywhere (cross-lesion repulsion)
        weight_mat = torch.ones(n, n, device=device)

        # For same-lesion off-diagonal pairs: look up center-pair weight
        # vectorised: cw_row = center_weights[center_i, center_j] for all pairs
        ci = center_idx.unsqueeze(1).expand(n, n).cpu()   # [3B,3B]
        cj = center_idx.unsqueeze(0).expand(n, n).cpu()   # [3B,3B]
        center_pair_w = self.center_weights[ci, cj].to(device)  # [3B,3B]

        # Apply same-lesion override (keep diagonal separately — it's the positive)
        same_lesion_offdiag = same_lesion & ~diag
        weight_mat[same_lesion_offdiag] = center_pair_w[same_lesion_offdiag]

        # Diagonal: set weight to 0 so exp(diag) is excluded from denominator
        # (numerator is handled separately)
        weight_mat[diag] = 0.0

        # Numerator: exp(logits[i,i]) — the positive pair
        pos_logits = logits[diag]                   # [3B]

        # Denominator: Σ_j w_{i,j} * exp(logits_{i,j})
        # Fully masked (w=0) entries contribute zero → equivalent to -inf masking
        exp_logits = torch.exp(logits)
        denom = (exp_logits * weight_mat).sum(dim=1)   # [3B]

        # Loss: -log(exp_pos / denom)
        loss = -torch.log(torch.exp(pos_logits) / denom.clamp(min=1e-8))
        return loss.mean()
