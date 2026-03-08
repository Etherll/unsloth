# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Chunked contrastive loss (InfoNCE / NTXent) that avoids materializing the
full similarity matrix.  Drop-in replacement for
`sentence_transformers.losses.MultipleNegativesRankingLoss`.

Supports non-square matrices (B_a != B_b) for multi-positive setups.
"""

import torch
import torch.nn.functional as F


class FusedContrastiveLoss(torch.autograd.Function):
    """
    Chunked forward + backward for contrastive (InfoNCE) loss.

    embeddings_a: (B_a, D) — anchors
    embeddings_b: (B_b, D) — positives (+ extra negatives when B_b > B_a)

    The positive pair for row i is at column i (diagonal).
    Columns beyond B_a are additional negatives.

    The two-pass forward avoids ever allocating a full (B_a, B_b) tensor:
      Pass 1 — row-wise max  (for numerically stable log-sum-exp)
      Pass 2 — log-sum-exp + positive-logit extraction
    """

    @staticmethod
    def forward(ctx, embeddings_a, embeddings_b, scale=20.0):
        B_a, _dim = embeddings_a.shape
        B_b = embeddings_b.shape[0]
        CHUNK = min(64, B_b)

        # ---- Pass 1: row-wise max ----
        row_max = torch.full(
            (B_a,), float("-inf"),
            device=embeddings_a.device, dtype=embeddings_a.dtype,
        )
        for j0 in range(0, B_b, CHUNK):
            j1 = min(j0 + CHUNK, B_b)
            sim = embeddings_a @ embeddings_b[j0:j1].t() * scale
            row_max = torch.max(row_max, sim.max(dim=1).values)

        # ---- Pass 2: log-sum-exp + positive logits ----
        row_lse = torch.zeros(B_a, device=embeddings_a.device, dtype=embeddings_a.dtype)
        pos_logits = torch.zeros(B_a, device=embeddings_a.device, dtype=embeddings_a.dtype)

        for j0 in range(0, B_b, CHUNK):
            j1 = min(j0 + CHUNK, B_b)
            sim = embeddings_a @ embeddings_b[j0:j1].t() * scale
            sim_shifted = sim - row_max.unsqueeze(1)
            row_lse += sim_shifted.exp().sum(dim=1)

            # Positive logits sit on the diagonal (row i, col i) for i < B_a
            diag_lo = max(0, j0)
            diag_hi = min(j1, B_a)
            for i in range(diag_lo, diag_hi):
                pos_logits[i] = sim_shifted[i, i - j0]

        row_lse = row_lse.log()
        loss = (-pos_logits + row_lse).mean()

        ctx.save_for_backward(embeddings_a, embeddings_b)
        ctx.scale = scale
        ctx.row_max = row_max
        ctx.row_lse = row_lse

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        embeddings_a, embeddings_b = ctx.saved_tensors
        scale = ctx.scale
        row_max = ctx.row_max
        row_lse = ctx.row_lse

        B_a = embeddings_a.shape[0]
        B_b = embeddings_b.shape[0]
        CHUNK = min(64, B_b)

        grad_a = torch.zeros_like(embeddings_a)
        grad_b = torch.zeros_like(embeddings_b)

        for j0 in range(0, B_b, CHUNK):
            j1 = min(j0 + CHUNK, B_b)
            b_chunk = embeddings_b[j0:j1]

            sim = embeddings_a @ b_chunk.t() * scale
            prob = (sim - row_max.unsqueeze(1) - row_lse.unsqueeze(1)).exp()

            # subtract one-hot for diagonal entries
            diag_lo = max(0, j0)
            diag_hi = min(j1, B_a)
            for i in range(diag_lo, diag_hi):
                prob[i, i - j0] -= 1.0

            prob = prob * (grad_output * scale / B_a)

            grad_a += prob @ b_chunk
            grad_b[j0:j1] += prob.t() @ embeddings_a

        return grad_a, grad_b, None


class FastMultipleNegativesRankingLoss(torch.nn.Module):
    """
    Drop-in replacement for
    ``sentence_transformers.losses.MultipleNegativesRankingLoss``
    that uses :class:`FusedContrastiveLoss` under the hood.
    """

    def __init__(self, model, scale=20.0, similarity_fct=None):
        super().__init__()
        if similarity_fct is not None:
            import warnings
            warnings.warn(
                "Unsloth: similarity_fct is ignored by FusedContrastiveLoss (cosine similarity is hardcoded).",
                stacklevel=2,
            )
        self.model = model
        self.scale = scale

    def forward(self, sentence_features, labels=None):
        reps = [
            self.model(sf)["sentence_embedding"] for sf in sentence_features
        ]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:], dim=0)

        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)

        return FusedContrastiveLoss.apply(embeddings_a, embeddings_b, self.scale)
