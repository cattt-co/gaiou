"""Pure-PyTorch fallback for the Rotated_IoU CUDA vertex-sort kernel.

The original Rotated_IoU repo (lilanxiao) sorts the (<=8) vertices of the
intersection polygon counter-clockwise with a compiled CUDA extension
(``cuda_op.cuda_ext.sort_v`` -> ``sort_vertices``). That kernel only exists once
it has been built with nvcc, which the thesis did on the A40 server.

This module reproduces the *contract* of ``sort_v`` in plain PyTorch so the loss
imports and runs on CPU / Apple Silicon for development and verification. On the
GPU server, build the real kernel (``cd rotated_iou/cuda_op && python setup.py
install``) and ``box_intersection_2d`` will prefer it automatically.

Contract of ``sort_v(vertices_normalized, mask, num_valid)`` -> idx (B, N, 9):
    * vertices_normalized: (B, N, 24, 2), polygon-candidate vertices with their
      centroid subtracted (corners[0:8] + intersections[8:24]).
    * mask: (B, N, 24) bool, which candidates are real polygon vertices.
    * num_valid: (B, N) int, number of real vertices (<= 8).
    * returns long indices with structure (v0, v1, ..., v_{k-1}, v0, X, X, ...)
      i.e. the k valid vertices in CCW order, the first one repeated to close the
      polygon, then padding indices that point at zero-valued (masked-out)
      *intersection* slots so they contribute nothing to the shoelace area.
"""

import torch


def sort_v(vertices_normalized: torch.Tensor, mask: torch.Tensor, num_valid: torch.Tensor) -> torch.Tensor:
    """CPU/PyTorch reimplementation of the CUDA ``sort_vertices`` kernel."""
    B, N, V, _ = vertices_normalized.shape
    device = vertices_normalized.device

    # CCW order by angle around the (already-subtracted) centroid; invalid last.
    angle = torch.atan2(vertices_normalized[..., 1], vertices_normalized[..., 0])  # (B, N, V)
    angle = angle.masked_fill(~mask, float("inf"))
    order = torch.argsort(angle, dim=2)  # (B, N, V); first `k` entries are valid CCW

    # Padding must reference a vertex whose *value* is (0, 0). Only invalid
    # intersection slots (index >= 8) are zero; invalid corner slots (0..7) still
    # hold real coordinates, so they must NOT be used as padding.
    inter_invalid = ~mask.clone()
    inter_invalid[..., :8] = False
    # argmax returns the first True; there are always >= 8 such slots (16 inter - <=8 valid).
    pad_idx = torch.argmax(inter_invalid.to(torch.int8), dim=2)  # (B, N)

    ar = torch.arange(9, device=device).view(1, 1, 9)
    k = num_valid.long().unsqueeze(-1)  # (B, N, 1)

    base = torch.gather(order, 2, ar.clamp(max=V - 1).expand(B, N, 9))  # order[pos]
    close = order[..., 0:1].expand(B, N, 9)  # repeat first valid vertex to close
    padv = pad_idx.unsqueeze(-1).expand(B, N, 9)

    out = torch.where(ar < k, base, torch.where(ar == k, close, padv))
    return out.long()
