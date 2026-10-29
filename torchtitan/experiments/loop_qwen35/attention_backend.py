# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility for PyTorch varlen layouts/causal windows and FA4."""

from functools import wraps


def install_fa4_causal_window_compat():
    """Normalize varlen layouts and redundant windows before FA4 registration.

    PyTorch varlen sends is_causal=True together with window_size=(-1, 0).
    Its FA4 backward adapter rejects every non-None window, including these
    redundant causal markers. Genuine sliding windows are left unchanged and
    retain the backend's unsupported-window error.

    The installed FA4 kernel also misreads strided varlen Q/K/V. Normalize
    only its varlen inputs at this boundary, in both forward and backward.
    Contiguous tensors are returned unchanged; FA2 and dense SDPA are untouched.
    """
    from torch.nn.attention import _fa4, current_flash_attention_impl

    original = _fa4._fa4_flash_attention_backward_impl
    if getattr(original, "_loop_causal_window_compat", False):
        return
    if current_flash_attention_impl() == "FA4":
        raise RuntimeError(
            "Install FA4 causal compatibility before activating the backend"
        )

    original_forward = _fa4._fa4_flash_attention_forward_impl

    @wraps(original_forward)
    def forward(query, key, value, cum_seq_q, cum_seq_k, *args, **kwargs):
        if cum_seq_q is not None:
            query, key, value = (tensor.contiguous() for tensor in (query, key, value))
        return original_forward(
            query, key, value, cum_seq_q, cum_seq_k, *args, **kwargs
        )

    @wraps(original)
    def backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        rng_state,
        unused,
        *,
        scale=None,
        window_size_left=None,
        window_size_right=None,
    ):
        if cum_seq_q is not None:
            grad_out, query, key, value, out = (
                tensor.contiguous() for tensor in (grad_out, query, key, value, out)
            )
        if (
            is_causal
            and window_size_left in (None, -1)
            and window_size_right in (None, -1, 0)
        ):
            window_size_left = window_size_right = None
        elif (
            not is_causal
            and window_size_left in (None, -1)
            and window_size_right in (None, -1)
        ):
            window_size_left = window_size_right = None
        return original(
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            rng_state,
            unused,
            scale=scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )

    backward._loop_causal_window_compat = True
    _fa4._fa4_flash_attention_forward_impl = forward
    _fa4._fa4_flash_attention_backward_impl = backward
