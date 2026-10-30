# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Pair-aware DPO execution on the existing Loop distributed runtime."""

import copy
from dataclasses import replace
from pathlib import Path

import spmd_types as spmd
import torch

from torchtitan.experiments.SenseNova_Loop_9B_Preview.engine import (
    LoopTrainingEngine,
)
from torchtitan.components.sequence_logps import (
    sequence_logps,
    weighted_sequence_backward,
)
from .loss import complete_sequence_scores, dpo_loss
from .reference import ReferenceRuntime, preserve_rng


class DPOTrainingEngine(LoopTrainingEngine):
    def prepare_dpo_step(
        self, global_pairs, global_valid_tokens, *, num_accumulation_steps
    ):
        if global_pairs <= 0 or global_valid_tokens <= 0:
            raise ValueError("DPO requires globally valid pairs and completion tokens")
        # Native auxiliary accounting keeps token units; only DPO uses pairs.
        return self.prepare_step(
            global_valid_tokens, num_accumulation_steps=num_accumulation_steps
        )

    def _initialize_model(self, model_spec, **kwargs):
        self.model_spec = model_spec
        super()._initialize_model(model_spec, **kwargs)
        self.reference = None
        for model in self.model_parts:
            model._skip_lm_head = True
            if any(
                isinstance(module, torch.nn.Dropout) and module.p
                for module in model.modules()
            ):
                raise ValueError(
                    "Loop DPO requires dropout=0 for policy/reference scoring"
                )

    def initialize_reference(self, *, from_snapshot=False):
        if self.reference is not None:
            return
        options = self.config.checkpointer.resolved_config["dpo"]
        source = options["ref_checkpoint"] or options["ref_model"]
        self.reference = ReferenceRuntime(self, copy_policy=not from_snapshot)
        if source and not from_snapshot:
            loader = copy.copy(self.checkpointer)
            loader.parts = self.reference.parts
            native = (Path(source) / "complete.json").is_file()
            import json

            strict = native or "loop_settings" in json.loads(
                (Path(source) / "config.json").read_text()
            )
            loader.config = replace(
                loader.config,
                initial_load_path=source,
                initial_rank_weights=source if native else "",
                strict_initial_weights=strict,
            )
            with preserve_rng():
                if native:
                    import json

                    metadata = json.loads((Path(source) / "complete.json").read_text())
                    if metadata["topology"] != loader.topology:
                        raise ValueError(
                            "Reference checkpoint requires identical topology, Loop and mHC"
                        )
                    loader._load_rank_weights()
                else:
                    loader._load_hf()

    def _inputs(self, microbatch):
        model = self.model_parts[0]
        inputs, labels, kwargs = model.preprocess_inputs(
            microbatch.to_input_dict(self.device, non_blocking=True),
            parallel_dims=self.parallel_dims,
            parallelism=self.config.parallelism,
            max_num_documents=self.max_num_documents,
            max_context_length=self.config.training.max_context_length,
            **self.preprocess_inputs_kwargs,
        )
        ids = microbatch.sequence_ids.to(self.device)
        if self.parallel_dims.cp > 1:
            ids = ids.chunk(self.parallel_dims.cp)[
                self.parallel_dims.get_mesh("cp").get_local_rank()
            ]
        if ids.shape != labels.shape:
            raise ValueError("CP response metadata does not match preprocessed labels")
        return inputs, labels, kwargs, ids

    def preference_microbatch(
        self, microbatch, *, global_pairs, loop_count, backward, accumulation_index=0
    ):
        if backward:
            if accumulation_index == 0:
                self.loss_is_finite = torch.ones(
                    (), dtype=torch.int32, device=self.device
                )
            if self.parallel_dims.dp_replicate_enabled:
                for model in self.model_parts:
                    model.set_requires_all_reduce(
                        accumulation_index == self.num_accumulation_steps - 1
                    )
        chunks = self.loss_fn.config.num_chunks
        cp_mesh = self.parallel_dims.get_optional_mesh("cp")
        cp_group = cp_mesh.get_group() if cp_mesh is not None else None
        valid = microbatch.pair_valid.to(self.device)
        count = valid.numel() * 2
        with self.train_context():
            inputs, labels, kwargs, ids = self._inputs(microbatch)
            kwargs["loop_count"] = loop_count
            with preserve_rng(), torch.no_grad():
                try:
                    hidden = self.reference.model(inputs, **kwargs)
                    reference = sequence_logps(
                        hidden,
                        self.reference.model.lm_head,
                        labels,
                        ids,
                        count,
                        num_chunks=max(1, chunks),
                    )
                    with spmd.local():
                        reference = complete_sequence_scores(
                            reference, cp_group
                        ).reshape(-1, 2)
                    del hidden
                finally:
                    self.reference.reshard()
            with torch.set_grad_enabled(backward):
                hidden = self.model_parts[0](inputs, **kwargs)
                if chunks or not backward:
                    with torch.no_grad():
                        policy = sequence_logps(
                            hidden.detach(),
                            self.model_parts[0].lm_head,
                            labels,
                            ids,
                            count,
                            num_chunks=max(1, chunks),
                        )
                else:
                    policy = sequence_logps(
                        hidden, self.model_parts[0].lm_head, labels, ids, count
                    )
                with spmd.local():
                    policy = complete_sequence_scores(policy, cp_group).reshape(-1, 2)
                    loss, coefficients, metrics = dpo_loss(
                        policy,
                        reference,
                        valid,
                        beta=self.loss_fn.config.beta,
                        global_pairs=global_pairs,
                    )
                if backward:
                    if chunks:
                        weighted_sequence_backward(
                            hidden,
                            self.model_parts[0].lm_head,
                            labels,
                            ids,
                            coefficients.flatten(),
                            num_chunks=chunks,
                        )
                    else:
                        with spmd.no_typecheck():
                            loss.backward()
            if backward:
                with spmd.local():
                    finite = (
                        torch.isfinite(policy).all()
                        & torch.isfinite(reference).all()
                        & torch.isfinite(loss)
                    )
                    self.loss_is_finite.logical_and_(finite)
                self.ntokens_seen += sum(microbatch.document_lengths)
        metrics["chosen_tokens"] = (
            microbatch.completion_counts[:, 0].sum().to(self.device)
        )
        metrics["rejected_tokens"] = (
            microbatch.completion_counts[:, 1].sum().to(self.device)
        )
        return {key: value.detach() for key, value in metrics.items()}
