# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Strict DPO resume with a content-addressed, immutable reference snapshot."""

import hashlib
import json
from dataclasses import dataclass

import torch
import torch.distributed as dist

from torchtitan.experiments.SenseNova_Loop_9B_Preview.checkpoint import (
    LoopCheckpoint,
)
from .metrics import synchronized_error
from .reference import local_tensor, physical_state, preserve_rng


def digest_state(state):
    digest = hashlib.sha256()
    for weights in state:
        for name, value in sorted(weights.items()):
            value = value.detach().cpu().contiguous()
            digest.update(
                json.dumps([name, list(value.shape), str(value.dtype)]).encode()
            )
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def snapshot_state(parts):
    return [
        {
            name: local_tensor(value).detach().cpu().clone()
            for name, value in physical_state(part).items()
        }
        for part in parts
    ]


class DPOCheckpoint(LoopCheckpoint):
    @dataclass(kw_only=True, slots=True)
    class Config(LoopCheckpoint.Config):
        strict_initial_weights: bool = False

    def contract(self):
        resolved = self.config.resolved_config
        return dict(
            version=1,
            algorithm=resolved["dpo"],
            data=self.dataloader.fingerprint,
            topology=self.topology,
            base_model=resolved["base_model"],
        )

    def ensure_reference_snapshot(self):
        if hasattr(self, "reference_manifest"):
            return self.reference_manifest
        with preserve_rng():
            state = snapshot_state(self.engine.reference.parts)
            shard_hash = digest_state(state)
            hashes = [None] * dist.get_world_size()
            dist.all_gather_object(hashes, shard_hash)
            identity = hashlib.sha256(
                json.dumps([self.topology, hashes], sort_keys=True).encode()
            ).hexdigest()
            relative = f"reference/{identity}"
            folder = self.folder.parent / relative
            manifest = dict(
                version=1,
                format="physical_local_shards",
                identity=identity,
                path=relative,
                shards=hashes,
                topology=self.topology,
            )
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"rank-{dist.get_rank()}.pt"
            error = None
            try:
                if path.exists():
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if digest_state(saved) != shard_hash:
                        raise ValueError("Existing reference snapshot was modified")
                else:
                    temporary = path.with_suffix(".pt.tmp")
                    torch.save(state, temporary)
                    temporary.rename(path)
            except Exception as exception:
                error = exception
            synchronized_error(error, self.engine.device)
            if dist.get_rank() == 0:
                temporary = folder / "complete.json.tmp"
                temporary.write_text(json.dumps(manifest, sort_keys=True) + "\n")
                temporary.rename(folder / "complete.json")
            dist.barrier()
            self.reference_manifest = manifest
        return manifest

    def extra_metadata(self):
        return dict(
            dpo=dict(
                contract=self.contract(), reference=self.ensure_reference_snapshot()
            )
        )

    def validate_extra_metadata(self, metadata, source):
        saved = metadata.get("dpo")
        if saved is None:
            raise ValueError(
                "SFT checkpoint is not a DPO resume; use --initial-rank-weights for a new stage"
            )
        current = self.contract()
        mismatch = [
            key for key in current if saved["contract"].get(key) != current[key]
        ]
        if mismatch:
            raise ValueError(
                "Incompatible DPO checkpoint fields: " + ", ".join(mismatch)
            )
        reference = saved["reference"]
        if reference.get("format") != "physical_local_shards":
            raise ValueError("Unsupported DPO reference snapshot format")
        root = source.parent.parent.resolve()
        folder = (root / reference["path"]).resolve()
        if (
            not folder.is_relative_to(root)
            or reference["path"] != f"reference/{reference['identity']}"
        ):
            raise ValueError("Invalid DPO reference snapshot path")
        error = None
        try:
            if json.loads((folder / "complete.json").read_text()) != reference:
                raise ValueError("Reference manifest mismatch")
            state = torch.load(
                folder / f"rank-{dist.get_rank()}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if digest_state(state) != reference["shards"][dist.get_rank()]:
                raise ValueError("Reference snapshot content fingerprint mismatch")
            identity = hashlib.sha256(
                json.dumps(
                    [self.topology, reference["shards"]], sort_keys=True
                ).encode()
            ).hexdigest()
            if identity != reference["identity"]:
                raise ValueError("Reference snapshot identity mismatch")
            self._loaded_reference_state = state
        except Exception as exception:
            error = exception
        synchronized_error(error, self.engine.device)

    def load_extra_state(self, state, metadata, source):
        self.engine.initialize_reference(from_snapshot=True)
        with torch.no_grad():
            for part, saved in zip(
                self.engine.reference.parts, self._loaded_reference_state, strict=True
            ):
                target = physical_state(part)
                if target.keys() != saved.keys():
                    raise ValueError("Reference snapshot model keys differ")
                for name, destination in target.items():
                    destination = local_tensor(destination)
                    value = saved[name]
                    if (
                        destination.shape != value.shape
                        or destination.dtype != value.dtype
                    ):
                        raise ValueError(
                            f"Reference shard shape/dtype mismatch: {name}"
                        )
                    destination.copy_(value)
        del self._loaded_reference_state
