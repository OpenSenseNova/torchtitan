# Code map

Loop-specific runtime code lives in torchtitan/experiments/loop_qwen35.
Shared TorchTitan model components, parallelism and optimizers retain their
existing upstream locations.

| Module | Responsibility |
| --- | --- |
| config.py | CLI and supported configuration combinations |
| train.py | Resolve HF assets, assemble runtime, select the objective |
| model_config.py / config_registry.py | Validate architecture and create recipes |
| models/qwen35_loop.py | Shared physical layers, Loop scheduling and distributed model |
| mhc/model.py / fused kernels | Residual streams and mHC routing |
| models/qwen35_mtp.py / mtp_data.py / loss.py | MTP weights, targets and auxiliary objective |
| encoding.py / rendering.py | Shared text, image, tool rendering and supervision |
| data.py / packing / prefetch / length_cache | Raw sources, document ownership and optional CPU lookahead |
| engine.py / trainer.py | SFT/MTP execution, optimization and evaluation |
| dpo/encoding.py / data.py / validation.py | Complete preference pairs and their conditioning context |
| dpo/engine.py / loss.py / reference.py | Policy/reference scores and DPO gradients |
| dpo/trainer.py / metrics.py | Pair-count batching, global normalization and evaluation |
| checkpoint.py / dpo/checkpoint.py | Same-topology resume and reference persistence |
| mhc/schedule.py / metrics / profiling | Schedules and optional runtime observations |

## Model and update

Each physical decoder layer is stored once. Loop executes the configured interval
repeatedly using the same parameter objects. Distributed training peers share the
sampled count; evaluation uses eval_count. Recomputations keep the selected count.
mHC changes residual routing without duplicating decoder weights.

SFT preserves document boundaries and shifts labels once. The engine normalizes
summed loss by global valid tokens before updating. MTP aligns an additional
prediction target within each document. In MTP-only mode, frozen parameters have
neither gradients nor optimizer state.

DPO packs chosen/rejected pairs intact. Reference and policy use the same Loop
count and inputs. The reference is gradient-free. Chunked policy log-probabilities
are combined across CP before sigmoid DPO normalization by global valid pairs.
Dummy pairs contribute no objective. The reference never follows policy updates.

## State

Dataset order, packing, tokenizer/template and image preprocessing form the saved
data contract. Resume restores the cursor and RNG without silently starting a
new epoch. Model/MTP contracts check tensor keys, shapes, trainability and optimizer
membership. Internal stage migration, stream duplication and machine-specific
launch history are intentionally absent.
