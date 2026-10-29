# Qwen3.5 Loop training

Train dense Qwen3.5 with shared-layer Loop and optional mHC using one TorchTitan
checkout. SFT, sigmoid DPO and one-layer MTP share the model, encoder and
distributed runtime. Swift and Megatron are not runtime dependencies.

## Install

The public environment targets Linux x86_64, Python 3.12.13 and PyTorch
2.11.0 with CUDA 13.0. It pins Transformers 5.18.0, Tokenizers 0.23.1,
Hugging Face Hub 1.31.0, HF Xet 1.5.2, Attention Gym 0.0.9,
FlashAttention 2.8.3 and CUTLASS DSL 4.6.2. The NVIDIA driver must support CUDA
13.0. Install the CUDA 13.0 toolkit (including nvcc), a C++ compiler and Git
before building CUDA extensions. The exercised host is Ubuntu 22.04.5 with
GCC 11.4.0 and nvcc 13.0.88.

From the repository root, using a conda-forge distribution such as Miniforge:

    conda create -n torchtitan-loop --override-channels -c conda-forge python=3.12.13 pip=26.1.2
    conda activate torchtitan-loop
    export CUDA_HOME=/usr/local/cuda-13.0
    export PATH="$CUDA_HOME/bin:$PATH"
    python -m pip install -c constraints-cu130.txt packaging psutil ninja wheel setuptools
    python -m pip install -c constraints-cu130.txt torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130 --no-deps
    python -m pip install -c constraints-cu130.txt torch torchvision
    MAX_JOBS=8 python -m pip install --no-build-isolation -c constraints-cu130.txt -e '.[loop-qwen35]' pytest
    python -m pip check
    python -m pytest tests/experiments/loop_qwen35 -q

Set CUDA_HOME to your toolkit location. An existing clean Python 3.12 virtual
environment can replace the two Conda commands. The
[PyTorch CUDA wheel selection](https://pytorch.org/get-started/previous-versions/#v2110)
and [FlashAttention build instructions](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3#installation-and-features)
explain the installation order: fetch the GPU wheels from the PyTorch index,
install their transitive dependencies, then compile FlashAttention
with --no-build-isolation. MAX_JOBS limits build-time CPU/RAM use.

Runtime dependencies are declared in the repository's pyproject.toml, with
Loop-specific dependencies in the loop-qwen35 extra. The root
constraints-cu130.txt pins the tested CUDA 13 runtime, build and test dependency
closure. The Conda command above selects the interpreter; a separate Conda
environment file is unnecessary. This is a version-pinned installation recipe,
not a binary environment export or a hash-locked container image. It contains no
private paths, model weights, data or Swift/Megatron dependencies. Other GPU/CUDA
combinations require their own kernel regression checks.

Activate the environment before launching. The scripts use its python for
both torch.distributed.run and every worker, without selecting a private Conda
path. The worker entry is a Python package module:
python -m torch.distributed.run ... -m torchtitan.experiments.loop_qwen35.train.
The first module launches distributed workers; the second runs the shared SFT/DPO
entry with normal package imports.

## Start SFT or DPO

From the repository root:

    MODEL_PATH=Qwen/Qwen3.5-9B DATASET=/data/sft.jsonl bash examples/train/loop_qwen35/mhc2_loop2_titan.sh
    MODEL_PATH=/models/Qwen3.5-9B DATASET=/data/preferences.jsonl bash examples/train/loop_qwen35/mhc2_loop2_dpo_titan.sh

MODEL_PATH accepts a Hugging Face model ID or a local snapshot. A Hub ID downloads
assets and safetensors into the normal Hugging Face cache; authenticate separately
for gated/private models. Downloaded Python code is not executed.

Defaults: 8 GPUs, CP4/DP2, 8K context, Loop interval [4,28) repeated twice,
two mHC streams, frozen vision and full activation checkpointing. DP2 means two
independent data streams. FSDP shards parameters, gradients and optimizer state
over dp_shard * CP = 2 * 4 = 8 GPUs; CP4 splits each stream's context. With TP1,
PP1 and dp_replicate=1, world_size = dp_shard * CP. Both examples
start from HF weights at step zero. Original base weights initialize newly
introduced routers. Set NPROC_PER_NODE, CP, MAX_LENGTH, GLOBAL_BATCH_SIZE,
TRAIN_ITERS, LR and OUTPUT_DIR directly. Multi-node runs also set NNODES,
NODE_RANK, MASTER_ADDR and a shared OUTPUT_DIR.

Without DATASET, the scripts use the small synthetic examples beside them.
These demonstrate the format and are only smoke-test data, not a quality benchmark.

For a smaller first run, retain four layers from the original HF model:

    MODEL_PATH=/models/Qwen3.5-9B NUM_LAYERS=4 MHC_START=0 MHC_END=4 LOOP_CONFIG=examples/train/loop_qwen35/loop2_4layers.json TRAIN_ITERS=10 SAVE_STEPS=5 OUTPUT_DIR=output/sft-smoke bash examples/train/loop_qwen35/mhc2_loop2_titan.sh

The same overrides work with the DPO script. Hidden size, vocabulary and
attention/GDN dimensions remain unchanged; a reduced-depth model is still large.

For full-depth 256K SFT on eight GPUs with sufficient memory, keep all 32 backbone layers and
split the vocabulary loss into more chunks to bound peak logits memory:

    MODEL_PATH=/models/Qwen3.5-9B DATASET=/data/sft-arrow MAX_LENGTH=262144 LOSS_CHUNKS=256 CP=4 GLOBAL_BATCH_SIZE=2 TRAIN_ITERS=10 SAVE_STEPS=10 OUTPUT_DIR=output/sft-256k bash examples/train/loop_qwen35/mhc2_loop2_titan.sh

This uses the same launcher and Loop configuration, with no NUM_LAYERS override.
Each DP stream packs up to 262144 tokens and each optimizer step has 524288 token
slots. Packed documents retain separate attention boundaries; a full 256K packed
microbatch does not imply a single 256K document. Use long conversations to test
the latter. Packing and document lengths affect both memory and timing.

## Data and loss

Both objectives accept raw JSONL files and Arrow directories saved with
datasets.Dataset.save_to_disk. No preparation command is required.
For multiple sources, append --dataset /data/a.jsonl /data/b-arrow to a script.

SFT example:

    {"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4."}]}

DPO example:

    {"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4."}],"rejected_response":"5."}

rejected_messages can instead hold a complete rejected conversation. Branches
must share the conditioning prompt, tools and images. Static images use an images
list with <image> placeholders or structured image content. Tool messages support
assistant tool_calls and tool observations; tool_call/tool_response roles use the
same normalization. Message-level loss/loss_scale masks must be binary.

SFT scores assistant content and end markers, preserving historical reasoning.
DPO defaults to loss_scale=default: score all supervised assistant turns and
ignore empty thinking prefixes. last_round scores only the final response under
shared history. Sequence log-probabilities are SUMMED over selected tokens;
sigmoid DPO is averaged over globally valid pairs, including partial epoch tails.
Defaults are beta=0.1 and label_smoothing=0. Other DPO losses are rejected.

DPO batch sizes count pairs. Each branch is limited to MAX_LENGTH; the physical
token budget is 2 * MAX_LENGTH * MICRO_BATCH_SIZE. Pairs remain intact and
overlong pairs fail with a source ID. SFT greedy packing skips whole overlong or
unlabelled rows with warnings, rather than truncating conversations.
Balanced packing is available explicitly for data with a lengths column. An
optional length_cache supplies verified lengths; mismatches are errors.

Offline length builders and online training use the same Qwen35Encoder.
Use the same pinned environment, model/tokenizer assets, chat template and image
settings for both. Token IDs and unshifted labels must match exactly; training
then shifts labels once per document and masks its last position. Cache
fingerprints include rendering and image-processing dependencies. A cache from
a different encoding contract is rejected and must be rebuilt; matching lengths
alone do not establish token or label equivalence.

The example learning rates are constant: SFT 4e-5, DPO 5e-7; weight decay is
respectively 0.01 and 0.1. Backbone and routers have separate optimizer groups.
--decay-ratio 1 enables full cosine decay; intermediate values decay over the
final fraction of steps after warmup.

## MTP

MTP is supported for compatible Qwen3.5-9B HF checkpoints containing one pretrained
MTP layer. Enable it in the SFT entry:

    MODEL_PATH=/models/Qwen3.5-9B MTP_NUM_LAYERS=1 MTP_TRAINING_MODE=mtp_only bash examples/train/loop_qwen35/mhc2_loop2_titan.sh

mtp_only freezes every non-MTP parameter, including embeddings/head and routers.
joint trains both objectives. Default auxiliary scaling is 1.0 for MTP-only and
0.1 for joint; override with --mtp-loss-scaling-factor. Targets do not cross
packed document boundaries; visual targets are masked. MTP plus DPO is rejected.

## Resume and evaluation

Native checkpoints save model, optimizer, scheduler, data cursor and RNG.
DPO additionally persists an immutable frozen-reference snapshot. complete.json
marks a completed checkpoint. Resume requires the same model, data/encoding,
schedule and parallel topology.

Reuse the original model/data/Loop overrides and append:

    --resume true --resume-checkpoint output/sft-smoke/checkpoints/step-5

Use the same total TRAIN_ITERS for ordinary SFT/DPO. Stage conversion, stream
duplication and topology-changing resume are outside this public entry.
Old two-repository experiment checkpoints are not a supported resume format.
Keep encoder package versions unchanged for exact native resume: Transformers
and Tokenizers versions are part of the data fingerprint, because upgrades can
change token boundaries. A changed fingerprint is rejected during data restore.

To start a new DPO stage from a compatible SFT native checkpoint, pass
--initial-rank-weights <step-directory>. This imports weights only; optimizer,
data cursor and RNG start fresh. The default reference is an independent frozen
copy of the initialized policy. --ref-model or --ref-checkpoint selects an
explicit compatible reference.

Append --val-dataset <path> --eval-steps 50 --eval-batches 4 for evaluation.
Use separate training and validation data. Evaluation restores RNG and uses
the fixed Loop eval_count, preserving training sampling.

## Tests and boundaries

    python -m pytest tests/experiments/loop_qwen35 -q

Tests cover Loop sharing, mHC, token masks, DPO values and gradients, packing,
MTP targets/weights/frozen parameters and checkpoint resume. GPU commands and
their validation scope are documented in the tests directory. Use deterministic
settings and warm kernel caches for numerical comparisons.

This entry supports TP1/PP1, CP and FSDP/HSDP data parallelism. Video/audio,
arbitrary model families, LoRA, non-sigmoid DPO and topology-changing resume
are outside its scope. See [ARCHITECTURE.md](ARCHITECTURE.md) for the code map.

For hardware absent from the peak-performance table, set GPU_PEAK_FLOPS to the
per-device dense BF16 peak in FLOP/s when reporting MFU. Without a known or
configured peak, throughput and TFLOPS are reported and MFU is omitted.

## License and model assets

The Loop implementation and tests use the repository's [BSD-3-Clause license](../../../LICENSE).
The encoder supplies training supervision spans and reads tool prompt instructions
from the selected HF model's chat template. Model assets and external dependencies
retain their own licenses; no model template is bundled in this repository.

Offline length building and online SFT/DPO share the same encoder and renderer.
Their fingerprints include both source files and the model assets. After an
encoding-source update, rebuild length caches and use the original checkout to
resume checkpoints whose stored data fingerprint belongs to that version.
