# LLAISYS Assignment Report

## Summary

This submission implements the tensor and operator assignments, Qwen2 inference with a dynamic KV cache, and the NVIDIA CUDA backend. The CUDA backend provides runtime APIs and FP32, FP16, and BF16 implementations of add, argmax, embedding, linear, RMSNorm, RoPE, self-attention, and SwiGLU. Linear uses cuBLAS; the other operators use CUDA kernels.

The target model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. Model weights are loaded from safetensors in Python and copied into the C++ backend. Prefill and token-by-token decoding run in C++ without using a Python inference framework.

## Tested Environment

| Component | Version / configuration |
| --- | --- |
| OS | Linux x86_64 |
| GPU | NVIDIA GeForce RTX 4090 D, compute capability 8.9 |
| NVIDIA driver | 595.58.03 |
| CUDA toolkit | 13.0 (`nvcc` 13.0.88) |
| Xmake | 2.8.7 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.6.0 |
| Model dtype | BF16 |

## Reproduction

### 1. Install dependencies

Install a C++17 compiler, CUDA, Xmake, and Python 3.9 or newer. Then install the Python package dependencies:

```bash
pip install ./python/
```

### 2. Build and install the NVIDIA backend

```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
```

When building as root inside a container, set `XMAKE_ROOT=y` for the Xmake commands.

### 3. Download the model

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    local_dir="/path/to/DeepSeek-R1-Distill-Qwen-1.5B",
)
PY
```

The tested safetensors file contains 339 BF16 tensors and is 3,554,214,621 bytes.

### 4. Run runtime and operator tests

CPU:

```bash
python test/test_runtime.py --device cpu
python test/test_tensor.py
```

NVIDIA:

```bash
python test/test_runtime.py --device nvidia

for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    python "test/ops/${op}.py" --device nvidia
done
```

If a machine has several GPUs, `CUDA_VISIBLE_DEVICES=<index>` can be used to select an idle GPU.

### 5. Run model inference parity test

```bash
python test/test_infer.py \
    --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --device nvidia \
    --test
```

The `--test` option selects greedy argmax generation and verifies that the complete LLAISYS token sequence equals the Transformers reference sequence.

## Results

| Test | Result |
| --- | --- |
| CPU runtime | Passed |
| Tensor load/view/permute/slice | Passed |
| Tiny Qwen2 CPU parity | Passed |
| NVIDIA runtime | Passed |
| NVIDIA add, argmax, embedding | Passed for FP32, FP16, and BF16 |
| NVIDIA linear | Passed for FP32, FP16, and BF16, including the 512x4096 by 4096x4096 case |
| NVIDIA RMSNorm, RoPE, self-attention, SwiGLU | Passed for FP32, FP16, and BF16 |
| Tiny Qwen2 NVIDIA prefill and decode parity | Passed |
| KV cache growth from 256 to 512 tokens | Passed and matched Transformers |
| DeepSeek-R1-Distill-Qwen-1.5B greedy generation | Passed; complete token sequence matched Transformers through EOS token 151643 |

For one functional test using the prompt `Who are you?`, Transformers took approximately 1.19 seconds and LLAISYS took approximately 0.28 seconds. Both measurements exclude model loading and are single-run observations, not a formal benchmark.

## Platform Status

| Platform | Runtime | Operators | Qwen2 inference | Status / notes |
| --- | --- | --- | --- | --- |
| CPU | Supported | Supported | Functional | Tensor and operator paths are implemented. Tiny Qwen2 parity passed; full 1.5B CPU performance was not evaluated because the CPU linear implementation is naive. |
| NVIDIA CUDA | Supported | Supported | Supported | Verified on RTX 4090 D with CUDA 13.0. FP32, FP16, and BF16 operators pass. Qwen2 currently uses one GPU. |
| Iluvatar | Not implemented | Not implemented | Not implemented | No backend is included in this submission. |
| Metax | Not implemented | Not implemented | Not implemented | No backend is included in this submission. |
| Moore Threads | Not implemented | Not implemented | Not implemented | No backend is included in this submission. |

## Known Limitations

- Qwen2 generation currently implements greedy argmax sampling only.
- Qwen2 currently supports one device per model instance; tensor parallelism is not implemented.
- The assignment asks for two CUDA or CUDA-like accelerator platforms. This submission completes the NVIDIA platform only; a second accelerator backend remains outstanding.
- Linux was tested. Windows CUDA compilation was not verified.
