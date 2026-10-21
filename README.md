# llama-abominated

<div align="center">

<b>llama.cpp tuned for running big MoE models on a small GPU</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (branch `async-overlap`, based on upstream commit `781dbc5`) for hybrid CPU-GPU inference of mixture-of-experts models when most of the model has to stay in system RAM. It was built and tested on a laptop with a 4 GB GTX 1650 and an AVX2 CPU, running Qwen3.6-35B-A3B.

Everything that works in llama.cpp at that commit works here: the same GGUF files, tools, server API and backends. With the new code paths turned off (`GGML_CPU_IQK=0`, `GGML_SCHED_PREFETCH=0`, no `-ser`/`--moe-n-expert`), the output matches stock llama.cpp exactly. The iqk kernels are on by default; their results differ from stock only by rounding (see `tests/test-iqk-mul-mat-id.cpp`).

## What this fork adds

| Feature | What it does | Control |
|---|---|---|
| Faster CPU MoE kernels | AVX2 Q4_K/Q5_K/Q6_K `MUL_MAT_ID` kernels ported from [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp). Each weight block is unpacked once and reused for up to 8 tokens; experts with 32 or more tokens are converted to an interleaved 8-bit layout (near-lossless: the combined block scales are rounded to fp16) and run through a GEMM kernel. 1.15-1.4x faster prompt processing with the experts on the CPU. | On by default; `GGML_CPU_IQK=0` turns it off |
| Async weight prefetch | When expert weights are copied to the GPU for a large batch, the next split's weights are uploaded on a second CUDA stream while the current split computes. Turns itself off if VRAM is short. | On by default; `GGML_SCHED_PREFETCH=0` turns it off, `3`/`4` uses more slots |
| Offload policy | Sets the batch size from which weights in host memory are copied to the GPU (stock: 32), and which ops may be copied. | `--op-offload-min-batch N`, `--op-offload-ops LIST` (llama-bench `-oomb`) |
| Expert reduction | Uses fewer experts per token. Decode is 7-14% faster; the quality cost has not been measured yet. | `--moe-n-expert N`, `-ser KMIN,T` |

Design notes, measurements and open work: [ASYNC_PREFETCH.md](ASYNC_PREFETCH.md).

## Quick start

There are no prebuilt binaries, packages or Docker images for this fork; build it from source. You need CMake, a C++17 compiler and, for NVIDIA GPUs, the CUDA toolkit.

```sh
cd llama-abominated

# NVIDIA GPU (set CMAKE_CUDA_ARCHITECTURES to your GPU; 75 = GTX 1650 / Turing)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON -DBUILD_SHARED_LIBS=ON
cmake --build build -j6

# CPU only
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build -j6
```

The ported kernels need a CPU with AVX2, F16C and FMA (build with `-DGGML_NATIVE=ON` or `-march` that includes them); on other CPUs the stock kernels are used. Other backends build as in upstream: see [docs/build.md](docs/build.md).

The binaries are in `build/bin`:

```sh
# chat in the terminal, experts of 40 layers kept on the CPU
build/bin/llama-cli -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf -ngl 99 -ncmoe 40 -fa 1

# OpenAI-compatible API server with the recommended settings
# (agent use: 64K context, Qwen thinking-mode coding sampling, bounded thinking)
build/bin/llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --jinja \
  -c 65536 -ngl 99 -ncmoe 40 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024 -t 4 \
  --op-offload-min-batch 2048 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1.0 \
  --reasoning-budget 8192 --reasoning-budget-message "Enough thinking, act now." \
  --host 127.0.0.1 --port 8080

# compare the old and new CPU kernels
GGML_CPU_IQK=0 build/bin/llama-bench -m model.gguf -ngl 99 -ncmoe 40 -oomb 2048 -p 512 -n 64
build/bin/llama-bench -m model.gguf -ngl 99 -ncmoe 40 -oomb 2048 -p 512 -n 64
```

`-hf user/repo` downloads a model from Hugging Face, as in upstream.

### Tuning for a small GPU

- Put all layers on the GPU (`-ngl 99`) and move the experts of the first N layers back to the CPU with `-ncmoe N`. Raise N until the model, KV cache and compute buffers fit in VRAM.
- Use a large batch (`-b 2048 -ub 1024`) so prompts are processed in few passes.
- With this fork's CPU kernels, keeping the experts on the CPU is usually faster than copying them to a small GPU for every batch: set `--op-offload-min-batch` above your `-ub`.
- Use 4 threads (`-t`) on a 4-core CPU for decode. Decode speed is limited by RAM bandwidth.
- Keep the default mmap loading. `-lm none` (pinned host buffers) turns off the CPU weight repacking and makes decode about 35% slower.

## Benchmarks

Test machine: i5-10300H (4C/8T, AVX2), 32 GB DDR4-2933, GTX 1650 4 GB on PCIe 3.0.

**Qwen3.6-35B-A3B UD-Q4_K_XL**, tok/s, `-ngl 99 -ncmoe 40 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024 -t 4`:

| Test | Old CPU kernels (`GGML_CPU_IQK=0`, `-oomb 512`) | Ported kernels (`-oomb 2048`) | Speedup | ik_llama.cpp |
|---|---:|---:|---:|---:|
| pp64 | 63.6 | 84.7 | 1.33x | 81.9 |
| pp128 | 71.3 | 90.5 | 1.27x | 97.2 |
| pp256 | 76.4 | 106.8 | 1.40x | 112.3 |
| pp512 | 87.4 | 114.3 | 1.31x | 120.1 |
| pp1024 | 104.0 | 119.6 | 1.15x | 100.7 |
| tg64 | 23.5 | 23.7 | same | 24.2 |

- The old-kernel column uses its best setting: from 512 tokens the experts go to the GPU, which is faster than the old CPU kernels there.
- With the ported kernels, the CPU beats the GPU copy at every batch size up to 1024, so `-oomb 2048` (never copy the experts) is best.
- Perplexity (6 chunks of 512): old kernels 6.156, ported kernels 6.131, ik_llama.cpp 6.159, all within the +/- 0.40 error.

Kernel time per MoE layer (gate + up + down, 4 threads, Qwen3.6 shapes):

| Tokens | Old kernels | Ported kernels | Speedup |
|---:|---:|---:|---:|
| 64 | 23.8 ms | 17.6 ms | 1.35x |
| 128 | 45.8 ms | 27.5 ms | 1.67x |
| 512 | 129.9 ms | 90.9 ms | 1.43x |
| 1024 | 209.4 ms | 148.8 ms | 1.41x |
| 2048 | 360.6 ms | 237.4 ms | 1.52x |

**Qwen3-Coder-30B-A3B UD-Q4_K_XL**, async prefetch only (before the kernel port), tok/s, `-ngl 99 -ncmoe 48 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -t 4`:

| Test | Stock llama.cpp | Fork | Change |
|---|---:|---:|---:|
| pp2048 | 120.3 | 129.6 | +7.7% |
| tg32 | 20.3 | 20.7 | noise |
| pp2048 @ 16k depth | 34.7 | 35.5 | +2.4% |
| tg32 @ 16k depth | 10.4 | 10.8 | noise |

The prefetch hides most of the ~1.3 s PCIe upload per 2048-token batch, so ~7% is close to the maximum on this GPU. It matters only when experts are copied to the GPU, so with the ported kernels and `-oomb 2048` it is not used.


## Supported backends

The fork changes are tested on **CUDA** (NVIDIA) with an **x86 AVX2 CPU**. The other llama.cpp backends are unchanged from upstream and should build and run as before, but they have not been tested here. The async prefetch uses CUDA streams, and the ported kernels are x86-only.

| Backend | Target devices | Tested in this fork |
| --- | --- | --- |
| [CUDA](docs/build.md#cuda) | Nvidia GPU | yes |
| CPU (AVX2) | x86 | yes |
| [Vulkan](docs/build.md#vulkan) | GPU | no |
| [HIP](docs/build.md#hip) | AMD GPU | no |
| [Metal](docs/build.md#metal-build) | Apple Silicon | no |
| [SYCL](docs/backend/SYCL.md) | Intel GPU | no |
| [BLAS](docs/build.md#blas-build) | All | no |

## Documentation

This fork:

- [ASYNC_PREFETCH.md](ASYNC_PREFETCH.md) - design, measurements and open work

Upstream docs (still valid for this fork):

- [cli](tools/cli/README.md), [completion](tools/completion/README.md), [server](tools/server/README.md), [llama-bench](tools/llama-bench/README.md)
- [GBNF grammars](grammars/README.md)
- [How to build](docs/build.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [Models](docs/models.md)

## Status

- Personal fork, kept locally; not published and not intended for upstream as is.
- To pick up new upstream changes, rebase the `async-overlap` branch on llama.cpp `master`.
- Open work (quality of expert reduction, `--threads-batch 8`, more kernels): see section 10 of [ASYNC_PREFETCH.md](ASYNC_PREFETCH.md).

## Acknowledgements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml) by Georgi Gerganov and contributors - MIT license
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) by Iwan Kawrakow - source of the CPU MoE kernels in `ggml/src/ggml-cpu/iqk/` - MIT license
- ATSInfer, "Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices" (Nanjing University, 2026) - the idea behind the async weight prefetch
- Upstream's bundled libraries: [cpp-httplib](https://github.com/yhirose/cpp-httplib), [stb](https://github.com/nothings/stb), [nlohmann/json](https://github.com/nlohmann/json), [miniaudio](https://github.com/mackron/miniaudio), [subprocess.h](https://github.com/sheredom/subprocess.h)

## License

MIT, like llama.cpp. See [LICENSE](LICENSE).
