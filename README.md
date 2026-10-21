# llama-abominated

<div align="center">

<b>llama.cpp tuned for running big MoE models on a small GPU</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (based on upstream commit `781dbc5`, plus the later upstream fixes listed below) for hybrid CPU-GPU inference of mixture-of-experts models when most of the model has to stay in system RAM. It was built and tested on a laptop with a 4 GB GTX 1650 and an AVX2 CPU, running Qwen3.6-35B-A3B.

Everything that works in llama.cpp at that commit works here: the same GGUF files, tools, server API and backends. With `GGML_CPU_IQK=0`, `GGML_SCHED_PREFETCH=0` and no `-ser`/`--moe-n-expert`, the output matches stock llama.cpp exactly (the faster CPU kernels are on by default).

## What this fork adds

| Feature | What it does | Control |
|---|---|---|
| Faster CPU MoE kernels | AVX2 Q4_K/Q5_K/Q6_K `MUL_MAT_ID` kernels ported from [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp). Each weight block is unpacked once and reused for up to 8 tokens; experts with 32 or more tokens are converted to an interleaved 8-bit layout (near-lossless: the block scales are rounded to fp16; measured end to end it is slightly more accurate than stock) and run through a GEMM kernel. 1.15-1.4x faster prompt processing with the experts on the CPU. | On by default; `GGML_CPU_IQK=0` turns it off |
| Faster CPU dense kernels | The same kernels for plain `MUL_MAT` (attention, shared experts, dense models on the CPU), with ik_llama.cpp's work split between threads. They run before upstream's tiled path: 2.1-2.4x faster at 8 rows, 1.26x at 512 rows for Q4_K. Q6_K with 128 or more rows keeps the tiled path, which is faster there. | Same switch; `GGML_CPU_IQK_REPACK=1` keeps dense Q4_K in upstream's repack buffer |
| Upstream fixes | Cherry-picked from llama.cpp after `781dbc5`: `f174553` server checkpoint invalidation (stale state on hybrid/recurrent models such as Qwen3.6 when the client rewrites history), `99db719` CUDA flash-attention decode race, `5aeac37` flash-attention tile loop bound, `839a572` Qwen3.5/3.6 linear-attention output reshape. | Always on |
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

# OpenAI-compatible API server with the recommended settings for a 4 GB GPU
# (the -gpuQ6_K file is made below, under "Shrink the GPU-side weights")
build/bin/llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL-gpuQ6_K.gguf --jinja -c 65536 \
  -ngl 99 -ncmoe 40 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -t 4 \
  --op-offload-min-batch 4096 --host 127.0.0.1 --port 8080

# compare the old and new CPU kernels
GGML_CPU_IQK=0 build/bin/llama-bench -m model.gguf -ngl 99 -ncmoe 40 -oomb 2048 -p 512 -n 64
build/bin/llama-bench -m model.gguf -ngl 99 -ncmoe 40 -oomb 2048 -p 512 -n 64

# accuracy of the CPU kernels against an f64 reference (also run by ctest)
build/bin/test-iqk-mul-mat 4 && GGML_CPU_IQK=0 build/bin/test-iqk-mul-mat 4
```

`-hf user/repo` downloads a model from Hugging Face, as in upstream.

### Tuning for a small GPU

- Put all layers on the GPU (`-ngl 99`) and move the experts of the first N layers back to the CPU with `-ncmoe N`. Raise N until the model, KV cache and compute buffers fit in VRAM.
- Use the largest batch that fits (`-b 2048 -ub 2048` with the smaller GPU-side weights below, else `-ub 1024`) so prompts are processed in few passes.
- With this fork's CPU kernels, keeping the experts on the CPU is faster than copying them to a small GPU for every batch: set `--op-offload-min-batch` (or `GGML_OP_OFFLOAD_MIN_BATCH`) above your `-ub`. An older threshold of 640, measured before the kernel port, made 2048-token prompts 17% slower (95.8 vs 112.3 tok/s).
- Use 4 threads (`-t`) on a 4-core CPU for decode. Decode speed is limited by RAM bandwidth.
- Keep the default mmap loading. `-lm none` (pinned host buffers) turns off the CPU weight repacking and makes decode about 35% slower.
- Leave the GPU's memory to the model: a desktop app drawing on the discrete GPU took 374 MiB here. On a laptop, let other programs use the integrated GPU.
- Measure `-ub` above 1024 with `llama-server`, not `llama-bench`: llama-bench reserves logits for every token of the batch (2048 x 248k vocab = ~1.9 GB here) and runs out of VRAM, while the server reserves them for one token per sequence.

### Shrink the GPU-side weights

Unsloth's UD-Q4_K_XL keeps every tensor that lives on the GPU (attention, DeltaNet projections, shared experts, output head) at Q8_0: 2039 MiB of VRAM for Qwen3.6-35B-A3B. Requantizing those to Q6_K, and copying the experts unchanged, brings it to 1602 MiB, which is what makes room for `-ub 2048` and a 64k context on 4 GB. `llama-quantize` copies a tensor whose target type is its current type, so give every tensor its type in a `--tensor-type-file` (one anchored pattern per tensor, `^blk\.0\.ffn_up_exps\.weight$=q4_k`), with the Q8_0 matrices set to `q6_k` except the router (`ffn_gate_inp`), `attn_k`, `attn_v`, `ssm_alpha`, `ssm_beta` and `token_embd`, which stay `q8_0`. List the F32 tensors too (`=f32`): without them the default type turns some of them (here `ssm_alpha`/`ssm_beta` in 30 layers) into Q4_K.

```sh
build/bin/llama-quantize --allow-requantize --tensor-type-file types.txt \
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf Qwen3.6-35B-A3B-UD-Q4_K_XL-gpuQ6_K.gguf Q4_K_M 4
```

## Benchmarks

Test machine: i5-10300H (4C/8T, AVX2), 32 GB DDR4-2933, GTX 1650 4 GB on PCIe 3.0.

**Qwen3.6-35B-A3B in `llama-server`** (2026-10-21), one request per prompt with `cache_prompt: false`, 64 tokens generated, `-ngl 99 -ncmoe 40 -fa on -ctk q8_0 -ctv q8_0 -t 4`, nothing else running. Prompt and decode in tok/s, VRAM as reported by `nvidia-smi` after load:

| Setup | 1.7k-token prompt | 6.8k | 13.6k | Decode | VRAM |
|---|---:|---:|---:|---:|---:|
| Before: UD-Q4_K_XL, `-ub 1024`, offload from 640 tokens, 32k context, build of 2026-10-18 | 74.0 | 95.8 | 90.8 | 18.8-20.7 | 3226 MiB |
| gpuQ6_K file, `-ub 1024`, no expert offload, 32k | 81.6 | 99.1 | 91.3 | 19.8-22.0 | 2392 MiB |
| gpuQ6_K, `-ub 2048`, 32k | 111.7 | 104.6 | 96.8 | 20.3-22.2 | 2544 MiB |
| **gpuQ6_K, `-ub 2048`, 64k context (recommended)** | **115.3** | **107.1** | **97.8** | **21.2-23.8** | 3172 MiB |
| gpuQ6_K, `-ub 4096`, 32k | 117.2 | 109.9 | 99.2 | 20.0-23.2 | 3200 MiB |

- `-ub 2048` gains most on prompts that fit in one batch (+51% at 1.7k tokens); on long prompts attention over the growing context takes a larger share.
- `-ub 4096` adds 2-5% for 656 MiB more VRAM; on 4 GB, a 64k context is the better use of that memory.
- The prompt speed falls with depth because only 10 of the 40 layers use full attention, and those grow with the context.

**Dense `MUL_MAT`, CPU** (`test-backend-ops perf`, m=4096, k=14336, 4 threads, GFLOPS). These ran while a llama-server used the other cores, so read them as ratios:

| Rows (n) | Q4_K iqk | Q4_K tiled (upstream) | Q6_K iqk | Q6_K tiled |
|---:|---:|---:|---:|---:|
| 8 | **231** | 110 | **185** | 78 |
| 512 | **359** | 286 | 220 | **269** |

Below 8 rows the tiled path does not run; the iqk kernels measured 82, 167 and 219 GFLOPS at 1, 2 and 4 rows (upstream's plain dot products at those sizes were not measured). Accuracy, `test-iqk-mul-mat`: normalized squared error ~3e-5 with the iqk kernels and ~5e-5 with stock, on every `MUL_MAT` and `MUL_MAT_ID` case.

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
- [tests/test-iqk-mul-mat.cpp](tests/test-iqk-mul-mat.cpp) - accuracy test of the CPU kernels (`ctest -R iqk` runs it with and without them)

Upstream docs (still valid for this fork):

- [cli](tools/cli/README.md), [completion](tools/completion/README.md), [server](tools/server/README.md), [llama-bench](tools/llama-bench/README.md)
- [GBNF grammars](grammars/README.md)
- [How to build](docs/build.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [Models](docs/models.md)

## Status

- Personal fork, kept locally; not published and not intended for upstream as is.
- To pick up new upstream changes, rebase the fork's commits on llama.cpp `master` (the fixes above were cherry-picked instead; expect conflicts mainly in `ggml-backend.cpp`).
- Open work (quality of expert reduction, `--threads-batch 8`, more kernels): see section 10 of [ASYNC_PREFETCH.md](ASYNC_PREFETCH.md).

## Acknowledgements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml) by Georgi Gerganov and contributors - MIT license
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) by Iwan Kawrakow - source of the CPU MoE kernels in `ggml/src/ggml-cpu/iqk/` - MIT license
- ATSInfer, "Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices" (Nanjing University, 2026) - the idea behind the async weight prefetch
- Upstream's bundled libraries: [cpp-httplib](https://github.com/yhirose/cpp-httplib), [stb](https://github.com/nothings/stb), [nlohmann/json](https://github.com/nlohmann/json), [miniaudio](https://github.com/mackron/miniaudio), [subprocess.h](https://github.com/sheredom/subprocess.h)

## License

MIT, like llama.cpp. See [LICENSE](LICENSE).
