# Async weight prefetch for hybrid CPU-GPU inference

Notes on adding the "asynchronous CPU-GPU overlap" idea from the ATSInfer paper to llama.cpp, tested on a 4 GB laptop GPU.

- Fork: `~/ws/src/cpp/nn/llama-abominated`, branch `async-overlap` (cloned from `~/ws/src/cpp/nn/llama` at commit `781dbc5`)
- Status: working, tested, not committed
- Date: 2026-10-16

## 1. Background: the paper

`ref_doc/2607.10183v2.pdf` - "Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices" (ATSInfer, Nanjing University, July 2026). It extends llama.cpp (about 15k lines of C++, code not published) with three parts:

| Part | Idea | Paper's result |
|---|---|---|
| Async CPU-GPU coordination | Weight uploads on a second CUDA stream (copy engine), activations on the compute stream, synchronized with events | Main prefill gain |
| Static tensor placement | Knapsack over per-tensor measured CPU-GPU time gap per MiB, with a penalty for each CPU/GPU switch | Small gain in decode |
| Load-aware dynamic transfer | DP each step that picks CPU tensors to send to the GPU while overlapping the transfer | Main decode gain |

Reported results vs llama.cpp: up to 1.94x prefill, 3.29x decode, on RTX 3060 6 GB and RTX 4090 24 GB with PCIe Gen4.

## 2. Our hardware and what it allows

| Part | Value | Effect |
|---|---|---|
| GPU | GTX 1650, 4 GB, PCIe Gen3 x16 (~12 GB/s), no tensor cores | Holds little, slow at prefill matmuls |
| CPU | i5-10300H, 4C/8T, AVX2 only | Slow at matmul |
| RAM | 2x16 GB DDR4-3200 running at 2933 (CPU limit), about 47 GB/s theoretical, ~35 GB/s in practice | Sets the decode speed ceiling |
| RAM slots | Board reports 64 GB max | Upgrade path: 2x32 GB |

Main conclusions:

- Decode reads every active weight once per token, so decode speed is about RAM bandwidth divided by the bytes of weights per token that sit in RAM. No scheduling trick gets past that.
- A 70B dense model does not fit in 31 GB at Q4 (~42 GB), and at 2-bit it runs at ~1 tok/s with a large loss in quality.
- MoE models with few active parameters are the realistic way to get "big and fast". An 80B-A3B at Q4 needs the 64 GB upgrade.
- Faster RAM does not help: the CPU already caps the RAM at 2933 MT/s.
- The paper's decode gains need a much bigger GPU. Here, PCIe uploads also read from the same DDR4 the CPU uses, so dynamic transfer has little room to help.
- The part of the paper that carries over is prefill: copy and compute overlap.

## 3. Config tuning first (biggest win, no code)

`llama-gpu` logs showed the 30B taking 5.3 minutes on a 7k-token claw prompt. The cause: with a batch of 32 or more tokens, llama.cpp copies all CPU-resident expert weights (~16 GB) to the GPU once per ubatch (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32). With `-ub 128` that copy happened every 128 tokens, from pageable (mmap) memory.

Changes now in `~/.config/llama-gpu/models.ini` for `qwen3coder30A3B`: `n-gpu-layers = 99`, `n-cpu-moe = 48`, `batch-size = 2048`, `ubatch-size = 2048`.

Qwen3-Coder-30B-A3B UD-Q4_K_XL, measured through `llama-gpu`:

| | Before | After |
|---|---|---|
| Prefill, claw's 7.3k-token prompt | 22.8 tok/s (5.4 min) | 83 tok/s (88 s) |
| Decode, 400-token reply | 7.8 -> 3.7 tok/s | 17 tok/s |

`llama-bench` prefill at 4k tokens: ub 128 = 27 tok/s, 1024 = 93, 2048 = 105, 4096 = out of VRAM.

## 4. What llama.cpp did before the change

In `ggml/src/ggml-backend.cpp`:

- The scheduler starts a new split for every op whose weight lives in host memory and must be offloaded ("by starting a new split, the memory of the previously offloaded weights can be reused").
- `ggml_backend_sched_compute_splits()` copies each split's inputs, including the host weights, and then runs the split. The copy for split N+1 only starts after split N is queued, and on the same stream, so copy and compute take turns.
- The weight copies are allocated by ggml-alloc in the shared compute buffer, which reuses memory across splits. So the copy for the next split cannot simply be started early: it might overwrite tensors still in use.
- With one GPU and no pipeline parallelism (`n_copies == 1`), `ggml_backend_sched_copy_input()` also blocks the host with `ggml_backend_synchronize()` before each copy.

## 5. The change

Files: `ggml/src/ggml-backend.cpp` (~300 lines), `ggml/include/ggml-backend.h` (+4), `src/llama-context.cpp` (+2).

### Design

1. **Upload stream.** For each GPU backend the scheduler creates a second backend instance on the same device with `ggml_backend_dev_init()`. With CUDA this gives a second, non-blocking stream. Only devices whose `caps.async` and `caps.events` are set are used.
2. **Slot ring.** A dedicated VRAM buffer with N slots (default 2), each sized for the largest prefetched split. The prefetched weight copies are placed in it with `ggml_backend_tensor_alloc()`, so ggml-alloc skips them (it treats tensors that already have a buffer as external).
3. **Planning** (`ggml_backend_sched_prefetch_plan()`, run at the end of graph splitting, before allocation). A host weight input can be prefetched only if:
   - its split runs on a GPU backend that supports async uploads and events;
   - the copy is used only inside the split that uploads it (otherwise the slot could be overwritten before its last use);
   - for `MUL_MAT_ID`, `n_used * n_tokens >= 4 * n_expert`, so almost all experts are used. With few tokens, the existing copy callback, which uploads only the used experts, is cheaper;
   - its split is not a size outlier. Sizes are sorted in descending order and the top ones are dropped while each is more than 2x the next. This keeps the large one-off output matrix (LM head) from setting the slot size.

   Splits of one backend use the slots in turn. Split k + N - 1 is uploaded after split k is queued.
4. **Ordering, with no host blocking:**
   - upload stream: wait on `done[slot]`, upload the weights, record `ready[slot]`
   - compute stream: wait on `ready[slot]`, run the split, record `done[slot]`, then queue the next upload

   The order in which these are issued guarantees that each wait sees the matching record. The events carry over between graphs, so the first uploads of the next graph wait for the last users of each slot.
5. **Memory accounting:**
   - `ggml_backend_sched_get_buffer_size()` includes the slot buffer.
   - `ggml_backend_sched_reserve_size()`, used by `--fit`, adds the extra `(N - 1) * slot` bytes. ggml-alloc already counts one split's copies.
6. **Fallbacks.** Prefetch is not used with pipeline parallelism (`n_copies > 1`) or with the MoE cache (`--moe-cache-mib`), which relies on the copy callback; llama-context turns it off. If the slot buffer cannot be allocated, a warning is printed and the normal path is used.

### Controls

| Control | Effect |
|---|---|
| `GGML_SCHED_PREFETCH=0` (or `1`) | Prefetch off |
| `GGML_SCHED_PREFETCH=2` | Default, 2 slots |
| `GGML_SCHED_PREFETCH=3` / `4` | More slots (max 4) |
| `ggml_backend_sched_set_prefetch(sched, false)` | Turn it off from code |

Run with `-v` to see `weight prefetch buffer size = ... MiB (N slots)` in the log.

Keep the default mmap loading. Measurements (section 7) show that prefetch also works with mmap (+7.7% prefill), while `-lm none` (pinned `CUDA_Host` weights) switches off the AVX2-repacked CPU weight layouts and cuts decode from ~20 to ~13 tok/s.

## 6. Bugs found during testing (fixed)

1. **Crash on exit:** `ggml_backend_sched_free()` synchronized the split backends, which `llama_context` has already freed at that point. It now synchronizes only its own upload backends.
2. **Prefetch never active:** the first outlier rule used 2x the median split size. With many tiny norm-weight splits the median was tiny, so every real matrix was excluded (the slot buffer was 0.25 MiB). It was replaced with the descending-gap rule above.
3. **Out-of-VRAM fallback** checked: it warns, turns prefetch off and continues.

## 7. Results

### Correctness

Qwen3.5-4B Q4_K_M, `-ngl 0` (all weights streamed), `-c 512 -ub 512`, 8 chunks:

| GGML_SCHED_PREFETCH | Slot buffer | PPL | s/pass |
|---|---|---|---|
| 0 | - | 6.6812 +/- 0.38957 | 3.93 |
| 2 | 36.9 MiB | 6.6812 +/- 0.38957 | 3.80 |
| 3 | - | 6.6812 +/- 0.38957 | - |

Every chunk value is identical. The 4B gains little because its largest upload, the 497 MB LM head, is excluded as an outlier.

### Speed, Qwen3-Coder-30B-A3B UD-Q4_K_XL

Settings: `-ngl 99 -ncmoe 48 -fa 1 -ctk q8_0 -ctv q8_0 -t 4 -b 2048 -lm none`, 2 runs each.

| Test | Prefetch off | Prefetch on | Change |
|---|---|---|---|
| pp4096, ub 1024 | 86.33 +/- 0.27 | 93.54 +/- 0.23 | +8.4% |
| pp4096, ub 2048 | 105.22 +/- 0.37 | 111.50 +/- 0.24 | +6.0% |
| tg128 | 18.39 - 18.79 | 18.38 - 18.87 | same (the gap flips with run order, so it is heat) |

### Stock build vs fork, tuned settings (full CUDA build)

Baseline as suggested: stock `llama/build` vs `llama-abominated/build`, same command, `-r 1`:

```bash
llama-bench -m models/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
  -ngl 99 -ncmoe 48 -fa 1 -ctk q8_0 -ctv q8_0 -t 4 -b 2048 -ub 2048 \
  -p 2048 -n 32 -d 0,16384 -r 1 -lm mmap|none
```

| Test (tok/s) | Stock mmap | Fork mmap | Change | Stock `-lm none` | Fork `-lm none` | Change |
|---|---|---|---|---|---|---|
| pp2048 | 120.29 | 129.56 | +7.7% | 121.34 | 127.19 | +4.8% |
| tg32 | 20.26 | 20.68 | noise | 13.29 | 14.06 | noise |
| pp2048 @ d16384 | 34.67 | 35.50 | +2.4% | 34.68 | 35.56 | +2.5% |
| tg32 @ d16384 | 10.44 | 10.76 | noise | 9.37 | 9.69 | noise |

- Prefill: +5-8% at short context, +2.5% at 16k depth, where attention over the long context dominates.
- Decode: unchanged, as expected (no weight streaming for batches under 32 tokens). Single runs, so differences under ~3% are noise.
- `-lm none` makes decode much slower (13.3 vs 20.3 tok/s): pinned host buffers switch off CPU weight repacking. Keep mmap.

### Why the gain is small here

At ub 2048 one 2048-token batch takes ~19.5 s. Uploading the ~16 GB of experts over PCIe Gen3 takes ~1.3 s of that. Prefetch hides that upload almost completely, so ~6-7% is about the maximum, and that matches what we measured. Prefill on this machine is now limited by GTX 1650 compute, not PCIe. The overlap would matter more with a faster GPU, smaller ubatches, or a slower link.

## 8. How it was built and tested

- Quick tests used a library overlay: the new `libggml-base` and `libllama` (built with `-DGGML_CUDA=OFF` in `build-overlay/`), plus the existing `libggml`, `libggml-cpu` and `libggml-cuda`, all linked from `llama-abominated/overlay/` and loaded through `LD_LIBRARY_PATH`. This works because it is the same commit and ABI, and only ggml-base and llama changed.
- Full CUDA build, with the same options as the main tree:

  ```bash
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_GRAPHS=ON -DBUILD_SHARED_LIBS=ON
  cmake --build build -j6
  ```

  Built in ~17.5 min (582 targets, no new warnings; the only warnings are existing `u8path` deprecations in `common/`).

- A/B benchmark:

  ```bash
  for P in 0 2; do
    GGML_SCHED_PREFETCH=$P build/bin/llama-bench \
      -m ../llama/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
      -ngl 99 -ncmoe 48 -fa 1 -ctk q8_0 -ctv q8_0 -t 4 \
      -p 4096 -n 0 -r 2 -b 2048 -ub 1024,2048 -lm none
  done
  ```

## 9. Not done yet / next steps

- [x] Full CUDA build, `llama-bench` from `llama-abominated/build` compared against the stock build
- [ ] Run `llama-gpu` from the fork for real claw use (keep mmap; do not set `load-mode = none`)
- [ ] Untested: multi-GPU, MoE cache, long sessions, `test-backend-ops`
- [ ] Possible: let llama-server choose the slot count, and try 3 slots at ub 1024
- [ ] Not worth it on this hardware: the paper's per-tensor knapsack (`--fit` / `--n-cpu-moe` already cover it) and dynamic decode transfer (limited by RAM bandwidth)
- Hardware path to "70B+ and fast": upgrade to 2x32 GB RAM, then run an 80B-A3B MoE at Q4

## 10. Status of the follow-up work (2026-10-18)

Done and measured on Qwen3.6-35B-A3B (`-ngl 99 -ncmoe 40 -ub 1024`, q8 KV, 4 threads):

- [x] OOM fix for the prefetch ring: the ring is allocated during reserve, after the compute buffers, and only if at least `GGML_SCHED_PREFETCH_MARGIN` MiB (default 320) stay free; otherwise prefetch turns itself off. The retry restores the node sources that `split_graph` rewrote. ub 1024: prefetch off, no crash; ub 768: prefetch on, +20% prefill over stock.
- [x] Offload policy `--op-offload-min-batch N` / `--op-offload-ops LIST` (llama-bench `-oomb`). With the ported iqk kernels (section 11), keeping the experts on the CPU is faster at every batch size up to ub 1024, so `-oomb 2048` (never offload the experts) is best.
- [x] Expert reduction `--moe-n-expert N`, `-ser KMIN,T`: decode +7% (`-moen 7`, `-ser 5,0.4`) and +14% (`-moen 6`). Quality not measured yet.
- [x] All options off: perplexity matches stock exactly.
- [x] ik_llama.cpp test (it loads Qwen3.6): ik was 1.3-1.4x faster than the fork on 64-512 token prompts with the experts on the CPU, all from its CPU kernels (its fused MoE op made no difference). Led to the port in section 11.

Still open:

- [ ] Perplexity for `-moen 6,7` and `-ser 5,0.4`.
- [ ] `--threads-batch 8` for prefill (ik went from 93 to 102 tok/s at 128 tokens with 8 threads; decode stays at 4).
- [ ] q6_K -> q8_0_r8 path for experts with >= 64 rows (only 4 tensors in this model).
- [ ] `ngram` speculative decoding on code-editing prompts.
- [ ] Optional: 80B-A3B at Q4 with mmap (RAM/NVMe tiering), needs a ~45 GB download.
- [ ] Optional: `--moe-cache-mib` with prefetch.

## 11. Port of ik_llama.cpp's CPU MoE kernels

Files: `ggml/src/ggml-cpu/iqk/iqk-kquants.{h,cpp}` (MIT, Iwan Kawrakow; about 600 lines, AVX2 only), hooked into `ggml_compute_forward_mul_mat_id` in `ggml-cpu.c` ahead of the tiled path.

What it does, for Q4_K/Q5_K/Q6_K expert tensors in a plain CPU buffer:

- Activations are quantized to ik's `block_q8_2_x4` (bf16 scale + int16 quant sum per 32 values) instead of Q8_K. Rows are smaller than Q8_K rows, so the existing wdata reservation covers them.
- Experts with fewer than 32 rows: kernels that unpack each weight block once and reuse it for up to 8 token columns (stock does one `vec_dot` per column, so it unpacks again for every token).
- Q4_K/Q5_K experts with 32 or more rows: the weights are first converted (losslessly) to an 8-row interleaved 8-bit layout (`q8_1_r8`) in a thread-local buffer, then a GEMM kernel runs on that.
- Rows of each expert are split evenly over the threads.

Switches: `GGML_CPU_IQK=0` turns it off. `GGML_CPU_IQK_REPACK=1` keeps Q4_K experts in the repack buffer (by default they are kept out of it so iqk gets them; this only matters without `-ncmoe`, because `-ncmoe` already puts the experts in a plain CPU buffer).

Kernel time per MoE layer (gate + up + down, 4 threads, Qwen3.6 shapes, uniform routing):

| tokens | stock | iqk | speedup |
|---:|---:|---:|---:|
| 64 | 23.8 ms | 17.6 ms | 1.35x |
| 128 | 45.8 ms | 27.5 ms | 1.67x |
| 512 | 129.9 ms | 90.9 ms | 1.43x |
| 1024 | 209.4 ms | 148.8 ms | 1.41x |
| 2048 | 360.6 ms | 237.4 ms | 1.52x |

The outputs differ from stock by 0.8% relative RMS (both quantize the activations to 8 bits, with different block sizes).

Full model, tok/s (`-ngl 99 -ncmoe 40 -ub 1024 -b 2048`, q8 KV, `-t 4`):

| test | stock CPU kernels, `-oomb 512` | iqk, `-oomb 512` | iqk, `-oomb 2048` | ik_llama.cpp |
|---|---:|---:|---:|---:|
| pp64 | 63.6 | 87.7 | 84.7 | 81.9 |
| pp128 | 71.3 | 92.6 | 90.5 | 97.2 |
| pp256 | 76.4 | 111.1 | 106.8 | 112.3 |
| pp512 | 87.4 | 87.5 (GPU) | 114.3 | 120.1 |
| pp1024 | 104.0 | 104.0 (GPU) | 119.6 | 100.7 |
| tg64 | 23.5 | 23.8 | 23.7 | 24.2 |

Perplexity (6 chunks of 512, experts on the CPU): stock 6.156, iqk 6.131, ik 6.159 (all within the +/- 0.40 error).

Recommended preset for Qwen3.6-35B-A3B on this laptop: the fork's `llama-server` with `op-offload-min-batch = 2048` (with ub 1024 the experts then never go to the GPU, so the PCIe copies and the prefetch ring are not used).
