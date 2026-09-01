# NInfer QUASAR Post-Merge Test Results — Ubuntu 26.04, RTX 5090 Laptop GPU

> **Machine:** Ubuntu 26.04.1 LTS (x86_64, kernel 7.2.0-070200rc5-generic), NVIDIA GeForce RTX
> 5090 Laptop GPU (24,463 MiB VRAM), driver 595.84, CUDA Toolkit 13.1, and 64 GiB system RAM.
>
> **Runtime revision:** merge commit `c7f85aa231955848e7fcc532935c131a75bad246` on
> `feat/quasar-nvfp4-converter-local-changes`. This combines the working QUASAR branch at
> `5a209251` with Neroued upstream `21a0e85f`.
>
> **Artifact:** `qwen3_8_27b_nvfp4.ninfer`, 17,555,331,072 bytes,
> SHA-256 `931816373707010b03e6e4dcba10f5265c3e820584dacd0eb2c6039e397045cd` — **checksum verified**.
>
> **Baseline:** `docs/ninfer-quasar-test-results.md`, measured at pre-merge revision `2d11c992`.
> **Test date:** 2026-09-01.

---

## 1. Environment and build

| Check | Result |
|---|---|
| OS / architecture | Ubuntu 26.04.1 LTS, x86_64 |
| GPU | RTX 5090 Laptop GPU, 24,463 MiB, driver 595.84 |
| Competing compute processes | none before validation |
| CUDA | 13.1 |
| CMake / Ninja / g++ / pkg-config | 4.2.3 / 1.13.2 / 15.2.0 / 2.5.1 |
| FFmpeg libraries | avformat 62.3.100, avcodec 62.11.100, avutil 60.8.100, swscale 9.1.100 |
| libcurl | 8.18.0 |
| Product build | ✅ `ninfer`, `ninfer-serve`, and `ninfer-perplexity` |
| Full test-target build | ✅ all configured test executables compiled and linked |
| Unresolved `ninfer-serve` libraries | ✅ none |

Configure and build used the Release Ninja tree with `BUILD_TESTING=ON`; the repository enforces
CUDA architecture `120a`.

## 2. Focused and artifact-dependent tests

| Test | Result | Time / detail |
|---|---|---|
| Full product build | ✅ | 293 build steps after integration |
| Full test-target build | ✅ | all configured test targets linked |
| `ninfer_kv_cache_test` | ✅ | 3.36 s |
| `ninfer_softmax_attention_test` | ✅ | 75.48 s |
| `ninfer_softmax_attention_nvfp4_test` | ✅ | 22.50 s |
| `ninfer_softmax_attention_k8v4_test` | ✅ | 38.33 s |
| `ninfer_linear_swiglu_nvfp4_test` | ✅ | 1.58 s |
| QUASAR Python converter test | ✅ | 3 tests passed in 0.74 s |
| QUASAR real-artifact load plan | ✅ | 3.40 s |
| BF16/INT8 ordinary + MTP greedy parity | ✅ | 416.21 s |

The greedy-parity test loaded the real QUASAR artifact and covered:

- BF16 and INT8 group-64 KV;
- ordinary decoding and MTP draft counts 1 through 5;
- concurrency frontiers 1 through 8;
- a second maximum-concurrency pass to exercise slot reuse;
- exact equality of all 128 generated token IDs against ordinary decode.

The causal-scoring real test was not run with this artifact because it uses the ordinary
Qwen3.6-27B environment-variable contract rather than the QUASAR artifact route. The relevant
QUASAR merge gates are the load-plan and exact MTP parity tests above. No source checkpoint was
available for a full conversion; the converter's inventory, dispatch, and exact control-decode
tests passed against CPU PyTorch.

## 3. CLI smoke test, 4K context

```bash
build/apps/ninfer "$MODEL" \
  --prompt "Explain quantization-aware training in exactly three concise sentences." \
  --max-context 4096 --max-new 256 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

**Result: PASS** — exit code 0 and a correct answer of exactly three sentences.

| Metric | Post-merge | Pre-merge baseline |
|---|---:|---:|
| Generated tokens | 232 | 232 |
| Model elapsed | 2.652 s | 2.579 s |
| Prefill speed | 989.17 tok/s | 1,008 tok/s |
| Decode speed | 89.27 tok/s | 91.9 tok/s |
| Overall throughput | 87.47 tok/s | 90.0 tok/s |
| GPU weights | 16.06 GiB | 16.06 GiB |
| Runtime reservation | 472.50 MiB | 472.50 MiB |
| Free after weights | 7.09 GiB | 7.09 GiB |
| MTP drafted / accepted | 273 / 142 | 273 / 142 |
| MTP acceptance rate | 52.01% | 52.0% |
| MTP accepted length | 2.56 tok/round | 2.56 tok/round |
| MTP accepted by position | 61 / 47 / 34 | 61 / 47 / 34 |

The exact MTP acceptance statistics and generated-token count match the baseline. Throughput in this
single rerun was approximately 3% below the baseline; this is a smoke comparison, not a controlled
benchmark campaign.

## 4. HTTP API, 96K context

The documented 96K MTP/INT8 profile started successfully.

| Check | Result |
|---|---|
| Engine ready | ✅ 5.91 s target load |
| `GET /health` | ✅ `{"status":"ok"}` |
| `GET /v1/models` | ✅ model `qwen3.8-27b-quasar-nvfp4`, `max_model_len: 98304` |
| `POST /v1/chat/completions` | ✅ `finish_reason: stop`, content `The API works.` |
| Usage | ✅ 62 prompt + 44 completion = 106 total; 38 reasoning tokens |
| KV capacity | 98,304 tokens / 1,536 page groups |
| Runtime reservation | 3.82 GiB |
| Free after weights / startup | 7.09 GiB / 3.33 GiB |
| Planned slack | 3.26 GiB |

The server's new structured logging format reports the same capacity and memory picture as the
pre-merge result.

## 5. Staged context validation

Each passing stage launched `ninfer-serve`, polled `/health`, sent a non-streaming OpenAI chat
completion, verified `finish_reason: stop`, and stopped the server. The harness was
`.codex/ninfer-stage-test.sh`.

| # | Profile | Result | Runtime reservation | Free after startup | Planned slack |
|---|---|:---:|---:|---:|---:|
| 1 | 4,096 explicit, MTP, INT8 | ✅ | 0.60 GiB | 6.48 GiB | 6.48 GiB |
| 2 | 32,768 explicit, MTP, INT8 | ✅ | 1.63 GiB | 5.52 GiB | 5.46 GiB |
| 3 | 98,304 explicit, MTP, INT8 | ✅ | 3.82 GiB | 3.33 GiB | 3.26 GiB |
| 4 | 131,072 explicit, MTP, INT8 | ✅ | 4.92 GiB | 2.24 GiB | 2.17 GiB |
| 5 | 131,072 explicit, no MTP, INT8 | ✅ | 4.58 GiB | 3.26 GiB | 3.26 GiB |
| 6 | 262,144 auto, MTP, INT8 | expected init failure | — | — | — |
| 7 | 262,144 auto, no MTP, INT8 | expected init failure | — | — | — |
| 8 | 131,072 auto, no MTP, INT8 | ✅ resolves 131,072 | 4.58 GiB | 3.26 GiB | 3.26 GiB |
| 9 | 131,072 explicit, MTP, FP8 | ✅ | 4.82 GiB | 2.34 GiB | 2.27 GiB |

All passing stages returned coherent completion content. The harness's legacy `grep 'KV capacity'`
line is blank under upstream's structured logging; the authoritative values above were extracted
from the `engine capacity` records in each server log.

### Expected 262K failures

MTP:

```text
minimum Engine runtime reservation requires 9987798272 bytes in addition to
1073741824 bytes of automatic headroom, but only 7610317824 bytes are available after weights
```

No MTP:

```text
minimum Engine runtime reservation requires 9351343872 bytes in addition to
1073741824 bytes of automatic headroom, but only 8418625536 bytes are available after weights
```

These exactly match the pre-merge capacity boundary. They are expected product admission failures,
not regressions.

## 6. Post-merge conclusions

1. **QUASAR support survived the upstream merge.** Artifact identification, load planning, weight
   materialization, CLI generation, HTTP serving, and staged context startup all pass.
2. **Mirko's parity guarantees survived the KV-cache rework.** Exact BF16/INT8 greedy output passed
   across ordinary decode, MTP k1–k5, concurrency 1–8, and maximum-width slot reuse.
3. **Upstream NVFP4/K8V4 attention paths are qualified on this GPU.** Both dedicated low-bit
   attention suites pass after integrating canonical-column handling with FP16-V/FP32-PV changes.
4. **The usable memory boundary is unchanged.** 128K MTP/INT8 remains valid with 2.17 GiB planned
   slack; 262K still fails the minimum runtime-reservation check before automatic KV sizing.
5. **The baseline recommendation remains valid:** 131,072 MTP/INT8 for the tested machine, or
   98,304 as the conservative profile.
6. **Performance is not claimed from this run.** The one CLI smoke point was about 3% slower than
   the baseline, while all output and MTP statistics matched. A repeated warm benchmark campaign is
   required for a performance conclusion.

## 7. Reproduction

```bash
MODEL="$HOME/models/ninfer/qwen3.8-27b-quasar/qwen3_8_27b_nvfp4.ninfer"

NINFER_QWEN3_8_27B_QUASAR_NVFP4_WEIGHTS="$MODEL" \
  ctest --test-dir build -R '^ninfer_qwen3_6_27b_load_plan_test$' --output-on-failure

NINFER_MTP_GREEDY_PARITY_WEIGHTS="$MODEL" \
  ctest --test-dir build \
  -R '^ninfer_qwen3_6_27b_mtp_greedy_parity_real_test$' --output-on-failure

bash .codex/ninfer-stage-test.sh 131072 131072 mtp int8
```

Raw runtime logs for this run were captured under `/tmp/ninfer-c7f85aa2-*.log` and
`/tmp/ninfer-stage-*.log`; `/tmp` logs are transient and are not committed evidence.
