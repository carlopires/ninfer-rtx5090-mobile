# NInfer QUASAR Setup Test Results — Ubuntu 26.04, RTX 5090 Laptop GPU

> **Machine:** Ubuntu 26.04.1 LTS (x86_64, kernel 7.2.0-070200rc5-generic), NVIDIA GeForce RTX
> 5090 Laptop GPU (24,463 MiB VRAM), driver 595.84, CUDA Toolkit 13.1 (`/usr/local/cuda-13.1`),
> 64 GiB system RAM, 2.8 TiB free disk.
>
> **Runtime revision:** commit `2d11c9927be0ed8be20566814ec4bf97db2fab4f` on
> `feat/quasar-nvfp4-converter` (the exact revision pinned by the QUASAR artifact).
>
> **Artifact:** `qwen3_8_27b_nvfp4.ninfer`, 17,555,331,072 bytes,
> SHA-256 `931816373707010b03e6e4dcba10f5265c3e820584dacd0eb2c6039e397045cd` — **checksum verified**.
>
> **Tutorial followed:** `docs/ninfer-ubuntu-26.04-rtx5090-laptop-tutorial.md` (snapshot 2026-08-30).
> **Test date:** 2026-08-31.
>
> **Local deviation:** the repository was built at `~/code/ninfer` instead of the tutorial's
> `~/src/ninfer-quasar`; the model lives at `~/models/ninfer/qwen3.8-27b-quasar/`.

---

## 1. Environment verification (tutorial §4–§5)

| Check | Requirement | Result | Status |
|---|---|---|---|
| OS / architecture | 64-bit Linux | Ubuntu 26.04.1 LTS, x86_64 | ✅ |
| GPU | RTX 5090 Laptop, ~24 GiB | RTX 5090 Laptop GPU, 24,463 MiB, driver 595.84 | ✅ |
| VRAM contention | no competing compute processes | none at start | ✅ |
| nvcc | CUDA ≥ 13.1 | 13.1.115 (`/usr/local/cuda-13.1/bin/nvcc`) | ✅ |
| CMake | ≥ 3.28 | 4.2.3 | ✅ |
| Ninja / g++ / pkg-config | present | 1.13.2 / 15.2.0 / 2.5.1 | ✅ |
| libavformat / libavcodec | ≥ 60 | 62.3.100 / 62.11.100 | ✅ (installed during setup) |
| libavutil / libswscale | ≥ 58 / ≥ 7 | 60.8.100 / 9.1.100 | ✅ (installed during setup) |
| libcurl | ≥ 7.85 | 8.18.0 | ✅ |
| `uvx` (HF CLI runner) | present | available | ✅ |

Notes:

- The FFmpeg development packages were **not** installed initially; they were installed via
  `sudo apt install libavformat-dev libavcodec-dev libavutil-dev libswscale-dev` (FFmpeg 8.0.1).
- Ubuntu 26.04's FFmpeg 8.0 provides library versions (62/60/60/9) well above the CMake minimums.
- `apt` reported a pre-existing, unrelated `zfs-dkms` initramfs error during the install; it did
  not affect the FFmpeg packages or the build.

## 2. Build results (tutorial §8)

Configure:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120a
```

| Check | Result |
|---|---|
| Configure | ✅ all FFmpeg/libcurl modules found by pkg-config |
| Build (452 targets, `--parallel $(nproc)`) | ✅ no errors |
| `build/apps/ninfer` executable, `--help` runs | ✅ |
| `build/apps/ninfer-serve` executable, `--help` runs | ✅ |
| `ldd build/apps/ninfer-serve` unresolved libraries | ✅ none ("All … resolved") |

CMake auto-detected the runtime cost profile
`nvidia-geforce-rtx-5090-laptop-gpu-sm120/qwen3.8-27b/nvfp4`, confirming the sm_120a target.

## 3. Artifact download and verification (tutorial §9)

```bash
uvx hf download MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer --local-dir "$HOME/models/ninfer/qwen3.8-27b-quasar"
```

| Check | Result |
|---|---|
| Download | ✅ public repo, no authentication needed |
| Size | 17 GiB on disk (17,555,331,072 bytes) |
| SHA-256 | ✅ `qwen3_8_27b_nvfp4.ninfer: OK` (matches the pinned checksum) |

## 4. Smoke test, 4K context (tutorial §10)

```bash
build/apps/ninfer "$MODEL" \
  --prompt "Explain quantization-aware training in exactly three concise sentences." \
  --max-context 4096 --max-new 256 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

**Result: PASS** (exit code 0; correct, coherent three-sentence answer on stdout).

| Metric | Value |
|---|---|
| Generated tokens | 232 |
| Model elapsed | 2.579 s |
| Prefill speed | 1,008 tok/s |
| Decode speed | 91.9 tok/s |
| Throughput (overall) | 90.0 tok/s |
| GPU weights | 16.06 GiB / 16.06 GiB (matches the publisher's MTP figure exactly) |
| KV dtype / payload | int8-group64 / 140.38 MiB |
| Runtime reservation | 472.50 MiB |
| Free after weights / startup | 7.09 GiB / 6.63 GiB |
| MTP rounds / fallback steps | 91 / 0 |
| MTP drafted / accepted tokens | 273 / 142 |
| MTP acceptance rate / length | 52.0% / 2.56 tok/round |
| MTP acceptance by position | 61 / 47 / 34 |

No CUDA OOM, invalid-device-function, or artifact-profile errors.

## 5. Server startup and HTTP API (tutorial §11–§12)

```bash
build/apps/ninfer-serve "$MODEL" \
  --host 127.0.0.1 --port 18080 \
  --model-id qwen3.8-27b-quasar-nvfp4 \
  --max-concurrency 1 \
  --max-context 98304 --kv-capacity 98304 \
  --kv-dtype int8 \
  --device-state-slots 1 --host-state-slots 2 --host-kv-mib 2048 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --default-max-tokens 8192
```

**Result: PASS** — model loaded in 11.4 s (weights transfer 6.2 s).

| Check | Result |
|---|---|
| `GET /health` | ✅ `{"status":"ok"}` |
| `GET /v1/models` | ✅ id `qwen3.8-27b-quasar-nvfp4`, `max_model_len: 98304` |
| `POST /v1/chat/completions` | ✅ finish `stop`, correct content |

96K startup memory picture: KV 98,304 tokens = 3.82 GiB; free after weights 7.09 GiB;
free after startup 3.33 GiB; slack 3.26 GiB; CUDA graph allowance 82 MiB; context-cache on.

Chat-completion response shape verified:

- `message.content` separated from `message.reasoning_content` (thinking on by default;
  51 reasoning tokens of 61 completion tokens in the test request);
- usage accounting correct (66 prompt + 61 completion = 127 total);
- auth disabled when started without `--api-key`.

## 6. Staged context tuning (tutorial §13)

Each stage launched `ninfer-serve` with the profile below, polled `/health`, ran one
`/v1/chat/completions` request, and shut down. Harness: `.codex/ninfer-stage-test.sh`.

| # | Stage | max-context | kv-capacity | MTP | KV dtype | Result | Runtime KV | Free after startup | Slack |
|---|---|---:|---:|:---:|:---:|:---|---:|---:|---:|
| 1 | Smoke | 4,096 | 4,096 | on | int8 | ✅ | 0.13 GiB | 6.63 GiB | 6.63 GiB |
| 2 | Baseline | 32,768 | 32,768 | on | int8 | ✅ | 1.63 GiB | 5.52 GiB | 5.46 GiB |
| 3 | 96K (§11 profile) | 98,304 | 98,304 | on | int8 | ✅ | 3.82 GiB | 3.33 GiB | 3.26 GiB |
| 4 | Target | 131,072 | 131,072 | on | int8 | ✅ | 4.92 GiB | 2.24 GiB | 2.17 GiB |
| 5 | Target fallback | 131,072 | 131,072 | off | int8 | ✅ | 4.58 GiB | 3.26 GiB | 3.26 GiB |
| 6 | Capacity probe | 262,144 | auto | on | int8 | ❌ init failure | — | — | — |
| 7 | Capacity probe | 262,144 | auto | off | int8 | ❌ init failure | — | — | — |
| 8 | Auto probe | 131,072 | auto | off | int8 | ✅ resolved 131,072 | 4.58 GiB | 3.26 GiB | 3.26 GiB |
| 9 | FP8 experiment | 131,072 | 131,072 | on | fp8 | ✅ | 4.82 GiB | 2.34 GiB | 2.27 GiB |

Every passing stage returned a correct chat completion (`finish_reason: stop`).

### 6.1 Probe failures (stages 6–7) — exact errors

The 262K logical ceiling fails **before** KV sizing, at the Engine's minimum runtime
reservation check:

```text
[MTP]    minimum Engine runtime reservation requires 9987798272 bytes in addition to
         1073741824 bytes of automatic headroom, but only 7610317824 bytes are available
         after weights
[no-MTP] minimum Engine runtime reservation requires 9351343872 bytes in addition to
         1073741824 bytes of automatic headroom, but only 8418625536 bytes are available
         after weights
```

Interpretation: the Engine requires ~9.99 GB (MTP) / ~9.35 GB (no-MTP) of runtime reservation
**plus** 1 GiB of auto-sizing headroom, independent of the KV pool size. With 16.06 GiB of
weights on a 24 GiB card, only 7.6–8.4 GiB remains, so `--kv-capacity auto` never reaches the
point of resolving a physical capacity. A 262K logical ceiling is **not reachable** on 24 GiB
with this artifact, regardless of MTP.

## 7. Findings and recommendations

1. **128K works with MTP on the first attempt.** 131,072 / MTP / int8 leaves 2.24 GiB free
   after startup and 2.17 GiB of planned slack. No fallback was needed.
2. **MTP costs ~0.75 GiB of residency plus CUDA-graph memory.** Dropping MTP at 128K raises
   free-after-startup from 2.24 → 3.26 GiB but reduces the CUDA graph allowance from
   82 MiB → 12 MiB. Since MTP fits, it should stay enabled: it delivered 52% acceptance and
   2.56 tokens/round in the smoke test.
3. **The 262K ceiling is a hard floor, not a tuning problem.** The Engine's minimum runtime
   reservation exceeds available VRAM after weights. Do not retry without changing the memory
   picture (e.g., smaller artifact or more VRAM).
4. **`--kv-capacity auto` resolves to the logical ceiling when memory is plentiful** (stage 8
   resolved the full 131,072). A fixed `--kv-capacity` is preferable for predictable admission
   behavior.
5. **FP8 KV is viable at 128K** (4.82 GiB runtime vs 4.92 GiB for int8) but the tutorial's
   guidance stands: INT8 is the explicitly validated path for this artifact, so INT8 remains
   the default recommendation.
6. **Context accounting:** at 131,072, a 123K prompt + 8,192 `max_tokens` exactly saturates the
   logical ceiling; requests above that cannot be admitted.

### Recommended daily profile for this machine

| Setting | Value |
|---|---|
| `--max-context` / `--kv-capacity` | 131,072 (conservative: 98,304) |
| `--kv-dtype` | int8 |
| Speculation | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| `--max-concurrency` | 1 |
| `--device-state-slots` / `--host-state-slots` / `--host-kv-mib` | 1 / 2 / 2048 |
| `--default-max-tokens` | 8192 |
| Bind / port | `127.0.0.1` / `18080` |

## 8. Reproduction

Build and verify exactly as in the tutorial (§5–§9 above), then either run the individual
commands quoted in sections 4–6, or use the staged harness:

```bash
# usage: <max-context> <kv-capacity> mtp|nomtp [int8|fp8]
bash .codex/ninfer-stage-test.sh 131072 131072 mtp int8
```

The harness launches `ninfer-serve`, polls `/health`, runs one chat completion, prints the
resolved KV line from the startup log, and stops the server. A copy also lives at
`~/.local/share/ninfer/ninfer-stage-test.sh` for use outside the repository.

## 9. Remaining tutorial work

Sections 14–15 (MCC provider configuration, `~/.local/bin/ninfer-qwen38-quasar` launcher,
`~/.config/ninfer/qwen38.env`, and the `~/.config/systemd/user/ninfer-qwen38.service` unit)
have not been executed yet. The validated 128K profile above should be used as the launcher
default, with 98,304 as the conservative fallback.
