# QUASAR Performance Exploration Plan — RTX 5090 Laptop

## Status and scope

This document records future performance work after merging the working QUASAR NVFP4 branch with
Neroued upstream through `21a0e85f`. The validated integration landed on `master` through merge
commit `8f4f95c8`; complete post-merge correctness and capacity results are in
[`ninfer-quasar-post-merge-test-results.md`](ninfer-quasar-post-merge-test-results.md).

The current evidence establishes correctness and compatibility, not a performance improvement. One
cold 4K CLI smoke comparison measured the merged build about 3% below the Mirko baseline while
producing the same token count and exact MTP acceptance statistics. That sample is insufficient to
classify either a regression or an improvement. Performance claims require repeated, controlled
measurements with identical artifacts, prompts, settings, and machine state.

The governing objective is maximum useful QUASAR serving performance on one RTX 5090 Laptop GPU
without weakening exact greedy parity, artifact compatibility, numerical qualification, or the
startup memory contract.

## Current performance expectations

For the established `MTP3 + INT8 KV + concurrency 1` profile, raw throughput is expected to remain
approximately neutral until targeted kernel work is completed. The upstream merge mainly adds:

- NVFP4 and K8V4 KV modes with lower physical KV residency;
- FP16 V storage and PV compute for the BF16-named KV profile;
- fused TMA NVFP4 SwiGLU routing for all supported multiples of 256;
- value-aware shared-prefix scheduling and pressure fixes;
- NVTX instrumentation needed for reliable bottleneck analysis.

These changes are most likely to improve memory-constrained concurrency, long-context admission,
prefix-reuse TTFT, and non-1024 prefill chunks. They do not imply an automatic single-request
decode speedup.

For one D256 K/V head and token, physical storage is:

| KV profile | Physical bytes | Relative to INT8 | Storage-only capacity ratio vs. INT8 |
|---|---:|---:|---:|
| BF16 K + FP16 V | 1,024 | 1.94× | 0.52× |
| INT8 group-64 | 528 | 1.00× | 1.00× |
| FP8 row-scaled | 516 | 0.98× | 1.02× |
| K8V4 | 402 | 0.76× | 1.31× |
| NVFP4 | 288 | 0.55× | 1.83× |

The capacity ratios are theoretical KV-only limits. Actual Engine gains are smaller because
weights, State, workspaces, CUDA Graphs, and MTP residency are fixed costs.

## Phase 1 — establish a controlled performance baseline

Before changing kernels, compare Mirko baseline `2d11c992` and current `master` on the same machine
and artifact.

### Workloads

1. Ordinary decode at concurrency 1.
2. MTP3 decode at concurrency 1, 2, and 4.
3. Prefill at approximately 1K, 32K, and 128K context.
4. Prefill chunks of 256, 512, 768, and 1,024 tokens.
5. INT8, FP8, K8V4, and NVFP4 KV on current `master`.
6. Repeated-prefix and Host-resume TTFT scenarios.

### Measurements

Record at minimum:

- prefill tokens/s;
- committed decode tokens/s;
- request makespan and TTFT;
- average and exact decode batch;
- MTP drafted and accepted tokens, acceptance by position, and committed tokens per round;
- GPU memory after weights and startup, runtime reservation, planned slack, and resolved KV capacity;
- Device/Host KV transfer bytes and restore duration;
- output token IDs for greedy-parity workloads;
- quality or perplexity for lossy KV profiles.

Use warm-up followed by multiple fixed-seed repetitions. Report medians and dispersion; preserve raw
JSON/log records. Do not compare stochastic runs by tokens/s without also retaining generated-token
counts and MTP acceptance.

### Profiling

Use the merged NVTX instrumentation and Nsight Systems/Compute to separate:

- MTP proposal, verification, and commit;
- causal softmax attention;
- GDN input/gating/output projections;
- NVFP4 MLP/SwiGLU and LinearAdd;
- KV transform, append, and decode;
- sampling and host scheduling;
- CUDA launch, graph replay, and synchronization overhead.

Capture ordinary C=1, MTP3 C=1, and MTP3 C=4 first. Optimization work should follow measured GPU
and wall-clock ownership rather than kernel size or intuition.

## Phase 2 — low-risk, hardware-specific kernel experiments

### 2.1 Port Mirko's small-batch NVFP4 SwiGLU tuning

Mirko's `perf/nvfp4-swiglu-m16n256` line contains:

- `81e685fc` — `perf(nvfp4): optimize swiglu for small batches`.

Port the launch geometry and route changes onto current `master` rather than cherry-picking the old
commit. Benchmark the public NVFP4 SwiGLU Op at effective compact widths produced by:

```text
effective columns = active requests × verification width
```

Cover widths 1–48, not only T=1 and T=1,024. Evaluate CTA shape, warps per CTA, minimum blocks per
SM, split policy, and route crossover.

### 2.2 Port RTX 5090 Laptop MTP tuning

Mirko's hardware-specific line also contains:

- `bf768886` — `perf(cuda): tune mtp verification for laptop 5090`.

Review and independently port applicable changes affecting:

- NVFP4 GDN input projection;
- GQA attention decode;
- NVFP4 LinearAdd route selection.

Each subchange should have an operator benchmark and qualification result before combination. Old
commits predate the merged KV and attention architecture and must not be cherry-picked blindly.

### 2.3 Extend fused TMA SwiGLU to prefill tails

Current routing uses the fused TMA NVFP4 SwiGLU path for every multiple of 256. Widths above 48 that
are not multiples of 256 still materialize a `34816 × T` BF16 gate/up intermediate and invoke a
separate SiLU/multiply path.

Explore:

- masked final TMA tiles;
- workspace padding to 256;
- fused full blocks plus a baseline remainder;
- specialized fused routes for common 64-, 128-, and 192-token tails.

Measure final-chunk latency and full-prompt prefill, including non-default chunk sizes. A gain on an
isolated tail kernel is useful only if it improves end-to-end prefill.

## Phase 3 — compact MTP verification without serial column launches

The current INT8 compact verification path processes widths 2–6 one causal column at a time to
preserve the exact ordinary-decode arithmetic:

```cpp
if (cache.storage == KvCacheStorage::Int8Group64 && width > 1 &&
    width <= kSmallTChunkTokens) {
    launch_chunked_small_t(..., chunk_tokens = 1, ...);
}
```

This correctness-first serialization is a likely MTP hot-path opportunity.

### Proposed design

Build a compact batched kernel that:

- processes all verification columns in one launch;
- retains the canonical one-column reduction geometry and order for each column;
- assigns independent split-local State to each column;
- publishes appended K/V in causal order without cross-column races;
- reuses K/V tiles across adjacent columns where this does not alter arithmetic;
- produces bit-identical output to the current serial implementation.

A column dimension in the grid can remove repeated host launches while preserving independent
per-column reductions. More aggressive CTA-level column grouping should follow only after the
single-launch implementation is qualified.

### Acceptance criteria

- exact greedy output across BF16 and INT8 KV;
- ordinary decode and MTP k1–k5;
- concurrency 1–8;
- repeated maximum-concurrency slot reuse;
- no increase in workspace or graph residency that offsets the latency gain;
- lower verification wall time and higher committed decode tokens/s.

## Phase 4 — adaptive MTP within the registered k1–k5 contract

A fixed MTP3 window is not necessarily optimal across prompts and generation phases. The observed
QUASAR smoke acceptance was approximately 52%, with acceptance declining by draft position.

Mirko's experimental branches contain:

- `298a37b1` — adaptive draft windows through k15;
- `0df999b3` — reduced adaptive graph residency.

Use these as design references, but initially stay inside the current registered k1–k5 contract.

### Policy objective

Select the window maximizing expected committed tokens per unit wall time rather than acceptance
alone:

```text
utility(k) = expected committed tokens(k) / measured round time(k)
```

The policy should:

- estimate positional acceptance per request or workload cohort;
- include proposal and verification cost;
- use hysteresis and a minimum observation window;
- avoid changing graph width every round;
- retain graphs only for widths that provide positive measured value;
- expose selected-width and utility data in request/benchmark logs.

Do not expand QUASAR to k6–k15 until the pinned MTP module, workspace, graph residency, exact parity,
and capability quality are separately validated.

## Phase 5 — convert low-bit KV memory savings into throughput

### 5.1 Compare INT8, FP8, K8V4, and NVFP4

Benchmark each profile at 4K, 32K, 96K, and 128K context with concurrency 1, 2, and 4. Separate
prefill, steady decode, and complete request makespan.

Possible product outcomes:

- INT8 remains the latency profile at C=1;
- K8V4 becomes the balanced long-context/concurrent profile;
- NVFP4 becomes the maximum-capacity/aggregate-throughput profile;
- compression overhead dominates, requiring codec optimization before recommendation.

A lower-memory profile is successful when it improves an observable serving outcome—admitted batch,
TTFT, makespan, retained prefixes, or tokens/s—not merely allocated bytes.

### 5.2 Fuse KV production and append

If profiles show codec overhead, explore:

- fusing rotation/Hadamard output directly into quantization;
- avoiding intermediate BF16 rotated vectors in global memory;
- warp-level group-max and scale generation in the producer;
- direct packed K/V and scale writes into paged storage;
- combining MTP append and small-T consumption where ownership permits;
- caching page-table metadata and scale addresses in registers;
- batching or overlapping Host/Device packed-KV transfers.

Qualification must compare independently represented K/V values and attention outputs, not only
packed bytes.

### 5.3 Improve BF16 compact execution

The BF16 small-T path preserves canonical-column behavior and now uses BF16 K, FP16 V, and FP32
split-local accumulation. Profile whether independent columns can be grouped while preserving the
same reduction order. Candidate work includes vectorized FP32 partial writes, K/V tile reuse, and a
fused compact-width reduction.

## Phase 6 — scheduling and context-cache serving efficiency

The value-aware scheduler targets avoided recomputation rather than raw kernel throughput. Measure:

- resource-search and materialization-planning duration;
- candidate portfolios evaluated per admission;
- Device/Host capture, restore, and eviction bytes;
- prefixes captured but never reused;
- hot, Host-resume, and cold TTFT;
- computed prefill tokens avoided by exact reuse.

Potential improvements include:

- caching stable portfolio valuations;
- bounding candidate expansion by measured transfer cost;
- batching and overlapping Host KV transfers;
- proactive retention of high-value system/tool prefixes;
- a simplified policy for the common fixed-concurrency-one laptop profile.

Scheduler changes must improve TTFT or complete-workload makespan without weakening admission,
reservation, or exact-prefix semantics.

## Prioritized execution order

1. **Controlled baseline and NVTX profile:** compare `2d11c992` with current `master`.
2. **KV profile matrix:** establish whether K8V4/NVFP4 provide useful throughput or capacity gains.
3. **Small kernel ports:** evaluate `81e685fc` and applicable parts of `bf768886` independently.
4. **Compact INT8 verification kernel:** replace serial column launches while preserving exact
   arithmetic.
5. **Adaptive MTP k1–k5:** optimize committed tokens per second using measured costs.
6. **Fused KV production/append:** pursue only if codec work owns meaningful trace time.
7. **Prefill-tail and scheduler work:** prioritize according to measured workload impact.

## Validation gates for every optimization

An optimization is complete only when the affected observable behavior and relevant performance are
both validated.

### Correctness

- affected public operator numerical qualification;
- QUASAR converter and real-artifact load-plan tests;
- exact BF16/INT8 greedy parity for ordinary decode, MTP k1–k5, and C=1–8;
- CLI and HTTP smoke tests;
- KV represented-value and attention-output oracles;
- quality/perplexity checks for K8V4 and NVFP4 KV;
- no regression in context admission, prefix reuse, or usage accounting.

### Performance

- identical artifact, prompt corpus, settings, seeds, and GPU power/thermal state;
- warm-up before measurement;
- repeated samples with median and dispersion;
- prefill, verification, and committed decode rates reported separately;
- MTP acceptance and generated-token counts retained with throughput;
- memory reservation, resolved KV capacity, average batch, and TTFT retained;
- end-to-end request or corpus makespan, not only isolated kernel time.

### Rollout

- land one coherent optimization at a time;
- preserve a benchmarkable baseline commit;
- do not combine unrelated route, arithmetic, and scheduler changes before attribution;
- update `docs/performance.md` only from reproducible campaigns;
- keep experimental results out of product defaults until the relevant workload shows a stable win.

## First recommended experiment

Profile MTP3 at concurrency 1 and 4 on INT8 KV, then measure the contribution of the serial compact
attention launches. In parallel, port the small-batch NVFP4 SwiGLU and laptop-specific MTP tuning as
separate benchmark branches. This provides a quick path to hardware-specific gains while the larger
single-launch canonical verification kernel is designed and qualified.
