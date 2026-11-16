# Analytical SOL references

SOL estimates device throughput for the specified workload, execution graph,
and hardware assumptions. These are analytical references, not measured
service throughput, formal rewards, or proofs of a globally optimal configuration.

| Task | H20 SOL | H200 SXM SOL | Unit |
| --- | ---: | ---: | --- |
| Qwen3.8 27B | 300.224167 | 395.415134 | tokens/s |
| Muse Glimmer 30B | 71.141944 | 88.550921 | tokens/s |
| Qwen3 VL 32B | 301.400723 | 729.139081 | tokens/s |
| LLaDA2.1 mini | 642.288852 | 1054.792824 | tokens/s |
| Wan2.2 TI2V 5B | 10.209773 | 65.953724 | videos/hour |

The [H20 results](evidence/sol/h20.json) and [H200 results](evidence/sol/h200.json)
contain unrounded values. [Hardware presets](../assets/sol-presets.json) bind
these files and their SHA-256 identities to the scorer. Ratios are computed
against the selected hardware's baseline at read time:

```bash
python -m schemas.sol_presets
python -m schemas.sol_presets --hardware H200 --task infer-llada2p1mini-16b --json
```

## Method

AISimulate operator-level costs use
`database_mode=SOL`, `fallback_policy=deny`, with learned correction disabled.
Compute and memory costs are evaluated per operator. Native forward shapes,
prefix/cache decisions, acceptance, and routing remain fixed during hardware
recosting. Additional H200 memory does not imply new cache hits or batching.

H200 assumes one **SXM** GPU with dense BF16 989.5 TFLOPS, dense FP8 1979 TFLOPS,
strict FP32 67 TFLOPS, and HBM bandwidth 4.8 TB/s. The dense tensor-core rates
are half the sparse rates in the [NVIDIA specification](https://www.nvidia.com/en-us/data-center/h200/).
This assumption is not a verification of a Modal worker's SKU and does not
apply to H200 NVL. H20 uses the task-specific [hardware models](evidence/sol/systems/).

| Task | Workload and scope |
| --- | --- |
| Qwen3.8 | Native SGLang trajectory with approximate MTP cost. H200 is a prefix-rate estimate; available records cover 2771.847790 analytical seconds, not a complete hour. |
| Muse | Native vLLM scheduler/KV replay with a fixed 16 GiB KV budget. H200 is a prefix-rate estimate with 2892.245469 analytical seconds of records. |
| Qwen3 VL | 32 requests and 7,968 output tokens; native language forwards plus analytical vision encoding. CPU preprocessing and transport are omitted. |
| LLaDA | One 140-request round, 57,329 output tokens, 40,458 forwards, and 768,702 native per-layer expert histograms. Routing is observed on H20 and repriced for H200. |
| Wan | Fixed text encoder, 50-step dual-branch CFG denoising, and approximate VAE graph for 1280×704, 121-frame video. Native runtime, MP4 encoding, and delivery overhead are omitted. |

LLaDA includes full-logits projection, bidirectional block attention, a strict
FP32 router, sampling, and native distinct-expert counts. Each accessed BF16
expert weight is assumed read once per forward. Physical HBM traffic, L2 hits,
padding, redundant loads, and auxiliary implementation overhead are unmeasured.
Its analytical totals are 89.257349 seconds on H20 and 54.350958 seconds on H200.

## Reproduce

The [numerical inputs](evidence/sol/) include model geometry, hardware models,
anonymous workload aggregates, and source hashes. They contain no prompts or
private runtime locations. H20 control costs are reproduced before reporting
H200 results.

For LLaDA, Python and PyYAML suffice to evaluate retained operator budgets and
the explicit native-routing formulas:

```bash
python scripts/analysis/reprice_llada_routing_sol.py --output /tmp/llada-sol.json
```

Add `--aisimulate-source /path/to/aisimulate/python/aisimulate/src` and, if needed,
`--native-package-source /path/to/built-aisimulate/python/aisimulate/src` to check
retained operators against AISimulate. The included [LLaDA receipt](evidence/sol/llada.json)
records agreement across 999 contexts and 12,987 retained operators, alongside
per-phase costs and input hashes.

For the other four tasks, use an external AISimulate build matching the source
and native-module hashes in the [calculation receipt](evidence/sol/four-tasks.json):

```bash
python scripts/analysis/reprice_lite_sol.py \
  --aisimulate-source /path/to/aisimulate/python/aisimulate/src \
  --native-package-source /path/to/built-aisimulate/python/aisimulate/src \
  --output /tmp/lite-sol.json
```

The build must provide the model implementations recorded in the receipt,
including the Wan graph. The commit alone does not identify local model-support
changes; loaded source and native-module hashes are part of the calculation
identity. These scripts launch no GPU resources.
