# Strata sm_75 server

This tree is the CLI and server distribution for one production shape:

- Linux x86_64
- CUDA compute capability 7.5 (`sm_75`)
- 2 x NVIDIA RTX 2080 Ti 22 GB
- Qwen3.8-Flash-Next
- 6 serving slots, 256K context, INT8 KV cache and two batch groups

The retained product is the `strata` command line plus the OpenAI/Anthropic-compatible server. The desktop app,
installers, updater, model marketplace, non-CUDA backends and engines for other model families are not part of this
distribution.

## DFlash2

DFlash2 is retained as an optional drafter for Qwen3.8-Flash-Next. Its engine kernels, server wiring and pack tools
remain in this tree. Select it in the run config with:

```json
{
  "drafting": "dflash2",
  "args": [
    "--mtp", "/path/to/model/mtp",
    "--max-context", "262144"
  ]
}
```

The server looks for `<data-dir>/dflash2/qwen3.8-flash-next-dflash2.q8.pack`, where `data-dir` is inferred from the
model's `--mtp` or `--pack` path. It adds `--draft dflash2 --draft-model <pack>` when the pack is ready. A missing pack,
a different model, or an allocation failure falls back to MTP and reports the reason in logs and telemetry.

`drafting: "auto"` still chooses MTP. The current tree has no sm_75 prompt-class sweep showing DFlash2 winning every
case, so choosing DFlash2 is explicit. The production server currently lacks the DFlash2 pack, so this refactor does
not claim that DFlash2 has run on the target host.

## Layout

| Path | Purpose |
|---|---|
| `strata.py`, `strata_cli/` | server CLI, saved profiles, local supervisor and read-only telemetry clients |
| `strata_control/` | profile, runtime, CUDA-only component, production-shape and drafting logic |
| `serve/` | OpenAI and Anthropic APIs, admission, batching, streaming and tool calls |
| `src/`, `include/`, `cmake/` | CUDA `sm_75` engine sources |
| `tools/server/` | measured server probes and production environment evidence |
| `tools/dflash2/` | DFlash2 pack and evaluation tools |
| `docs/STATE.md`, `docs/HANDOVER.md`, `docs/ITERATION_LOG.md`, `docs/EVIDENCE_LEDGER.md` | current server state and measured evidence |

## Build

Requirements are Linux, CMake 3.24 or newer, CUDA 13.0 at the path selected by CMake, and the engine's normal C++
dependencies.

```sh
cmake -S . -B build-sm75 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-sm75 --target strata -j
```

The build refuses Linux-incompatible hosts, CUDA architectures other than 75, and removed HIP, SYCL, Metal and
experimental pre-Turing options.

## Use

The server starts from an existing run config and existing model files. Start with read-only checks:

```sh
python3 strata.py --help
python3 strata.py doctor --profile /opt/ai/src/strata-0.2.2/strata-022.json --live
python3 strata.py check-production --config /opt/ai/src/strata-0.2.2/strata-022.json
python3 strata.py serve --config /opt/ai/src/strata-0.2.2/strata-022.json --plan
```

Run in the foreground:

```sh
python3 strata.py serve --config /opt/ai/src/strata-0.2.2/strata-022.json
```

Or save the config as a profile and manage it through the owner-local runtime:

```sh
python3 strata.py profile create production \
  --from-config /opt/ai/src/strata-0.2.2/strata-022.json
python3 strata.py runtime start --profile production
python3 strata.py runtime verify --profile production --load
python3 strata.py runtime status --profile production
python3 strata.py runtime stop --profile production
```

`strata launch` is only a compatibility alias for `strata serve`. It does not install, download or configure a model
or engine.

Read-only service inspection:

```sh
python3 strata.py status --url http://127.0.0.1:8080
python3 strata.py monitor --url http://127.0.0.1:8080 --once
python3 strata.py logs --url http://127.0.0.1:8080 --output json
```

The interactive `monitor` needs the optional `textual` package:

```sh
python3 -m pip install -r requirements-monitor.txt
```

## Production shape

`strata check-production` compares the run config and effective environment with the documented host profile:

```text
backend=cuda
arch=qwen4exp
parallel=6
batch=6
batch_groups=2
max_context=262144
kv=int8
kv_resident=32768
```

It also checks the critical variables in
[`tools/server/production-env-2080ti.json`](tools/server/production-env-2080ti.json).
`STRATA_ADMIT_SHORT_FIRST=2048` must be present in the systemd service environment. Other switches may be in the run
config's `env` map or the server process environment.

This check is static. A passing result does not prove that the engine binary was built for `sm_75`, that both cards
are healthy, or that a 256K mixed soak passed.

## API

The server exposes:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/messages`
- `GET /v1/models`
- `GET /props`, `/health`, `/slots`, `/metrics`
- `GET /v1/telemetry` for read-only monitoring

## Verification

Checks that do not need the target GPU:

```sh
python3 -m compileall -q serve strata_cli strata_control strata.py
python3 -m pytest strata_control/test_sm75_profile.py \
  strata_control/test_drafting.py strata_control/test_engine_component.py -q
python3 -m pytest tests/terminal/test_profiles.py tests/terminal/test_runtime.py \
  serve/test_parallel.py tests/terminal/test_overview.py -q
git diff --check
```

CUDA compilation, engine load, two-card telemetry, API generation over the real model, DFlash2 with its pack, and a
256K mixed soak require the production host and must be reported separately after they are run there.
