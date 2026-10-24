# cuRobo on DGX Spark

Proof of concept: [NVlabs cuRobo](https://github.com/NVlabs/curobo) running natively on a DGX Spark
(aarch64, GB10 GPU / `sm_121`, CUDA 13). No Docker, no sudo, only `uv`.

## Setup

Prerequisites: an NVIDIA driver that loads (`nvidia-smi` shows the GB10) and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

That's it. `uv sync` creates `.venv/` and installs:

- `torch` (cu130 build for aarch64) from the PyTorch index
- `nvidia-curobo[cu13-torch]` straight from GitHub, pinned to commit `78fd485` (2026-09-10),
  plus its dependencies (`warp-lang`, `cuda-core`, ...)

Both sources are set in `pyproject.toml` under `[tool.uv.sources]`.

### Why no build step?

Current cuRobo no longer ships compiled CUDA extensions. Its kernels are compiled at runtime
with NVRTC (via `cuda.core`) for whatever GPU is present, and warp JIT-compiles its own kernels
too. So nothing is built at install time: the **first run takes ~30 s** to compile kernels,
after which they're cached in `~/.cache` (warmup is then ~3 s).

This is why cuRobo works on the Spark even though older versions didn't: the old versions
needed nvcc-built extensions and x86/CUDA 12 wheels.

## Validate

```bash
scripts/test.sh
```

Runs `tests/test_spark.py` (~15 s once kernels are cached): GPU visible, kernels compile for
this GPU, forward kinematics, a motion plan, and cuRobo's official example (pose + grasp).

## Viewer

```bash
scripts/serve.sh        # port 8080, or: scripts/serve.sh 8081
```

Open `http://<spark-tailscale-ip>:8080`. Stop with Ctrl+C.

## Run manually

```bash
# Official example: pose-to-pose + grasp planning (plots go to poc/out/)
uv run python -m curobo.examples.getting_started.motion_planning --output-dir poc/out

# Timing: 20 plans to random reachable poses
uv run python poc/plan_timing.py

# Interactive 3D viewer in the browser on port 8080
uv run python -m curobo.examples.getting_started.motion_planning --visualize
```

The viewer listens on all interfaces, so over Tailscale open `http://<spark-tailscale-ip>:8080`.

Results on the GB10 (2026-10-24): 20/20 plans succeeded, median **~35 ms** per plan (Franka arm, `collision_test` scene).

## Files

| Path | What |
|---|---|
| `pyproject.toml` | uv project: torch cu130 + cuRobo from git |
| `poc/plan_timing.py` | Planning timing benchmark |
| `poc/nvrtc_smoke.py` | Offline check: compiles all cuRobo kernel groups for `sm_121` (no GPU needed) |
| `poc/out/` | Plots from the example run |
| `tests/test_spark.py` | Validation tests |
| `scripts/test.sh` | Run the tests |
| `scripts/serve.sh` | Start the interactive viewer |

## Hacking on cuRobo itself

To edit cuRobo's source, clone it and switch the source to an editable path:

```bash
git clone https://github.com/NVlabs/curobo.git src
```

```toml
# pyproject.toml, [tool.uv.sources]
nvidia-curobo = { path = "src", editable = true }
```

Then run `uv sync` again.
