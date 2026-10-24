"""Offline NVRTC compile of cuRobo kernels for DGX Spark (GB10, sm_121). No GPU/driver needed."""

import sys

from cuda.core import Program, ProgramOptions

from curobo._src.curobolib.backends.cuda_core_backend.dynamics_config import DynamicsKernelCfg
from curobo._src.curobolib.backends.cuda_core_backend.geometry_config import GeometryKernelCfg
from curobo._src.curobolib.backends.cuda_core_backend.kernel_cache import CudaCoreKernelCache
from curobo._src.curobolib.backends.cuda_core_backend.kinematics_config import KinematicsKernelCfg
from curobo._src.curobolib.backends.cuda_core_backend.optimization_config import (
    OptimizationKernelCfg,
)
from curobo._src.curobolib.backends.cuda_core_backend.pba_config import PBAKernelCfg
from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg

ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"

# (cfg, kernel_type, name_expressions to instantiate)
CASES = [
    (KinematicsKernelCfg(), "forward",
     ("curobo::kinematics::kinematics_forward_kernel<8, false>",)),
    (KinematicsKernelCfg(), "backward", ()),
    (GeometryKernelCfg(), "self_collision", ()),
    (DynamicsKernelCfg(), "forward", ()),
    (DynamicsKernelCfg(), "backward", ()),
    (TrajectoryKernelCfg(), "bspline_forward", ()),
    (TrajectoryKernelCfg(), "differentiation_forward", ()),
    (TrajectoryKernelCfg(), "integration", ()),
    (OptimizationKernelCfg(), "line_search", ()),
    (OptimizationKernelCfg(), "lbfgs", ()),
    (PBAKernelCfg(), None, ()),
]

cache = CudaCoreKernelCache()
failed = 0
for cfg, ktype, names in CASES:
    files = cfg.get_kernel_files(ktype) if ktype else cfg.get_kernel_files()
    paths = [cfg.kernel_dir / f for f in files]
    src = cache._read_sources(paths, cfg.get_include_dirs())
    opts = ProgramOptions(
        std="c++17", arch=ARCH, include_path=[str(p) for p in cfg.get_include_dirs()],
        ftz=True, fma=True, prec_div=False, prec_sqrt=False, lineinfo=True,
        device_code_optimize=True,
    )
    label = f"{type(cfg).__name__}:{ktype}"
    try:
        mod = Program(src, code_type="c++", options=opts).compile(
            "cubin", name_expressions=names
        )
        print(f"OK   {label}  ({len(mod.code)} bytes cubin)")
    except Exception as e:
        failed += 1
        print(f"FAIL {label}: {str(e)[:2000]}")

print(f"\n{len(CASES) - failed}/{len(CASES)} kernel groups compiled for {ARCH}")
sys.exit(1 if failed else 0)
