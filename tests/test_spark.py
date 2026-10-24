"""Validate that cuRobo works on this machine (DGX Spark / GB10)."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_visible():
    assert torch.cuda.is_available(), "torch can't see a GPU - check nvidia-smi / driver"
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))


def test_kernels_compile_for_this_gpu():
    major, minor = torch.cuda.get_device_capability(0)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "poc" / "nvrtc_smoke.py"), f"sm_{major}{minor}"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def planner():
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    p = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml", scene_model="collision_test.yml"))
    p.warmup(enable_graph=True, num_warmup_iterations=5)
    return p


def test_forward_kinematics(planner):
    from curobo.types import JointState

    q = planner.default_joint_state.position.unsqueeze(0)
    state = planner.kinematics.compute_kinematics(JointState.from_position(q, joint_names=planner.joint_names))
    pose = state.tool_poses.get_link_pose(planner.tool_frames[0])
    assert torch.isfinite(pose.position).all()


def test_plan_pose(planner):
    from curobo.types import GoalToolPose, JointState

    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
    )
    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[[0.5, 0.0, 0.3]]]]], device="cuda"),
        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], device="cuda"),
    )
    result = planner.plan_pose(goal, q_start)
    assert result is not None and result.success.any(), "motion planning failed"
    assert result.get_interpolated_plan().position.shape[-2] > 1


def test_official_examples(tmp_path):
    """cuRobo's own getting-started example: pose + grasp planning."""
    proc = subprocess.run(
        [sys.executable, "-m", "curobo.examples.getting_started.motion_planning",
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "✓ Planning succeeded" in proc.stdout
    assert "✓ Grasp planning succeeded" in proc.stdout
