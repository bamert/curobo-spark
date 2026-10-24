"""Time cuRobo pose-to-pose motion planning on DGX Spark (Franka, collision_test scene)."""

import time

import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState

N = 20

planner = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml", scene_model="collision_test.yml"))
t0 = time.perf_counter()
planner.warmup(enable_graph=True, num_warmup_iterations=5)
print(f"warmup: {time.perf_counter() - t0:.2f}s")

q_start = JointState.from_position(
    planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
)

# Reachable goals: FK of random joint configs near the default state.
g = torch.Generator(device="cuda").manual_seed(0)
q_rand = planner.default_joint_state.position + 0.6 * (
    torch.rand(N, q_start.position.shape[-1], device="cuda", generator=g) - 0.5
)
goals = planner.kinematics.compute_kinematics(JointState.from_position(q_rand, joint_names=planner.joint_names))

times, ok = [], 0
for i in range(N):
    pose = goals.tool_poses.get_link_pose(planner.tool_frames[0])
    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=pose.position[i].view(1, 1, 1, 1, 3),
        quaternion=pose.quaternion[i].view(1, 1, 1, 1, 4),
    )
    torch.cuda.synchronize()
    t = time.perf_counter()
    result = planner.plan_pose(goal, q_start)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t)
    ok += int(result is not None and bool(result.success.any()))

times = torch.tensor(times) * 1000
print(f"success: {ok}/{N}")
print(f"plan time ms: mean {times.mean():.1f}  median {times.median():.1f}  min {times.min():.1f}  max {times.max():.1f}")
