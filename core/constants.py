"""所有常量。改这里一个地方，全局生效。"""
PHYSICS_HZ = 120
DECISION_HZ = 30
TICKS_PER_DECISION = PHYSICS_HZ // DECISION_HZ   # 4

# ── 地面生成悬架稳定丢弃帧数 (Settle Ticks) ──────────────
# RocketSim 地面生成在 z=17.0，存在约 30 ticks (0.25s) 的轮胎悬挂落地回弹与微小滑移。
# 在 reset() 时先自然空跑 KICKOFF_SETTLE_TICKS 帧且不计分，从真正完全静止的第 31 帧开始记录与决策。
KICKOFF_SETTLE_TICKS = 30

# ── Stage / 模式时长设定 ─────────────────────────────
# 全场随机地面球 / 定点球 / 空中球模式：最大 180 个决策步 (720 有效物理 ticks = 6.0 秒)
MAX_DECISIONS = 180
MAX_DECISIONS_STAGE0 = 180
# 触球后延迟观察帧数（默认 0：首次触球所在决策步/物理帧立即结算并终止，专心优化最快首触时间）
POST_TOUCH_STEPS = 0

TOUCH_DIST = 200.0   # 仅用于 HUD 显示，实际触球用 ball_touch_callback
BALL_RADIUS = 93.0

# 观测维度：
#   obs_v2 (31 维，供 HUD/Plot 面板分组可视化)
#   obs_v3 (75 维，PPO 专属全自中心观测：含自身姿态三轴 Z 分量、球面距、Big Pads 相对位置与倒计时、Oracle 意图条件接口)
OBS_DIM = 31
OBS_V3_DIM = 75
PPO_HIDDEN_DIM = 256

# ── PPO 多离散键盘动作空间 (Multi-Discrete 6 通道) ────
# 完美贴合键盘玩家：
#   0. throttle_pitch : [-1.0 (S), 0.0 (None), +1.0 (W)] -> 地面油门 + 空中俯仰绑定 (W: thr=+1, pitch=-1; S: thr=-1, pitch=+1)
#   1. steer_yaw      : [-1.0 (A), 0.0 (None), +1.0 (D)] -> 地面转向 + 空中偏航绑定
#   2. roll           : [-1.0 (Q), 0.0 (None), +1.0 (E)] -> 空中横滚
#   3. jump           : [0, 1] (Space / Slash)
#   4. boost          : [0, 1] (Period)
#   5. handbrake      : [0, 1] (Shift / Powerslide)
PPO_ACTION_DIMS = [3, 3, 3, 2, 2, 2]
PPO_LOGITS_DIM = sum(PPO_ACTION_DIMS)  # 15
PPO_CHANNEL_NAMES = [
    "thr_pitch(S/0/W)",
    "steer_yaw(A/0/D)",
    "roll(Q/0/E)",
    "jump",
    "boost",
    "handbrake",
]
