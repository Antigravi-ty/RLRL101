"""core/constants.py —— 1v1 最快开球 (Fastest Kickoff Duel) 核心物理与网络常量。"""
import math

PHYSICS_HZ = 120
DECISION_HZ = 30
TICKS_PER_DECISION = PHYSICS_HZ // DECISION_HZ  # 4 ticks per decision step

# 开球重置后先让物理引擎空跑 60 frames (0.50s) 稳定悬挂，再允许模型/人类输入
SETTLE_TICKS = 60
GROUND_SETTLE_TICKS = SETTLE_TICKS

# 单回合最大时长: 6.0 秒 = 720 ticks = 180 决策步
MAX_KICKOFF_SECONDS = 6.0
MAX_TICKS = int(MAX_KICKOFF_SECONDS * PHYSICS_HZ)          # 720 frames
MAX_DECISIONS = MAX_TICKS // TICKS_PER_DECISION            # 180 decisions
MAX_DECISIONS_STAGE0 = MAX_DECISIONS
POST_TOUCH_STEPS = 0

BALL_RADIUS = 93.15
TOUCH_DIST = 200.0
TOUCH_SURFACE_DIST = 150.0

# Rocket League 5 个标准对称开球位 (对应 arena.reset_kickoff(seed) 的确定性种子)
NUM_KICKOFF_SLOTS = 5
KICKOFF_SLOT_SHORT = ("DR", "DL", "BR", "BL", "CT")
KICKOFF_SLOT_NAMES = (
    "0:DiagRight (右斜位 -2048,-2560)",
    "1:DiagLeft  (左斜位 +2048,-2560)",
    "2:BackRight (右中后  -256,-3840)",
    "3:BackLeft  (左中后  +256,-3840)",
    "4:Center    (门前正中   0,-4608)",
)
# Verified RocketSim reset_kickoff(seed) seeds that deterministically spawn each of the 5 standard slots
KICKOFF_SLOT_SEEDS = (
    525901256,  # 0: DiagRight (-2048, -2560)
    53710184,   # 1: DiagLeft  ( 2048, -2560)
    220106707,  # 2: BackRight ( -256, -3840)
    239081663,  # 3: BackLeft  (  256, -3840)
    479341423,  # 4: Center    (    0, -4608)
)

BLUE_KICKOFF_SLOTS = (
    (-2048.0, -2560.0, 17.0, 0.25 * math.pi),  # 0: 右斜位 (Diag Right)
    ( 2048.0, -2560.0, 17.0, 0.75 * math.pi),  # 1: 左斜位 (Diag Left)
    ( -256.0, -3840.0, 17.0, 0.50 * math.pi),  # 2: 右中后 (Back Right)
    (  256.0, -3840.0, 17.0, 0.50 * math.pi),  # 3: 左中后 (Back Left)
    (    0.0, -4608.0, 17.0, 0.50 * math.pi),  # 4: 门前正中 (Goalie Center)
)

# 极简 42 维观测空间 (无冗余 BoostPad Attention，纯向量极速推理):
#   0..32  (33): OBS_PANEL_DIM (A.face_dists[6], B.car_ang_b[3], C.car_vel_b[4], D.ball_rel_b[7], E.flags[6], F.closing[1], G.prev_act[6])
#   33..35 (3) : H.car_orient (fwd_z, right_z, up_z)
#   36..41 (6) : I.opp_rel_b  (opp_fwd, opp_right, opp_up, opp_dist, opp_vfwd, opp_closing)
OBS_PANEL_DIM = 33
TRUNK_OBS_DIM = 42
OBS_DIM = 42
OBS_V3_DIM = OBS_DIM

# 关键观测索引 (用于物理动作掩码 Action Mask)
IDX_ON_GROUND = 20
IDX_HAS_FLIP_OR_JUMP = 21
IDX_IS_FLIPPING = 22
IDX_IS_JUMPING = 23
IDX_AIR_TIME = 24
IDX_BOOST = 25

# 轻量化纯 MLP 网络隐藏层维度 (128 维双层 Tanh MLP，总参数仅 ~4.6 万，CPU 采样与 MPS 反向传播极快)
PPO_HIDDEN_DIM = 128
LOGIT_CLAMP_RANGE = 2.50

PPO_ACTION_DIMS = [3, 3, 3, 2, 2, 2]
PPO_LOGITS_DIM = sum(PPO_ACTION_DIMS)
PPO_CHANNEL_NAMES = [
    "thr_pitch(S/0/W)",
    "steer_yaw(A/0/D)",
    "roll(Q/0/E)",
    "jump",
    "boost",
    "handbrake",
]

# 6秒有限步开球专用长视距折扣参数与并行 Worker 数
DEFAULT_GAMMA = 0.998
DEFAULT_GAE_LAMBDA = 0.975
DEFAULT_WORKERS = 8

# 评测与兼容常量 (5 个标准开球位)
NUM_CURRICULUM_STAGES = NUM_KICKOFF_SLOTS
NUM_CURRICULUM_STEPS = NUM_KICKOFF_SLOTS
EVAL_SAMPLES_PER_STAGE = 4
DIR5_LABELS = KICKOFF_SLOT_SHORT
NUM_DIR5 = NUM_KICKOFF_SLOTS
