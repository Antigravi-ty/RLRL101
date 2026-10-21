"""core/constants.py —— Freestyle Air-Dribble Showman 物理与网络常量定义。"""

PHYSICS_HZ = 120
DECISION_HZ = 30
TICKS_PER_DECISION = PHYSICS_HZ // DECISION_HZ

# 15 seconds per training episode (450 decisions = 1800 ticks @ 120Hz); infinite in Tuner freerun
MAX_DECISIONS = 450
MAX_DECISIONS_STAGE0 = 360
POST_TOUCH_STEPS = 0

TOUCH_DIST = 200.0
BALL_RADIUS = 93.15

# Soccar Arena Geometry & Air-Box ("中间盒子区" — detached from all 6 arena planes)
FIELD_X = 4096.0
FIELD_Y = 5120.0
FIELD_Z = 2044.0
CEILING_Z = FIELD_Z
CORNER_SUM = 8064.0

GOAL_Y = 5120.0
GOAL_WIDTH_HALF = 893.0
GOAL_HEIGHT = 642.0
GOAL_DANGER_Y = 4450.0
GOAL_DANGER_X = 1150.0
GOAL_DANGER_Z = 750.0

# Air-Box boundaries: ball is inside the scoring Air-Box when detached from floor, ceiling, and walls
AIR_BOX_MIN_Z = 220.0
AIR_BOX_CLEARANCE = 180.0
SURFACE_CONTACT_DIST = 138.0

# Multi-touch combo chain parameters
TOUCH_DEBOUNCE_TICKS = 8
MIN_FULL_MULT_SEG_DIST = 220.0

# Boost modes: "unlimited" (locked at 100), "recharge" (air regen + touch refill), "pads" (standard 34 pads)
BOOST_MODES = ("unlimited", "recharge", "pads")
DEFAULT_BOOST_MODE = "unlimited"

# Air-Dribble Master observation dimensions (190 dims = 54 Trunk + 34 * 4 Boost Pads)
NUM_BOOST_PADS = 34
PAD_TOKEN_DIM = 4
PAD_OBS_DIM = NUM_BOOST_PADS * PAD_TOKEN_DIM  # 136
TRUNK_OBS_DIM = 54
OBS_DIM = TRUNK_OBS_DIM + PAD_OBS_DIM         # 190
OBS_V3_DIM = OBS_DIM
OBS_PANEL_DIM = 45                            # Trunk groups A..I for tuner plots

# Key observation indices in Trunk (0..53)
IDX_ON_GROUND = 20
IDX_HAS_FLIP_OR_JUMP = 21
IDX_IS_FLIPPING = 22
IDX_IS_JUMPING = 23
IDX_AIR_TIME = 24
IDX_BOOST = 25

# Boost Pad Attention hyperparameters
ATTN_NUM_HEADS = 2
ATTN_HEAD_DIM = 16
ATTN_OUT_DIM = ATTN_NUM_HEADS * ATTN_HEAD_DIM  # 32

PPO_HIDDEN_DIM = 256

# Global initial boost distribution when not using unlimited boost
LOW_BOOST_PROB = 0.30
LOW_BOOST_MAX = 50.0

LOGIT_CLAMP_RANGE = 2.25

# 5 Freestyle Spawn Categories:
#   C = Carry (空中带球初态)
#   A = Air-Catch (动态抛球接球)
#   B = Bounce (弹跳球挑球)
#   W = Wall-Roll (边墙滚球起飞)
#   G = Ground-Free (地面自由起球)
DIR5_LABELS = ("C", "A", "B", "W", "G")
DIR5_FULL_NAMES = ("Carry", "Catch", "Bounce", "Wall", "Ground")
NUM_DIR5 = 5
NUM_DIR5_BUCKETS = NUM_DIR5

# Alias for backward-compatible references
DIR6_LABELS = DIR5_LABELS
DIR6_FULL_NAMES = DIR5_FULL_NAMES
NUM_DIR6 = NUM_DIR5

VEL_DIR4_LABELS = ("F", "B", "R", "L")
NUM_VEL_DIR4 = 4

POOL_FRONTIER = 0
POOL_MEMORY   = 1
POOL_LABELS   = ("F", "M")
POOL_RATIOS   = (0.75, 0.25)

CAR_GND_Z = 17.0
CAR_REST_Z = CAR_GND_Z
CAR_MAX_SPAWN_Z = 1100.0
CAR_MAX_ANG_SPEED = 5.5
CAR_MAX_LIN_SPEED = 2300.0
GROUND_SETTLE_TICKS = 15

# Curriculum & Evaluation constants (4 Phases P1..P4, 5 scenario buckets, 16 progressive curriculum steps)
NUM_CURRICULUM_STAGES = 4
NUM_CURRICULUM_STEPS = 16
EVAL_SAMPLES_PER_STAGE = 6
EVAL_GRID_BINS = NUM_CURRICULUM_STAGES
EVAL_SAMPLES_PER_BIN = 6

DIST_TIERS = (
    (160.0,  750.0),   # D0: Near
    (750.0,  2100.0),  # D1: Mid
    (2100.0, 6000.0),  # D2: Far
)
NUM_DIST_TIERS = len(DIST_TIERS)
DIST_TIER_MAX_DECISIONS = (360, 450, 450)

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
