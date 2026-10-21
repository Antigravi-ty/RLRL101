PHYSICS_HZ = 120
DECISION_HZ = 30
TICKS_PER_DECISION = PHYSICS_HZ // DECISION_HZ

MAX_DECISIONS = 240
MAX_DECISIONS_STAGE0 = 180
POST_TOUCH_STEPS = 0

TOUCH_DIST = 200.0
BALL_RADIUS = 93.0

# Static Target Master observation dimensions (178 dims = 42 Trunk + 34 * 4 Boost Pads)
NUM_BOOST_PADS = 34
PAD_TOKEN_DIM = 4
PAD_OBS_DIM = NUM_BOOST_PADS * PAD_TOKEN_DIM  # 136
TRUNK_OBS_DIM = 42
OBS_DIM = TRUNK_OBS_DIM + PAD_OBS_DIM         # 178
OBS_V3_DIM = OBS_DIM
OBS_PANEL_DIM = 33                            # Trunk groups A..G for tuner plots

# Key observation indices in Trunk (0..41)
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

# Global initial boost distribution: 70% < 33, 30% in [33, 100]
LOW_BOOST_PROB = 0.70
LOW_BOOST_MAX = 33.0

LOGIT_CLAMP_RANGE = 2.25

# 5 Local Car Directions for Ball Placement [+Fwd, -Fwd, +Rgt, -Rgt, +Up] (NO Down!)
DIR5_LABELS = ("F", "B", "R", "L", "U")
DIR5_FULL_NAMES = ("+Fwd", "-Fwd", "+Rgt", "-Rgt", "+Up")
NUM_DIR5 = 5
NUM_DIR5_BUCKETS = NUM_DIR5

# Alias for backward-compatible references
DIR6_LABELS = DIR5_LABELS
DIR6_FULL_NAMES = DIR5_FULL_NAMES
NUM_DIR6 = NUM_DIR5

# 4 Local Car Horizontal Directions for Initial Velocity [+Fwd, -Fwd, +Rgt, -Rgt]
VEL_DIR4_LABELS = ("F", "B", "R", "L")
NUM_VEL_DIR4 = 4

# Sample Pools (70% Frontier / 30% Memory; all samples spawn car on/near ground with normal boost pads)
POOL_FRONTIER = 0
POOL_MEMORY   = 1
POOL_LABELS   = ("F", "M")
POOL_RATIOS   = (0.70, 0.30)

CAR_GND_Z = 17.0
CAR_REST_Z = CAR_GND_Z
CAR_MAX_SPAWN_Z = 75.0
CAR_MAX_ANG_SPEED = 5.5
CAR_MAX_LIN_SPEED = 2300.0
GROUND_SETTLE_TICKS = 45

# Evaluation & Stage compatibility constants (5 directions, 41 curriculum steps across 5 phases)
NUM_CURRICULUM_STAGES = NUM_DIR5
NUM_CURRICULUM_STEPS = 41
EVAL_SAMPLES_PER_STAGE = 24
EVAL_GRID_BINS = NUM_DIR5
EVAL_SAMPLES_PER_BIN = 24

DIST_TIERS = (
    (160.0,  750.0),   # D0: Near
    (750.0,  2100.0),  # D1: Mid
    (2100.0, 6000.0),  # D2: Far
)
NUM_DIST_TIERS = len(DIST_TIERS)
DIST_TIER_MAX_DECISIONS = (90, 150, 240)

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
