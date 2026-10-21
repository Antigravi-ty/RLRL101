"""core/curriculum.py —— 静态球大王 4-Phase 渐进课程阶梯 (车优先地面生成 + 5向半方盒球方位 + 4向水平/下压初速与75uu高度/跳跃标志变化)。"""
import math
import random
from dataclasses import dataclass
import numpy as np

from .constants import (
    LOW_BOOST_PROB, LOW_BOOST_MAX,
    NUM_CURRICULUM_STAGES,
    DIST_TIERS, NUM_DIST_TIERS,
    DIR5_LABELS, NUM_DIR5,
    VEL_DIR4_LABELS, NUM_VEL_DIR4,
    CAR_GND_Z, CAR_MAX_SPAWN_Z, CAR_MAX_LIN_SPEED,
)

# Safe spawn boundaries inside Soccar arena and inside the 45-degree corner walls
SAFE_SPAWN_X = 3650.0
SAFE_SPAWN_Y = 4650.0
SAFE_CORNER_SUM = 7350.0
SAFE_SPAWN_Z_MAX = 1900.0
SAFE_DJUMP_BALL_Z_MAX = 480.0
SAFE_BALL_Z_MIN = 93.15
SAFE_GND_CAR_Z = CAR_GND_Z
SAFE_MAX_CAR_Z = CAR_MAX_SPAWN_Z


def clamp_safe_arena_xy(x: float, y: float, margin: float = 0.0) -> tuple:
    """Clamps world (x, y) into the safe Soccar octagon:
    |x| <= SAFE_SPAWN_X - margin, |y| <= SAFE_SPAWN_Y - margin, and |x| + |y| <= SAFE_CORNER_SUM - margin.
    """
    x_lim = max(100.0, SAFE_SPAWN_X - float(margin))
    y_lim = max(100.0, SAFE_SPAWN_Y - float(margin))
    c_lim = max(200.0, SAFE_CORNER_SUM - float(margin))
    xc = float(np.clip(float(x), -x_lim, x_lim))
    yc = float(np.clip(float(y), -y_lim, y_lim))
    s = abs(xc) + abs(yc)
    if s > c_lim:
        excess = 0.5 * (s - c_lim)
        xc = math.copysign(max(0.0, abs(xc) - excess), xc)
        yc = math.copysign(max(0.0, abs(yc) - excess), yc)
        xc = float(np.clip(xc, -x_lim, x_lim))
        yc = float(np.clip(yc, -y_lim, y_lim))
    return xc, yc


def is_safe_arena_xy(x: float, y: float, margin: float = 0.0) -> bool:
    ax = abs(float(x))
    ay = abs(float(y))
    return (
        ax <= (SAFE_SPAWN_X - margin)
        and ay <= (SAFE_SPAWN_Y - margin)
        and (ax + ay) <= (SAFE_CORNER_SUM - margin)
    )


def is_safe_ball_pos(x: float, y: float, z: float) -> bool:
    return is_safe_arena_xy(x, y, margin=60.0) and (SAFE_BALL_Z_MIN <= float(z) <= SAFE_SPAWN_Z_MAX)


def is_safe_car_pos(x: float, y: float, z: float, on_ground: bool = False) -> bool:
    if not is_safe_arena_xy(x, y, margin=40.0):
        return False
    if on_ground:
        return abs(float(z) - SAFE_GND_CAR_Z) <= 5.0
    return SAFE_GND_CAR_Z <= float(z) <= SAFE_MAX_CAR_Z + 1.0


@dataclass(frozen=True)
class CurriculumStepSpec:
    step_idx: int
    phase: int
    substep: int
    d_lo: float
    d_hi: float
    mem_d_lo: float
    mem_d_hi: float
    ball_deg: float          # 5-direction ball half-angle in degrees: 5° -> 45° (half-cube)
    mem_ball_deg: float
    vel_unlocked: bool       # Whether car initial velocity & cz in [17, 75] & jump flag variations are active
    mem_vel_unlocked: bool
    vel_deg: float           # 4-direction horizontal velocity half-angle in degrees: 5° -> 45° (vertical is [-90°, 0°])
    mem_vel_deg: float
    v_lo: float
    v_hi: float
    mem_v_lo: float
    mem_v_hi: float
    cz_max: float            # 17.0 in Phase 1-2; 75.0 in Phase 3-5
    mem_cz_max: float
    boost_headroom: float = 47.0       # Extra boost headroom above theoretical min for bz > 500 (+47 -> Ceil=100 in P1-P4; 35->23->11->0 in P5)
    mem_boost_headroom: float = 47.0
    boost_override: bool = True        # True in P1..P5.04; False in final P5.05 (pure random 70/30 boost with zero override)
    mem_boost_override: bool = True
    bz_max: float = SAFE_DJUMP_BALL_Z_MAX      # 480.0 in P1-P4 (Double-Jump reachable height envelope); 1900.0 in P5 (High-Aerial + Boost routing)
    mem_bz_max: float = SAFE_DJUMP_BALL_Z_MAX
    gate: float = 0.70
    has_memory_pool: bool = True
    is_done: bool = False

    @property
    def ceil_min_boost(self) -> int:
        if not self.boost_override:
            return 0
        return min(100, 53 + int(round(self.boost_headroom)))

    @property
    def name(self) -> str:
        done_tag = "_DONE" if self.is_done else ""
        return (
            f"P{self.phase}.{self.substep:02d}_"
            f"D{int(round(self.d_lo)):04d}..{int(round(self.d_hi)):04d}_"
            f"B{int(round(self.ball_deg)):02d}_"
            f"BM{self.ceil_min_boost:03d}{done_tag}"
        )

    @property
    def code(self) -> str:
        return format_phase_coord_code(self.step_idx)


def _build_curriculum_steps() -> tuple:
    steps = []
    idx = 0
    deg_ladder = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)

    # ---- Phase 1 (9 steps, idx 0..8): Car static on ground (cz=17, v=0), d in [160, 500], 5-dir ball angle ±5° -> ±45° ----
    for i, b_deg in enumerate(deg_ladder):
        prev_deg = deg_ladder[i - 1] if i > 0 else b_deg
        steps.append(CurriculumStepSpec(
            step_idx=idx,
            phase=1,
            substep=i + 1,
            d_lo=160.0,
            d_hi=500.0,
            mem_d_lo=160.0,
            mem_d_hi=500.0,
            ball_deg=b_deg,
            mem_ball_deg=prev_deg,
            vel_unlocked=False,
            mem_vel_unlocked=False,
            vel_deg=0.0,
            mem_vel_deg=0.0,
            v_lo=0.0,
            v_hi=0.0,
            mem_v_lo=0.0,
            mem_v_hi=0.0,
            cz_max=SAFE_GND_CAR_Z,
            mem_cz_max=SAFE_GND_CAR_Z,
            boost_headroom=47.0,
            mem_boost_headroom=47.0,
            boost_override=True,
            mem_boost_override=True,
            gate=0.70,
            has_memory_pool=(i > 0),
            is_done=False,
        ))
        idx += 1

    # ---- Phase 2 (10 steps, idx 9..18): Full 5-dir box (45°), car static on ground, distance +100 uu shells [500..600] -> [1400..1500] ----
    for i in range(10):
        d_lo = 500.0 + 100.0 * i
        d_hi = d_lo + 100.0
        steps.append(CurriculumStepSpec(
            step_idx=idx,
            phase=2,
            substep=i + 1,
            d_lo=d_lo,
            d_hi=d_hi,
            mem_d_lo=160.0,
            mem_d_hi=d_lo,
            ball_deg=45.0,
            mem_ball_deg=45.0,
            vel_unlocked=False,
            mem_vel_unlocked=False,
            vel_deg=0.0,
            mem_vel_deg=0.0,
            v_lo=0.0,
            v_hi=0.0,
            mem_v_lo=0.0,
            mem_v_hi=0.0,
            cz_max=SAFE_GND_CAR_Z,
            mem_cz_max=SAFE_GND_CAR_Z,
            boost_headroom=47.0,
            mem_boost_headroom=47.0,
            boost_override=True,
            mem_boost_override=True,
            gate=0.70,
            has_memory_pool=True,
            is_done=False,
        ))
        idx += 1

    # ---- Phase 3 (9 steps, idx 19..27): d in [160, 1500], cz in [17, 75] + jump/djump variations, velocity horizontal angle ±5° -> ±45° & vertical [-90°, 0°] ----
    for i, v_deg in enumerate(deg_ladder):
        prev_v_deg = deg_ladder[i - 1] if i > 0 else 0.0
        steps.append(CurriculumStepSpec(
            step_idx=idx,
            phase=3,
            substep=i + 1,
            d_lo=160.0,
            d_hi=1500.0,
            mem_d_lo=160.0,
            mem_d_hi=1500.0,
            ball_deg=45.0,
            mem_ball_deg=45.0,
            vel_unlocked=True,
            mem_vel_unlocked=(i > 0),
            vel_deg=v_deg,
            mem_vel_deg=prev_v_deg,
            v_lo=0.0,
            v_hi=CAR_MAX_LIN_SPEED,
            mem_v_lo=0.0,
            mem_v_hi=CAR_MAX_LIN_SPEED if i > 0 else 0.0,
            cz_max=SAFE_MAX_CAR_Z,
            mem_cz_max=SAFE_MAX_CAR_Z if i > 0 else SAFE_GND_CAR_Z,
            boost_headroom=47.0,
            mem_boost_headroom=47.0,
            boost_override=True,
            mem_boost_override=True,
            gate=0.70,
            has_memory_pool=True,
            is_done=False,
        ))
        idx += 1

    # ---- Phase 4 (8 far shells, idx 28..35): d from 1500 -> 6000 uu with generous aerial boost (Ceil=100, headroom=+47) ----
    far_shells = (
        (1500.0, 1750.0),
        (1750.0, 2000.0),
        (2000.0, 2500.0),
        (2500.0, 3000.0),
        (3000.0, 3500.0),
        (3500.0, 4000.0),
        (4000.0, 5000.0),
        (5000.0, 6000.0),
    )
    for i, (d_lo, d_hi) in enumerate(far_shells):
        steps.append(CurriculumStepSpec(
            step_idx=idx,
            phase=4,
            substep=i + 1,
            d_lo=d_lo,
            d_hi=d_hi,
            mem_d_lo=160.0,
            mem_d_hi=d_lo,
            ball_deg=45.0,
            mem_ball_deg=45.0,
            vel_unlocked=True,
            mem_vel_unlocked=True,
            vel_deg=45.0,
            mem_vel_deg=45.0,
            v_lo=0.0,
            v_hi=CAR_MAX_LIN_SPEED,
            mem_v_lo=0.0,
            mem_v_hi=CAR_MAX_LIN_SPEED,
            cz_max=SAFE_MAX_CAR_Z,
            mem_cz_max=SAFE_MAX_CAR_Z,
            boost_headroom=47.0,
            mem_boost_headroom=47.0,
            boost_override=True,
            mem_boost_override=True,
            gate=0.70,
            has_memory_pool=True,
            is_done=False,
        ))
        idx += 1

    # ---- Phase 5 (5 steps, idx 36..40): Full-Arena Boost Budget Tightening Ladder (Ceil 100 -> 88 -> 76 -> 64 -> 53 -> Pure Random) ----
    # Deducts 12 boost per step on bz > 500 until reaching theoretical limit (53 at ceiling), then lifts override completely.
    headroom_ladder = (
        (35.0, 47.0, True,  True,  False),  # P5.01: Ceiling min boost = 88 (-12), Memory = 100
        (23.0, 35.0, True,  True,  False),  # P5.02: Ceiling min boost = 76 (-24), Memory = 88
        (11.0, 23.0, True,  True,  False),  # P5.03: Ceiling min boost = 64 (-36), Memory = 76
        (0.0,  11.0, True,  True,  False),  # P5.04: Ceiling min boost = 53 (-47, theoretical limit), Memory = 64
        (0.0,  0.0,  False, True,  True),   # P5.05: Pure random boost (no override), Memory = theoretical limit (DONE)
    )
    for i, (hr, mem_hr, b_ovr, mem_b_ovr, done_flag) in enumerate(headroom_ladder):
        steps.append(CurriculumStepSpec(
            step_idx=idx,
            phase=5,
            substep=i + 1,
            d_lo=160.0,
            d_hi=6000.0,
            mem_d_lo=160.0,
            mem_d_hi=6000.0,
            ball_deg=45.0,
            mem_ball_deg=45.0,
            vel_unlocked=True,
            mem_vel_unlocked=True,
            vel_deg=45.0,
            mem_vel_deg=45.0,
            v_lo=0.0,
            v_hi=CAR_MAX_LIN_SPEED,
            mem_v_lo=0.0,
            mem_v_hi=CAR_MAX_LIN_SPEED,
            cz_max=SAFE_MAX_CAR_Z,
            mem_cz_max=SAFE_MAX_CAR_Z,
            boost_headroom=hr,
            mem_boost_headroom=mem_hr,
            boost_override=b_ovr,
            mem_boost_override=mem_b_ovr,
            bz_max=SAFE_SPAWN_Z_MAX,
            mem_bz_max=SAFE_DJUMP_BALL_Z_MAX if i == 0 else SAFE_SPAWN_Z_MAX,
            gate=0.70,
            has_memory_pool=True,
            is_done=done_flag,
        ))
        idx += 1

    return tuple(steps)


CURRICULUM_STEPS = _build_curriculum_steps()
ALL_CURRICULUM_STEPS = CURRICULUM_STEPS
NUM_CURRICULUM_STEPS = len(CURRICULUM_STEPS)

PHASE_FIRST_STEP_IDX = {
    1: 0,
    2: 9,
    3: 19,
    4: 28,
    5: 36,
    6: NUM_CURRICULUM_STEPS - 1,
}


def get_curriculum_step_spec(step_idx: int) -> CurriculumStepSpec:
    idx = max(0, min(NUM_CURRICULUM_STEPS - 1, int(step_idx)))
    return CURRICULUM_STEPS[idx]


get_step_spec = get_curriculum_step_spec


def stage_level_to_step_idx(stage_level: int) -> int:
    val = int(stage_level)
    if 1 <= val <= 5 and val in PHASE_FIRST_STEP_IDX:
        return PHASE_FIRST_STEP_IDX[val]
    return max(0, min(NUM_CURRICULUM_STEPS - 1, val))


def classify_local_dir5(rel_fwd: float, rel_rgt: float, rel_up: float) -> int:
    """Classifies a ball position vector in the car's local body frame (fwd, rgt, up)
    into one of the 5 upper half-space direction buckets (NO Down!):
      0: +Fwd (F), 1: -Fwd (B), 2: +Rgt (R), 3: -Rgt (L), 4: +Up (U)
    """
    ax = abs(float(rel_fwd))
    ay = abs(float(rel_rgt))
    uz = max(0.0, float(rel_up))
    if uz >= ax and uz >= ay:
        return 4
    if ax >= ay:
        return 0 if rel_fwd >= 0.0 else 1
    return 2 if rel_rgt >= 0.0 else 3


classify_local_dir6 = classify_local_dir5


def sample_local_dir5_vector(dir_idx: int, angle_deg: float, rng=random) -> tuple:
    """Samples a unit direction vector (ux, uy, uz) in the car's local frame (fwd, rgt, up)
    within the pyramid of direction bucket `dir_idx in 0..4` with half-angle `angle_deg in [5°, 45°]`.
    When angle_deg == 45°, the 5 pyramids seamlessly tile the entire upper half-cube (z >= 0) above the car.
    For lateral buckets (F/B/R/L), uz is sampled in [-tan(a), +tan(a)] so that ~50% of samples clamp
    onto the ground plane (bz = 93.15) and ~50% span the aerial wedge up to +angle_deg.
    """
    deg = float(np.clip(angle_deg, 1.0, 45.0))
    t_max = math.tan(math.radians(deg))
    u = rng.uniform(-t_max, t_max)
    v = rng.uniform(-t_max, t_max)
    d = int(dir_idx) % NUM_DIR5
    if d == 0:    # +Fwd (F)
        vec = (1.0, u, v)
    elif d == 1:  # -Fwd (B)
        vec = (-1.0, u, v)
    elif d == 2:  # +Rgt (R)
        vec = (u, 1.0, v)
    elif d == 3:  # -Rgt (L)
        vec = (u, -1.0, v)
    else:         # +Up (U)
        vec = (u, v, 1.0)
    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    inv = 1.0 / max(norm, 1e-8)
    return (vec[0] * inv, vec[1] * inv, vec[2] * inv)


def sample_car_velocity_3d(
    vel_dir4_idx: int,
    vel_deg: float,
    v_lo: float,
    v_hi: float,
    fwd: tuple,
    rgt: tuple,
    rng=random,
    on_ground: bool = False,
) -> tuple:
    """Samples car initial 3D velocity in world coordinates:
    - Horizontal direction: one of 4 car-local axes (+Fwd, -Fwd, +Rgt, -Rgt) with horizontal offset in [-vel_deg, +vel_deg].
    - Vertical pitch: 0° when on_ground=True; strictly horizontal or downward in [-90°, 0°] when in air.
    - Speed magnitude: uniform in [v_lo, v_hi].
    """
    if v_hi <= 1e-3:
        return (0.0, 0.0, 0.0)
    spd = rng.uniform(v_lo, max(v_lo, v_hi))
    d4 = int(vel_dir4_idx) % NUM_VEL_DIR4
    base_deg = (0.0, 180.0, 90.0, -90.0)[d4]
    h_off_deg = rng.uniform(-float(vel_deg), float(vel_deg))
    h_rad = math.radians(base_deg + h_off_deg)

    if on_ground:
        v_h = spd
        vz = 0.0
    else:
        # Vertical angle strictly downward or horizontal: pitch_down in [0°, 90°] -> vz = -spd * sin(pitch_down) <= 0
        down_rad = math.radians(rng.uniform(0.0, 90.0))
        v_h = spd * math.cos(down_rad)
        vz = -spd * math.sin(down_rad)

    v_fwd = v_h * math.cos(h_rad)
    v_rgt = v_h * math.sin(h_rad)
    vx = v_fwd * fwd[0] + v_rgt * rgt[0]
    vy = v_fwd * fwd[1] + v_rgt * rgt[1]
    return (float(vx), float(vy), float(vz))


def min_boost_for_ball_z(
    bz: float,
    cz: float = SAFE_GND_CAR_Z,
    on_ground: bool = True,
    has_second_jump: bool = True,
    boost_headroom: float = 0.0,
    boost_override: bool = True,
) -> float:
    """Deterministically computes the minimum initial boost required to reach a ball at height `bz`
    via vertical double-jump + continuous boost + ballistic coast under gravity, calibrated against
    RocketSim physics and reference trajectory `1KWYnfez` (g=650, a_boost=1058.33, burn=33.33/s,
    frame-132 post-double-jump state z0=cz+279.4, vz0=406.0 -> 0-boost coast apex = cz+406.2 uu).
    - Below bz <= 500.0 (when on_ground): no override is applied (returns 0.0).
    - Above bz > 500.0: returns `ceil(1.15 * b_det + 4.0) + boost_headroom` capped at 100.0.
      In Phases 1-4, `boost_headroom = 47.0` sets ceiling (1940 uu) to 100 boost and 1500 uu to 88 boost (-12 delta).
      In Phase 5, `boost_headroom` steps down by 12 (`35 -> 23 -> 11 -> 0`) to the theoretical limit (53 at ceiling),
      before `boost_override=False` lifts the override completely.
    """
    if not boost_override:
        return 0.0
    if on_ground and float(bz) <= 500.0:
        return 0.0

    g = 650.0
    a_boost = 1058.3333
    a_net = a_boost - g          # 408.3333 uu/s^2 net upward acceleration while boosting vertically
    burn_rate = 33.3333          # boost units per second
    touch_reach_z = 140.0        # vertical center-to-center reach margin at apex

    if on_ground:
        vz0 = 406.0
        z_apex_0 = float(cz) + 279.4 + (vz0 * vz0) / (2.0 * g)
    elif has_second_jump:
        vz0 = 292.0
        z_apex_0 = float(cz) + (vz0 * vz0) / (2.0 * g)
    else:
        vz0 = 0.0
        z_apex_0 = float(cz)

    z_req = float(bz) - touch_reach_z
    dz_apex = z_req - z_apex_0
    if dz_apex <= 0.0:
        return 0.0

    # Exact ballistic-coast + vertical-boost quadratic:
    # dz_apex = (a_boost / g) * (vz0 * t_b + 0.5 * a_net * t_b^2)
    c_rhs = dz_apex * (g / a_boost)
    disc = max(0.0, vz0 * vz0 + 2.0 * a_net * c_rhs)
    t_b = (-vz0 + math.sqrt(disc)) / a_net
    b_det = burn_rate * t_b

    # Add 15% + 4.0 boost margin for pitch rotation & horizontal travel component, then ceil to integer + curriculum headroom
    b_theory = math.ceil(1.15 * b_det + 4.0)
    b_min = b_theory + max(0, int(round(float(boost_headroom))))
    return float(min(100, max(0, b_min)))


def sample_initial_boost(
    r=random,
    low_prob=LOW_BOOST_PROB,
    low_max=LOW_BOOST_MAX,
    bz: float = None,
    cz: float = SAFE_GND_CAR_Z,
    on_ground: bool = True,
    has_second_jump: bool = True,
    boost_headroom: float = 0.0,
    boost_override: bool = True,
) -> float:
    """Global initial boost rule across all phases:
    Base distribution is 70% probability < 33, 30% probability 33..100, clamped below by
    the curriculum's height-based minimum boost when `bz > 500.0` and `boost_override=True`.
    """
    if r.random() < low_prob:
        b = float(r.uniform(0.0, low_max))
    else:
        b = float(r.uniform(low_max, 100.0))
    if bz is not None and boost_override:
        b = max(
            b,
            min_boost_for_ball_z(
                bz,
                cz=cz,
                on_ground=on_ground,
                has_second_jump=has_second_jump,
                boost_headroom=boost_headroom,
                boost_override=boost_override,
            ),
        )
    return float(min(100.0, b))


def sample_random_prev_action(r=random) -> np.ndarray:
    """Samples a completely random initial previous action vector:
    [thr_pitch in {-1,0,1}, steer_yaw in {-1,0,1}, roll in {-1,0,1}, jump in {0,1}, boost in {0,1}, slide in {0,1}].
    """
    return np.array([
        float(r.choice((-1.0, 0.0, 1.0))),
        float(r.choice((-1.0, 0.0, 1.0))),
        float(r.choice((-1.0, 0.0, 1.0))),
        float(r.choice((0.0, 1.0))),
        float(r.choice((0.0, 1.0))),
        float(r.choice((0.0, 1.0))),
    ], dtype=np.float32)


def _rate_to_digit(rate: float) -> str:
    if rate <= 0.0:
        return "0"
    d = int(math.floor(rate * 10.0 + 1e-9))
    return str(max(0, min(9, d)))


def format_phase_coord_code(step_idx: int) -> str:
    """Formats Segment 1 of status log:
    e.g. P1[D0160..0500 | B05° | V0000(00°) | Z17 | BM100]
         P2[D0600..0700 | B45° | V0000(00°) | Z17 | BM100]
         P3[D0160..1500 | B45° | V2300(15°/D90°) | Z75 | BM100]
         P4[D2000..2500 | B45° | V2300(45°/D90°) | Z75 | BM100]
         P5[D0160..6000 | B45° | V2300(45°/D90°) | Z75 | BM088]
         P5[D0160..6000 | B45° | V2300(45°/D90°) | Z75 | BM000](DONE)
    """
    spec = get_curriculum_step_spec(step_idx)
    d_str = f"D{int(round(spec.d_lo)):04d}..{int(round(spec.d_hi)):04d}"
    b_str = f"B{int(round(spec.ball_deg)):02d}°"
    if spec.vel_unlocked:
        v_str = f"V{int(round(spec.v_hi)):04d}({int(round(spec.vel_deg)):02d}°/D90°)"
    else:
        v_str = "V0000(00°)"
    z_str = f"Z{int(round(spec.cz_max)):02d}"
    bm_str = f"BM{spec.ceil_min_boost:03d}"
    done_suffix = "(DONE)" if spec.is_done else ""
    return f"P{spec.phase}[{d_str} | {b_str} | {v_str} | {z_str} | {bm_str}]{done_suffix}"


def format_dir5_code(dir5_rates, gate: float = 0.70) -> str:
    """Formats Segment 2 of status log:
    e.g. Dir5[87787](U:68%/70%)
    """
    rates = [float(dir5_rates[i]) if i < len(dir5_rates) else 0.0 for i in range(NUM_DIR5)]
    digits = "".join(_rate_to_digit(r) for r in rates)
    min_idx = int(np.argmin(rates))
    min_lbl = DIR5_LABELS[min_idx]
    min_val = rates[min_idx]
    return f"Dir5[{digits}]({min_lbl}:{min_val * 100.0:02.0f}%/{gate * 100.0:02.0f}%)"


format_dir6_code = format_dir5_code


def format_pool_code(pool_rates, pool_counts=None) -> str:
    """Formats Segment 3 of status log:
    e.g. Pool[F:72% M:84%]
    """
    f_r = float(pool_rates[0]) if len(pool_rates) > 0 else 0.0
    m_r = float(pool_rates[1]) if len(pool_rates) > 1 else 0.0
    if pool_counts is not None and len(pool_counts) > 1 and int(pool_counts[1]) == 0:
        m_str = f"{f_r * 100.0:02.0f}%"
    else:
        m_str = f"{m_r * 100.0:02.0f}%"
    return f"Pool[F:{f_r * 100.0:02.0f}% M:{m_str}]"


def format_milestone_tag(step_idx: int, ep_num: int) -> str:
    spec = get_curriculum_step_spec(step_idx)
    return (
        f"P{spec.phase}_"
        f"D{int(round(spec.d_hi)):04d}_"
        f"B{int(round(spec.ball_deg)):02d}_"
        f"V{int(round(spec.v_hi)):04d}_"
        f"VD{int(round(spec.vel_deg)):02d}_"
        f"Z{int(round(spec.cz_max)):02d}_"
        f"BM{spec.ceil_min_boost:03d}_"
        f"ep{int(ep_num)}"
    )


def dist_to_tier(dist: float) -> int:
    for idx, (d_lo, d_hi) in enumerate(DIST_TIERS):
        if dist <= d_hi or idx == NUM_DIST_TIERS - 1:
            return idx
    return NUM_DIST_TIERS - 1


def get_adaptive_max_decisions(dist: float, bz: float = 93.15, default_max: int = 240, car_spd: float = 0.0) -> int:
    bonus = 30 if car_spd > 900.0 else 0
    if dist <= 650.0 and bz <= 600.0:
        return min(default_max, 90 + bonus)
    if dist <= 1600.0:
        return min(default_max, 135 + bonus)
    if dist <= 3000.0:
        return min(default_max, 180 + bonus)
    return min(default_max, 240)


@dataclass
class CurriculumState:
    step_idx: int = 0
    phase: int = 1
    dir5_rates: tuple = (0.0,) * NUM_DIR5
    pool_rates: tuple = (0.0, 0.0)
    pool_counts: tuple = (0, 0)

    @property
    def current_spec(self) -> CurriculumStepSpec:
        return get_curriculum_step_spec(self.step_idx)

    def code_str(self) -> str:
        spec = self.current_spec
        c1 = format_phase_coord_code(self.step_idx)
        c2 = format_dir5_code(self.dir5_rates, gate=spec.gate)
        c3 = format_pool_code(self.pool_rates, self.pool_counts)
        return f"{c1} {c2} {c3}"


STAGE_NAMES = (
    "Dir0:+Fwd(前向扇区)",
    "Dir1:-Fwd(后向扇区)",
    "Dir2:+Rgt(右向扇区)",
    "Dir3:-Rgt(左向扇区)",
    "Dir4:+Up (上向高空)",
)

STAGE_SHORT_LABELS = ("F:+Fwd", "B:-Fwd", "R:+Rgt", "L:-Rgt", "U:+Up")


def euler_to_rot_vectors(yaw: float, pitch: float = 0.0, roll: float = 0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    fwd = (cp * cy, cp * sy, sp)
    rgt = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp)
    up  = (-cr * sp * cy - sr * sy, -cr * sp * sy + sr * cy, cr * cp)
    return fwd, rgt, up
