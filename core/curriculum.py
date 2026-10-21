"""core/curriculum.py —— Freestyle Air-Dribble 4-Phase 渐进课程与五类动态发球分布规范 (Carry / Catch / Bounce / Wall / Ground)。"""
import math
import random
from dataclasses import dataclass
import numpy as np

from .constants import (
    LOW_BOOST_PROB, LOW_BOOST_MAX,
    NUM_CURRICULUM_STAGES, NUM_CURRICULUM_STEPS,
    DIST_TIERS, NUM_DIST_TIERS,
    DIR5_LABELS, NUM_DIR5,
    VEL_DIR4_LABELS, NUM_VEL_DIR4,
    CAR_GND_Z, CAR_MAX_SPAWN_Z, CAR_MAX_LIN_SPEED,
    MAX_DECISIONS,
)

# Safe spawn boundaries inside Soccar arena and inside the 45-degree corner walls
SAFE_SPAWN_X = 3650.0
SAFE_SPAWN_Y = 4450.0
SAFE_CORNER_SUM = 7200.0
SAFE_SPAWN_Z_MAX = 1850.0
SAFE_DJUMP_BALL_Z_MAX = 650.0
SAFE_BALL_Z_MIN = 93.15
SAFE_GND_CAR_Z = CAR_GND_Z
SAFE_MAX_CAR_Z = CAR_MAX_SPAWN_Z


def clamp_safe_arena_xy(x: float, y: float, margin: float = 0.0) -> tuple:
    """Clamps world (x, y) into the safe Soccar octagon away from goals and corner bevels."""
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
    ball_deg: float          # Directional spread angle in degrees
    mem_ball_deg: float
    vel_unlocked: bool
    mem_vel_unlocked: bool
    vel_deg: float
    mem_vel_deg: float
    v_lo: float
    v_hi: float
    mem_v_lo: float
    mem_v_hi: float
    cz_max: float
    mem_cz_max: float
    boost_headroom: float = 47.0
    mem_boost_headroom: float = 47.0
    boost_override: bool = True
    mem_boost_override: bool = True
    bz_max: float = 1650.0
    mem_bz_max: float = 1400.0
    ball_speed_max: float = 950.0
    gate: float = 0.55
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
            f"Z{int(round(self.bz_max)):04d}_"
            f"BS{int(round(self.ball_speed_max)):04d}{done_tag}"
        )

    @property
    def code(self) -> str:
        return format_phase_coord_code(self.step_idx)


def _build_curriculum_steps() -> tuple:
    """Builds 16 progressive steps across 4 Air-Dribble Freestyle Phases:
    - Phase 1 (Steps 0..3) : Close Mid-Air Carry & Gentle Pop Catch (d in [160, 650], bz up to 1100)
    - Phase 2 (Steps 4..7) : Medium-Range Dynamic Catch & Bounce Pop (d in [200, 1300], bz up to 1400)
    - Phase 3 (Steps 8..11): Side-Wall Roll Setup & Fast Aerial Catch (d in [250, 2200], bz up to 1650)
    - Phase 4 (Steps 12..15): Full-Arena Freestyle Showman (d in [180, 3400], bz up to 1800)
    """
    steps = []
    idx = 0

    configs = [
        # Phase 1: Close Carry & Gentle Catch
        (1, 1, 160.0,  420.0,  20.0,  550.0,  950.0,  500.0, 0.45, False),
        (1, 2, 160.0,  500.0,  25.0,  650.0, 1050.0,  600.0, 0.45, False),
        (1, 3, 180.0,  580.0,  30.0,  750.0, 1150.0,  700.0, 0.50, False),
        (1, 4, 180.0,  680.0,  35.0,  850.0, 1250.0,  800.0, 0.50, False),
        # Phase 2: Dynamic Catch & Bounce Pop
        (2, 1, 200.0,  850.0,  35.0,  950.0, 1350.0,  900.0, 0.50, False),
        (2, 2, 200.0, 1000.0,  40.0, 1050.0, 1450.0, 1000.0, 0.50, False),
        (2, 3, 220.0, 1180.0,  40.0, 1150.0, 1500.0, 1100.0, 0.55, False),
        (2, 4, 220.0, 1350.0,  45.0, 1250.0, 1550.0, 1200.0, 0.55, False),
        # Phase 3: Side-Wall Launch & Fast Aerial Catch
        (3, 1, 240.0, 1550.0,  45.0, 1350.0, 1600.0, 1300.0, 0.55, False),
        (3, 2, 240.0, 1800.0,  45.0, 1450.0, 1650.0, 1400.0, 0.55, False),
        (3, 3, 250.0, 2050.0,  45.0, 1550.0, 1700.0, 1500.0, 0.60, False),
        (3, 4, 250.0, 2300.0,  45.0, 1650.0, 1750.0, 1600.0, 0.60, False),
        # Phase 4: Full-Arena Freestyle Showman
        (4, 1, 200.0, 2600.0,  45.0, 1750.0, 1800.0, 1700.0, 0.60, False),
        (4, 2, 200.0, 2900.0,  45.0, 1850.0, 1820.0, 1800.0, 0.60, False),
        (4, 3, 180.0, 3200.0,  45.0, 1950.0, 1850.0, 1900.0, 0.65, False),
        (4, 4, 180.0, 3500.0,  45.0, 2100.0, 1850.0, 2000.0, 0.65, True),
    ]

    for (phase, sub, d_lo, d_hi, b_deg, v_hi, bz_max, bs_max, gate, is_done) in configs:
        mem_d_hi = max(400.0, d_hi * 0.75)
        steps.append(CurriculumStepSpec(
            step_idx=idx,
            phase=phase,
            substep=sub,
            d_lo=d_lo,
            d_hi=d_hi,
            mem_d_lo=160.0,
            mem_d_hi=mem_d_hi,
            ball_deg=b_deg,
            mem_ball_deg=max(15.0, b_deg - 5.0),
            vel_unlocked=True,
            mem_vel_unlocked=True,
            vel_deg=b_deg,
            mem_vel_deg=b_deg,
            v_lo=150.0,
            v_hi=v_hi,
            mem_v_lo=150.0,
            mem_v_hi=max(500.0, v_hi * 0.8),
            cz_max=min(CAR_MAX_SPAWN_Z, bz_max - 120.0),
            mem_cz_max=min(CAR_MAX_SPAWN_Z, bz_max - 150.0),
            boost_headroom=47.0,
            mem_boost_headroom=47.0,
            boost_override=True,
            mem_boost_override=True,
            bz_max=bz_max,
            mem_bz_max=max(900.0, bz_max - 100.0),
            ball_speed_max=bs_max,
            gate=gate,
            has_memory_pool=(idx > 0),
            is_done=is_done,
        ))
        idx += 1

    assert len(steps) == NUM_CURRICULUM_STEPS, f"Expected {NUM_CURRICULUM_STEPS} steps, got {len(steps)}"
    return tuple(steps)


CURRICULUM_STEPS = _build_curriculum_steps()

PHASE_FIRST_STEP_IDX = {
    1: 0,
    2: 4,
    3: 8,
    4: 12,
    5: 15,
}

STAGE_NAMES = (
    "S1: Carry  (空中贴球连击初态)",
    "S2: Catch  (动态高抛接球控球)",
    "S3: Bounce (地面弹跳挑球上天)",
    "S4: Wall   (弧形边墙推球起飞)",
    "S5: Ground (全场自由起球盘带)",
)

STAGE_SHORT_LABELS = ("Carry", "Catch", "Bounce", "Wall", "Ground")


@dataclass
class CurriculumState:
    step_idx: int = 0
    phase: int = 1
    dir5_rates: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)
    pool_rates: tuple = (0.0, 0.0)
    pool_counts: tuple = (0, 0)


def get_curriculum_step_spec(step_idx: int) -> CurriculumStepSpec:
    idx = max(0, min(NUM_CURRICULUM_STEPS - 1, int(step_idx)))
    return CURRICULUM_STEPS[idx]


get_step_spec = get_curriculum_step_spec


def stage_level_to_step_idx(stage_level: int) -> int:
    lv = max(1, min(5, int(stage_level)))
    return PHASE_FIRST_STEP_IDX.get(lv, 0)


def format_phase_coord_code(step_idx: int) -> str:
    sp = get_curriculum_step_spec(step_idx)
    return f"P{sp.phase}.{sp.substep:02d}[S{sp.step_idx + 1:02d}/{NUM_CURRICULUM_STEPS}]"


def format_dir5_code(rates, gate: float = 0.55) -> str:
    parts = []
    for lbl, r in zip(DIR5_LABELS, rates[:NUM_DIR5]):
        pct = int(round(float(r) * 100.0))
        mark = "✓" if float(r) >= gate - 1e-6 else " "
        parts.append(f"{lbl}:{pct:02d}%{mark}")
    return "[" + "|".join(parts) + "]"


def format_pool_code(pool_rates, pool_counts) -> str:
    rf = int(round(float(pool_rates[0]) * 100.0)) if len(pool_rates) > 0 else 0
    rm = int(round(float(pool_rates[1]) * 100.0)) if len(pool_rates) > 1 else 0
    return f"Pool[F:{rf:02d}%|M:{rm:02d}%]"


def format_milestone_tag(step_idx: int, ep_num: int) -> str:
    sp = get_curriculum_step_spec(step_idx)
    return f"P{sp.phase}_{sp.substep:02d}_step{sp.step_idx:02d}_ep{int(ep_num)}"


def classify_local_dir5(rel_fwd: float, rel_rgt: float, rel_up: float) -> int:
    """Classifies relative ball vector into 5 directional buckets (0..4)."""
    af = abs(float(rel_fwd))
    ar = abs(float(rel_rgt))
    au = float(rel_up)
    if au > max(140.0, 0.75 * math.hypot(af, ar)):
        return 4
    if af >= ar:
        return 0 if rel_fwd >= 0.0 else 1
    return 2 if rel_rgt >= 0.0 else 3


def sample_local_dir5_vector(dir_idx: int, max_deg: float = 35.0, rng=random) -> tuple:
    rad = math.radians(max(5.0, min(45.0, float(max_deg))))
    a1 = rng.uniform(-rad, rad)
    a2 = rng.uniform(0.0, rad)
    d = int(dir_idx) % NUM_DIR5
    if d == 0:
        vx, vy, vz = math.cos(a1) * math.cos(a2), math.sin(a1) * math.cos(a2), math.sin(a2)
    elif d == 1:
        vx, vy, vz = -math.cos(a1) * math.cos(a2), math.sin(a1) * math.cos(a2), math.sin(a2)
    elif d == 2:
        vx, vy, vz = math.sin(a1) * math.cos(a2), math.cos(a1) * math.cos(a2), math.sin(a2)
    elif d == 3:
        vx, vy, vz = math.sin(a1) * math.cos(a2), -math.cos(a1) * math.cos(a2), math.sin(a2)
    else:
        uz = math.cos(a1 * 0.6)
        horiz = math.sqrt(max(1e-6, 1.0 - uz * uz))
        az = rng.uniform(-math.pi, math.pi)
        vx, vy, vz = horiz * math.cos(az), horiz * math.sin(az), uz
    n = math.hypot(vx, vy, vz) + 1e-8
    return (vx / n, vy / n, vz / n)


def sample_car_velocity_3d(
    vel_dir4_idx: int,
    vel_deg: float,
    v_lo: float,
    v_hi: float,
    fwd: tuple,
    rgt: tuple,
    rng=random,
    on_ground: bool = True,
) -> tuple:
    spd = rng.uniform(float(v_lo), max(float(v_lo), float(v_hi)))
    ang = math.radians(rng.uniform(-float(vel_deg), float(vel_deg)))
    c, s = math.cos(ang), math.sin(ang)
    vx = spd * (c * fwd[0] + s * rgt[0])
    vy = spd * (c * fwd[1] + s * rgt[1])
    vz = 0.0 if on_ground else rng.uniform(-120.0, 280.0)
    return (float(vx), float(vy), float(vz))


def dist_to_tier(dist: float) -> int:
    d = float(dist)
    if d < DIST_TIERS[0][1]:
        return 0
    if d < DIST_TIERS[1][1]:
        return 1
    return 2


def get_adaptive_max_decisions(dist: float, bz: float = 93.15, base_max: int = MAX_DECISIONS, car_spd: float = 0.0) -> int:
    """In Air-Dribble Freestyle, episodes run for the full long horizon (default 450 decisions = 15s)."""
    return int(max(240, base_max))


def min_boost_for_ball_z(bz: float, cz: float = 17.0, boost_headroom: float = 0.0, boost_override: bool = True) -> float:
    if not boost_override:
        return 0.0
    dz = max(0.0, float(bz) - max(17.0, float(cz)))
    if dz <= 485.0:
        return 0.0
    raw = 25.0 * ((dz - 485.0) / (983.0 - 485.0)) if dz <= 983.0 else (25.0 + 28.0 * ((dz - 983.0) / (1923.0 - 983.0)))
    return float(min(100.0, max(0.0, round(raw + float(boost_headroom)))))


def sample_initial_boost(
    r=random,
    bz: float = 93.15,
    cz: float = 17.0,
    on_ground: bool = True,
    has_second_jump: bool = True,
    boost_headroom: float = 47.0,
    boost_override: bool = True,
) -> float:
    if boost_override:
        return 100.0
    req = min_boost_for_ball_z(bz, cz=cz, boost_headroom=boost_headroom, boost_override=boost_override)
    if r.random() < LOW_BOOST_PROB:
        raw = r.uniform(25.0, LOW_BOOST_MAX)
    else:
        raw = r.uniform(LOW_BOOST_MAX, 100.0)
    return float(max(req, min(100.0, raw)))


def sample_random_prev_action(r=random) -> list:
    return [
        float(r.choice((-1.0, 0.0, 1.0))),
        float(r.choice((-1.0, 0.0, 1.0))),
        float(r.choice((-1.0, 0.0, 1.0))),
        0.0,
        float(r.choice((0.0, 1.0))),
        0.0,
    ]


def euler_to_rot_vectors(yaw: float, pitch: float = 0.0, roll: float = 0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    fwd = (cp * cy, cp * sy, sp)
    rgt = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp)
    up  = (-cr * sp * cy - sr * sy, -cr * sp * sy + sr * cy, cr * cp)
    return fwd, rgt, up
