from dataclasses import dataclass
from .constants import MAX_DECISIONS


@dataclass
class RewardConfig:
    distance_delta_scale:   float = 0.01
    time_penalty_per_tick:  float = 0.04
    touch_bonus:            float = 80.0
    touch_car_speed_scale:  float = 0.05
    timeout_penalty:        float = 10.0
    max_decisions:          int   = MAX_DECISIONS


def compute_step_reward(
    cfg: RewardConfig,
    prev_dist: float,
    new_dist: float,
    car_speed: float = 0.0,
    touched: bool = False,
    timeout: bool = False,
    ticks_elapsed: int = 1,
):
    elapsed_ticks = float(max(1, int(ticks_elapsed)))

    distance_delta = float(prev_dist) - float(new_dist)
    r_dist = cfg.distance_delta_scale * distance_delta

    r_time = -cfg.time_penalty_per_tick * elapsed_ticks

    r_touch = (cfg.touch_bonus + cfg.touch_car_speed_scale * float(car_speed)) if touched else 0.0

    r_timeout = -cfg.timeout_penalty if timeout else 0.0

    r_total = r_dist + r_time + r_touch + r_timeout

    detail = {
        "r_dist":    r_dist,
        "r_time":    r_time,
        "r_touch":   r_touch,
        "r_timeout": r_timeout,
        "r_total":   r_total,
    }
    return r_total, detail
