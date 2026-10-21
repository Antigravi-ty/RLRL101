"""nexto —— 独立解耦的 Nexto 多车推理、观测构建与标准开球实时对抗环境包。"""
from .model import NextoAgent, action_8d_to_car_controls, action_8d_to_input_dict
from .obs import NextoObsBuilder
from .env import NextoMatchEnv, parse_nexto_spawn_spec, NEXTO_CS_BINS, NEXTO_BS_BINS, DEFAULT_NEXTO_RANDOM_SPEC


def parse_nexto_teams(val):
    """解析 `--nexto` 传参为 `(n_blue, n_orange)` 元组，例如:
    - `[6, 2]` / `("6", "2")` / `"6 2"` / `"6v2"` -> `(6, 2)`
    - `[3]` -> `(3, 3)`
    - `[]` (仅传 `--nexto` 无数字) -> `(3, 3)`
    - `None` (未启用 `--nexto`) -> `None`
    """
    if val is None or val is False:
        return None
    if val is True:
        return (3, 3)
    if isinstance(val, str):
        s = val.strip().lower().replace("v", " ").replace(",", " ")
        parts = [int(x) for x in s.split() if x.strip()]
        if len(parts) == 0:
            return (3, 3)
        if len(parts) == 1:
            return (max(0, parts[0]), max(0, parts[0]))
        return (max(0, parts[0]), max(0, parts[1]))
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return (3, 3)
        if len(val) == 1:
            return parse_nexto_teams(val[0])
        return (max(0, int(val[0])), max(0, int(val[1])))
    if isinstance(val, int):
        return (max(0, val), max(0, val))
    return (3, 3)


__all__ = [
    "NextoAgent",
    "NextoObsBuilder",
    "NextoMatchEnv",
    "parse_nexto_spawn_spec",
    "NEXTO_CS_BINS",
    "NEXTO_BS_BINS",
    "DEFAULT_NEXTO_RANDOM_SPEC",
    "action_8d_to_car_controls",
    "action_8d_to_input_dict",
    "parse_nexto_teams",
]
