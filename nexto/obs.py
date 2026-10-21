"""nexto/obs.py —— 面向 RocketSim 多车环境的独立 Nexto 批量观测构建器 (支持任意 N vs M 车辆数量)。"""
import numpy as np
from core.obs import BOOST_PAD_LOCATIONS

IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_BOOST = range(5)
POS = slice(5, 8)
LIN_VEL = slice(8, 11)
FW = slice(11, 14)
UP = slice(14, 17)
ANG_VEL = slice(17, 20)
BOOST, DEMO, ON_GROUND, HAS_FLIP = range(20, 24)
ACTIONS = slice(24, 32)


class NextoObsBuilder:
    """将 RocketSim 场上的任意数量车辆、球与 34 个氮气垫直接编码为 Nexto Perceiver 的 (q, kv, m) 批量张量。"""

    _invert = np.array([1.0] * 5 + [-1.0, -1.0, 1.0] * 5 + [1.0] * 4, dtype=np.float32)
    _norm = np.array([1.0] * 5 + [2300.0] * 6 + [1.0] * 6 + [5.5] * 3 + [1.0] * 4, dtype=np.float32)
    _boost_locations = np.asarray(BOOST_PAD_LOCATIONS, dtype=np.float32)
    _boost_types = (0.12 + 0.88 * (_boost_locations[:, 2] > 72.0)).astype(np.float32)

    @classmethod
    def build_batch_obs(
        cls,
        cars: list,
        car_teams: np.ndarray,
        arena,
        pads: list,
        prev_actions: np.ndarray,
    ):
        """构建全部车辆的 (q, kv, m):
        - q:  (n_players, 1, 32)
        - kv: (n_players, n_entities, 24)  其中 n_entities = n_players + 1 + 34
        - m:  (n_players, n_entities)
        """
        n_players = len(cars)
        n_boosts = len(cls._boost_locations)
        n_entities = n_players + 1 + n_boosts

        q = np.zeros((n_players, 1, 32), dtype=np.float32)
        kv = np.zeros((n_players, n_entities, 24), dtype=np.float32)
        m = np.zeros((n_players, n_entities), dtype=np.float32)

        if n_players == 0:
            return q, kv, m

        teams = np.asarray(car_teams, dtype=np.int32)

        # 1. Ball entity at slot `n_players`
        bs = arena.ball.get_state()
        sel_ball = n_players
        kv[:, sel_ball, IS_BALL] = 1.0
        kv[:, sel_ball, POS] = (float(bs.pos.x), float(bs.pos.y), float(bs.pos.z))
        kv[:, sel_ball, LIN_VEL] = (float(bs.vel.x), float(bs.vel.y), float(bs.vel.z))
        kv[:, sel_ball, ANG_VEL] = (float(bs.ang_vel.x), float(bs.ang_vel.y), float(bs.ang_vel.z))

        # 2. 34 Boost pad entities at slots `n_players + 1 :`
        sel_boosts = slice(n_players + 1, None)
        kv[:, sel_boosts, IS_BOOST] = 1.0
        kv[:, sel_boosts, POS] = cls._boost_locations
        kv[:, sel_boosts, BOOST] = cls._boost_types
        pad_active = np.array([1.0 if p.get_state().is_active else 0.0 for p in pads], dtype=np.float32)
        kv[:, sel_boosts, DEMO] = pad_active

        # 3. Player entities at slots `0 : n_players`
        kv[:, :n_players, IS_MATE] = (1 - teams)[None, :]
        kv[:, :n_players, IS_OPP] = teams[None, :]

        for i, car in enumerate(cars):
            cs = car.get_state()
            fw = car.get_forward_dir()
            up = car.get_up_dir()
            hf = cs.has_flip_or_jump() if callable(cs.has_flip_or_jump) else cs.has_flip_or_jump

            kv[i, i, IS_SELF] = 1.0
            kv[:, i, POS] = (float(cs.pos.x), float(cs.pos.y), float(cs.pos.z))
            kv[:, i, LIN_VEL] = (float(cs.vel.x), float(cs.vel.y), float(cs.vel.z))
            kv[:, i, FW] = (float(fw.x), float(fw.y), float(fw.z))
            kv[:, i, UP] = (float(up.x), float(up.y), float(up.z))
            kv[:, i, ANG_VEL] = (float(cs.ang_vel.x), float(cs.ang_vel.y), float(cs.ang_vel.z))
            kv[:, i, BOOST] = float(cs.boost) / 100.0
            kv[:, i, DEMO] = 1.0 if bool(cs.is_demoed) else 0.0
            kv[:, i, ON_GROUND] = 1.0 if bool(cs.is_on_ground) else 0.0
            kv[:, i, HAS_FLIP] = 1.0 if bool(hf) else 0.0

        # 4. Invert coordinates and swap IS_MATE / IS_OPP for Orange team players (team == 1)
        orange_mask = (teams == 1)
        if np.any(orange_mask):
            kv[orange_mask] *= cls._invert
            mate_tmp = kv[orange_mask, :, IS_MATE].copy()
            kv[orange_mask, :, IS_MATE] = kv[orange_mask, :, IS_OPP]
            kv[orange_mask, :, IS_OPP] = mate_tmp

        # 5. Normalize all features
        kv /= cls._norm

        # 6. Copy self entity state into Query before relative coordinate rotation
        for i in range(n_players):
            q[i, 0, :24] = kv[i, i, :]

        # 7. Convert kv positions to car-relative and rotate XY vectors by car yaw
        cls._convert_to_relative(q, kv)

        # 8. Attach previous 8D actions to Query slots 24..32
        if prev_actions is not None and len(prev_actions) == n_players:
            q[:, 0, ACTIONS] = np.asarray(prev_actions, dtype=np.float32)

        return q, kv, m

    @staticmethod
    def _convert_to_relative(q: np.ndarray, kv: np.ndarray):
        kv[:, :, POS] -= q[:, :, POS]
        forward = q[:, :, FW]
        theta = np.arctan2(forward[..., 0], forward[..., 1])[..., None]
        ct = np.cos(theta)
        st = np.sin(theta)
        xs = kv[..., POS.start:ANG_VEL.stop:3].copy()
        ys = kv[..., POS.start + 1:ANG_VEL.stop:3].copy()
        kv[..., POS.start:ANG_VEL.stop:3] = ct * xs - st * ys
        kv[..., POS.start + 1:ANG_VEL.stop:3] = st * xs + ct * ys
