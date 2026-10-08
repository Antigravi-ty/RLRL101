"""
action_abstraction.py — Rocket League 纯键盘动作抽象层与宏动作控制器

遵循规范：
1. 纯键盘模式：输入严格为离散值 {-1.0, 0.0, 1.0} 及 bool，不允许 float 模拟量。
2. 按键强绑定：
   - W: throttle = +1.0, pitch = -1.0
   - S: throttle = -1.0, pitch = +1.0
   - A: steer = -1.0, yaw = -1.0
   - D: steer = +1.0, yaw = +1.0
   - Q: roll = -1.0
   - E: roll = +1.0
   - Space / Slash: jump = True
   - Period / Space: boost = True
   - Shift: handbrake = True
3. 标定参数：
   - Single jump hold 有效区间：1-24 ticks (0.008s - 0.200s)，其中 1-3 ticks 产生相同最小跳跃高度 (~89.2 uu)，
     4-24 ticks 垂直升力持续累加 (最高 ~231.9 uu)，>=24 ticks 升力饱和。
   - 二段跳 / Dodge 窗口：首次起跳后 1.25s (150 ticks) - 1.45s (174 ticks)，离地悬挂分离延迟为 5 ticks。
   - Dodge 速度加成：约 +500 uu/s 冲量 (上限 2300 uu/s)。
   - Flip cancel 延迟：前翻后 2-4 ticks 按 S 锁止俯仰。
"""

from enum import IntEnum
from typing import List, Tuple, Optional
import math
import numpy as np
import RocketSim as rs


class Intent(IntEnum):
    # --- 地面基础移动与冲刺 (0-11) ---
    IDLE = 0
    FORWARD = 1
    FORWARD_LEFT = 2
    FORWARD_RIGHT = 3
    REVERSE = 4
    REVERSE_LEFT = 5
    REVERSE_RIGHT = 6
    BOOST_FORWARD = 7
    BOOST_LEFT = 8
    BOOST_RIGHT = 9
    DRIFT_LEFT = 10
    DRIFT_RIGHT = 11

    # --- 跳跃与战术 Dodge (12-23) ---
    JUMP_TAP = 12           # 短按起跳 (3 ticks, 快速回地)
    JUMP_HOLD = 13          # 长按起跳 (16 ticks, 高空球)
    DOUBLE_JUMP = 14        # 垂直二段跳
    DODGE_FRONT = 15        # 前翻 (+500 uu/s 冲量)
    DODGE_BACK = 16         # 后翻
    DODGE_LEFT = 17         # 左侧翻
    DODGE_RIGHT = 18        # 右侧翻
    DODGE_FRONT_LEFT = 19   # 左前斜翻
    DODGE_FRONT_RIGHT = 20  # 右前斜翻
    DODGE_BACK_LEFT = 21    # 左后斜翻
    DODGE_BACK_RIGHT = 22   # 右后斜翻

    # --- 高级连招宏动作 (23-27) ---
    FLIP_CANCEL_FRONT = 23  # 前翻 Cancel (锁定车头水平)
    SPEED_FLIP_LEFT = 24    # 左侧 Speed Flip (斜翻 + S Cancel + Roll + 全程喷气)
    SPEED_FLIP_RIGHT = 25   # 右侧 Speed Flip
    WAVEDASH_FRONT = 26     # 前向 Wave Dash (小跳压头拍地加速)
    HALF_FLIP = 27          # Half Flip (后翻 Cancel 180度快速调头)


INTENT_NAMES = {intent: intent.name for intent in Intent}
NUM_INTENTS = len(Intent)


def make_ctrl(
    throttle: float = 0.0,
    steer: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    jump: bool = False,
    boost: bool = False,
    handbrake: bool = False,
) -> rs.CarControls:
    """构建严格符合纯键盘限制的 CarControls"""
    c = rs.CarControls()
    c.throttle = float(np.clip(round(throttle), -1.0, 1.0))
    c.steer = float(np.clip(round(steer), -1.0, 1.0))
    c.pitch = float(np.clip(round(pitch), -1.0, 1.0))
    c.yaw = float(np.clip(round(yaw), -1.0, 1.0))
    c.roll = float(np.clip(round(roll), -1.0, 1.0))
    c.jump = bool(jump)
    c.boost = bool(boost)
    c.handbrake = bool(handbrake)
    return c


def keys_to_ctrl(
    w: bool = False,
    s: bool = False,
    a: bool = False,
    d: bool = False,
    q: bool = False,
    e: bool = False,
    space: bool = False,
    boost: bool = False,
    shift: bool = False,
) -> rs.CarControls:
    """按键映射到车辆控制（严格物理绑定）"""
    thr = 0.0
    pitch = 0.0
    if w and not s:
        thr = 1.0
        pitch = -1.0
    elif s and not w:
        thr = -1.0
        pitch = 1.0

    st = 0.0
    yaw = 0.0
    if d and not a:
        st = 1.0
        yaw = 1.0
    elif a and not d:
        st = -1.0
        yaw = -1.0

    roll = 0.0
    if e and not q:
        roll = 1.0
    elif q and not e:
        roll = -1.0

    return make_ctrl(
        throttle=thr,
        steer=st,
        pitch=pitch,
        yaw=yaw,
        roll=roll,
        jump=space,
        boost=boost,
        handbrake=shift,
    )


class MacroSequencer:
    """
    宏动作时序生成器
    输出逐物理帧 (120Hz ticks) 的按键输入队列
    """

    @staticmethod
    def generate_single_jump(hold_ticks: int = 3) -> List[rs.CarControls]:
        """单跳：按住 jump 持续 hold_ticks，之后松开"""
        seq = []
        for _ in range(hold_ticks):
            seq.append(keys_to_ctrl(space=True, w=True))
        seq.append(keys_to_ctrl(space=False, w=True))
        return seq

    @staticmethod
    def generate_double_jump(hold_ticks: int = 3, wait_ticks: int = 3) -> List[rs.CarControls]:
        """垂直二段跳"""
        seq = []
        for _ in range(hold_ticks):
            seq.append(keys_to_ctrl(space=True, w=True))
        for _ in range(wait_ticks):
            seq.append(keys_to_ctrl(space=False, w=True))
        seq.append(keys_to_ctrl(space=True, w=True))
        return seq

    @staticmethod
    def generate_dodge(
        d_name: str,
        hold_ticks: int = 3,
        wait_ticks: int = 3,
        boost_in_dodge: bool = False,
    ) -> List[rs.CarControls]:
        """
        标准 8 方向 Dodge:
        1. 起跳 hold_ticks (通常 3 ticks)
        2. 松开 jump 等待 wait_ticks (使悬挂完全分离, >=3 ticks)
        3. 对应方向键 + jump 触发 dodge
        4. 空中中立/微调并手刹落地
        """
        seq = []
        # 1. 1st jump
        for _ in range(hold_ticks):
            seq.append(keys_to_ctrl(space=True, w=True, boost=boost_in_dodge))
        # 2. release
        for _ in range(wait_ticks):
            seq.append(keys_to_ctrl(space=False, w=True, boost=boost_in_dodge))

        # 3. dodge trigger
        key_map = {
            "front": dict(w=True, space=True),
            "back": dict(s=True, space=True),
            "left": dict(a=True, space=True),
            "right": dict(d=True, space=True),
            "fl": dict(w=True, a=True, space=True),
            "fr": dict(w=True, d=True, space=True),
            "bl": dict(s=True, a=True, space=True),
            "br": dict(s=True, d=True, space=True),
        }
        trig = key_map.get(d_name, dict(space=True))
        trig_ctrl = keys_to_ctrl(**trig, boost=boost_in_dodge)
        seq.append(trig_ctrl)

        # 4. 保持惯性完成翻滚 (约 40-50 ticks) 并落地手刹
        for _ in range(45):
            seq.append(keys_to_ctrl(shift=True, boost=boost_in_dodge and d_name in ("fl", "fr")))

        return seq

    @staticmethod
    def generate_flip_cancel_front(hold_ticks: int = 3, wait_ticks: int = 3, cancel_delay: int = 3) -> List[rs.CarControls]:
        """前翻 Cancel: 起跳 -> 前翻 -> cancel_delay 帧后持续按 S 锁止俯仰"""
        seq = []
        for _ in range(hold_ticks):
            seq.append(keys_to_ctrl(space=True, w=True))
        for _ in range(wait_ticks):
            seq.append(keys_to_ctrl(space=False, w=True))

        # 前翻触发 (W + jump)
        seq.append(keys_to_ctrl(w=True, space=True))

        # 延迟 N 帧后 Cancel
        for _ in range(cancel_delay):
            seq.append(keys_to_ctrl())

        # S 键反向锁止俯仰 (S -> throttle=-1, pitch=1)
        for _ in range(35):
            seq.append(keys_to_ctrl(s=True, shift=True))

        return seq

    @staticmethod
    def generate_speed_flip(direction: str = "left") -> List[rs.CarControls]:
        """
        Speed Flip (开球必杀技，纯键盘实现):
        1. 地面微转向修正 (微向反方向偏角 ~2-3 ticks)
        2. 跳跃短按 (3 ticks)
        3. 跳跃松开 (3 ticks, 达到安全空高)
        4. 斜前翻触发 (W + A/D + Jump)
        5. 瞬时 Flip Cancel: 立即直拉后方 S (pitch=+1.0) 锁住前倾
        6. 空中 Roll 滚转修正 (E 或 Q) 将四轮对地
        7. 全程喷气 (Boost 不断) + 落地手刹 (Shift) 保留动量
        """
        seq = []
        is_left = direction == "left"

        # 1. 微调反向
        opp_key = dict(d=True) if is_left else dict(a=True)
        for _ in range(2):
            seq.append(keys_to_ctrl(**opp_key, w=True, boost=True))

        # 2. 跳跃起跳 3 ticks
        for _ in range(3):
            seq.append(keys_to_ctrl(space=True, w=True, boost=True))

        # 3. 松开 3 ticks
        for _ in range(3):
            seq.append(keys_to_ctrl(space=False, w=True, boost=True))

        # 4. 斜翻触发
        diag_keys = dict(w=True, a=True, space=True) if is_left else dict(w=True, d=True, space=True)
        seq.append(keys_to_ctrl(**diag_keys, boost=True))

        # 5. Flip Cancel: 拉 S + 空中 Roll
        # 左斜翻需向右滚 (E: roll=+1)，右斜翻需向左滚 (Q: roll=-1)
        roll_key = dict(e=True) if is_left else dict(q=True)
        for _ in range(30):
            seq.append(keys_to_ctrl(s=True, **roll_key, boost=True, shift=True))

        # 6. 落地稳定 (手刹防滑打转)
        for _ in range(15):
            seq.append(keys_to_ctrl(w=True, shift=True, boost=True))

        return seq

    @staticmethod
    def generate_wavedash_front() -> List[rs.CarControls]:
        """
        Wave Dash (前向):
        1. 极小短跳起跳 (2-3 ticks)
        2. 车头上仰 (S 键使 pitch=+1.0) 持续 ~8 ticks，使后轮低于前轮
        3. 滞空等待后轮接近地面 (约 15-20 ticks)
        4. 触地瞬间拍前翻 (W + Jump + Shift)，将翻滚力矩直接转化为地面爆发冲量！
        """
        seq = []
        # 小跳
        for _ in range(3):
            seq.append(keys_to_ctrl(space=True, w=True))
        # 仰头
        for _ in range(8):
            seq.append(keys_to_ctrl(s=True))
        # 滞空等待下落
        for _ in range(12):
            seq.append(keys_to_ctrl())
        # 触地前翻拍地 (Wave Dash 触发)
        seq.append(keys_to_ctrl(w=True, space=True, shift=True))
        # 落地手刹滑行
        for _ in range(15):
            seq.append(keys_to_ctrl(w=True, shift=True))
        return seq

    @staticmethod
    def generate_half_flip() -> List[rs.CarControls]:
        """
        Half Flip:
        1. 倒车蓄速 (S)
        2. 后翻起跳 (S + Jump)
        3. 半翻至 180° (约 15 ticks) 时前推 W (Cancel 后翻)
        4. 滚转 180° (Q/E) 使车身翻正，完成瞬间 180° 调头且保持前冲
        """
        seq = []
        # 起跳
        for _ in range(3):
            seq.append(keys_to_ctrl(space=True, s=True))
        for _ in range(3):
            seq.append(keys_to_ctrl(space=False, s=True))
        # 后翻触发
        seq.append(keys_to_ctrl(space=True, s=True))
        # 半程后翻
        for _ in range(12):
            seq.append(keys_to_ctrl(s=True))
        # W 键 Cancel 后翻 + 空中滚转翻正
        for _ in range(25):
            seq.append(keys_to_ctrl(w=True, e=True, shift=True))
        # 落地向前冲刺
        for _ in range(15):
            seq.append(keys_to_ctrl(w=True, shift=True))
        return seq


class ActionAbstractionLayer:
    """
    动作抽象转移层 (Policy Intent -> Tick-by-Tick Keyboard Controls)
    管理 RL Agent 在每个决策步输出的 Intent，将其转化为 120Hz 物理步的严谨按键输入。
    """

    def __init__(self, ticks_per_decision: int = 4):
        self.ticks_per_decision = ticks_per_decision
        self.active_macro_queue: List[rs.CarControls] = []
        self.current_intent: Intent = Intent.IDLE
        self.is_executing_macro = False

    def reset(self):
        self.active_macro_queue.clear()
        self.current_intent = Intent.IDLE
        self.is_executing_macro = False

    def parse_intent(self, intent_idx: int) -> List[rs.CarControls]:
        """将 Intent 转化为接下来所需执行的按键控制列表"""
        intent = Intent(intent_idx)
        self.current_intent = intent

        # 1. 地面基础移动 (单步持续 ticks_per_decision 帧)
        if intent == Intent.IDLE:
            return [keys_to_ctrl()] * self.ticks_per_decision
        elif intent == Intent.FORWARD:
            return [keys_to_ctrl(w=True)] * self.ticks_per_decision
        elif intent == Intent.FORWARD_LEFT:
            return [keys_to_ctrl(w=True, a=True)] * self.ticks_per_decision
        elif intent == Intent.FORWARD_RIGHT:
            return [keys_to_ctrl(w=True, d=True)] * self.ticks_per_decision
        elif intent == Intent.REVERSE:
            return [keys_to_ctrl(s=True)] * self.ticks_per_decision
        elif intent == Intent.REVERSE_LEFT:
            return [keys_to_ctrl(s=True, a=True)] * self.ticks_per_decision
        elif intent == Intent.REVERSE_RIGHT:
            return [keys_to_ctrl(s=True, d=True)] * self.ticks_per_decision
        elif intent == Intent.BOOST_FORWARD:
            return [keys_to_ctrl(w=True, boost=True)] * self.ticks_per_decision
        elif intent == Intent.BOOST_LEFT:
            return [keys_to_ctrl(w=True, a=True, boost=True)] * self.ticks_per_decision
        elif intent == Intent.BOOST_RIGHT:
            return [keys_to_ctrl(w=True, d=True, boost=True)] * self.ticks_per_decision
        elif intent == Intent.DRIFT_LEFT:
            return [keys_to_ctrl(w=True, a=True, shift=True)] * self.ticks_per_decision
        elif intent == Intent.DRIFT_RIGHT:
            return [keys_to_ctrl(w=True, d=True, shift=True)] * self.ticks_per_decision

        # 2. 跳跃与 Dodge
        elif intent == Intent.JUMP_TAP:
            return MacroSequencer.generate_single_jump(hold_ticks=3)
        elif intent == Intent.JUMP_HOLD:
            return MacroSequencer.generate_single_jump(hold_ticks=16)
        elif intent == Intent.DOUBLE_JUMP:
            return MacroSequencer.generate_double_jump(hold_ticks=3, wait_ticks=3)
        elif intent == Intent.DODGE_FRONT:
            return MacroSequencer.generate_dodge("front")
        elif intent == Intent.DODGE_BACK:
            return MacroSequencer.generate_dodge("back")
        elif intent == Intent.DODGE_LEFT:
            return MacroSequencer.generate_dodge("left")
        elif intent == Intent.DODGE_RIGHT:
            return MacroSequencer.generate_dodge("right")
        elif intent == Intent.DODGE_FRONT_LEFT:
            return MacroSequencer.generate_dodge("fl")
        elif intent == Intent.DODGE_FRONT_RIGHT:
            return MacroSequencer.generate_dodge("fr")
        elif intent == Intent.DODGE_BACK_LEFT:
            return MacroSequencer.generate_dodge("bl")
        elif intent == Intent.DODGE_BACK_RIGHT:
            return MacroSequencer.generate_dodge("br")

        # 3. 高级特技宏动作
        elif intent == Intent.FLIP_CANCEL_FRONT:
            return MacroSequencer.generate_flip_cancel_front()
        elif intent == Intent.SPEED_FLIP_LEFT:
            return MacroSequencer.generate_speed_flip("left")
        elif intent == Intent.SPEED_FLIP_RIGHT:
            return MacroSequencer.generate_speed_flip("right")
        elif intent == Intent.WAVEDASH_FRONT:
            return MacroSequencer.generate_wavedash_front()
        elif intent == Intent.HALF_FLIP:
            return MacroSequencer.generate_half_flip()

        return [keys_to_ctrl()] * self.ticks_per_decision

    def step(self, arena: rs.Arena, car: rs.Car, intent_idx: Optional[int] = None) -> int:
        """
        在环境中执行动作。
        如果当前有正在执行的宏队列，优先消化宏队列；
        否则消耗新的 intent_idx 并推入队列。
        返回实际执行的物理 tick 数量。
        """
        if not self.active_macro_queue:
            if intent_idx is None:
                intent_idx = int(Intent.IDLE)
            controls = self.parse_intent(intent_idx)
            self.active_macro_queue.extend(controls)

        # 按照 TICKS_PER_DECISION 或宏完成情况步进
        executed_ticks = 0
        limit = min(self.ticks_per_decision, len(self.active_macro_queue))
        for _ in range(limit):
            ctrl = self.active_macro_queue.pop(0)
            car.set_controls(ctrl)
            arena.step(1)
            executed_ticks += 1

        self.is_executing_macro = len(self.active_macro_queue) > 0
        return executed_ticks
