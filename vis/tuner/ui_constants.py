"""调参器共享常量与键位映射。"""
from core.constants import MAX_DECISIONS, TICKS_PER_DECISION

STATE_IDLE      = "IDLE"
STATE_RECORDING = "RECORDING"
STATE_DONE      = "DONE"
STATE_PLAYBACK  = "PLAYBACK"
STATE_FREERUN   = "FREERUN"

PHYSICS_HZ    = 120.0
RECORD_LIMIT  = MAX_DECISIONS * TICKS_PER_DECISION    # 720
SPEED_MIN     = 0.5
SPEED_MAX     = 1.0
SPEED_STEP    = 0.1

KEY_NAMES = ['W','A','S','D','Shift','Q','E','Period','Slash']

KEY_MAP = {}
try:
    from pyqtgraph.Qt import QtCore
    KEY_MAP = {
        QtCore.Qt.Key_W: 'W',
        QtCore.Qt.Key_A: 'A',
        QtCore.Qt.Key_S: 'S',
        QtCore.Qt.Key_D: 'D',
        QtCore.Qt.Key_Shift: 'Shift',
        QtCore.Qt.Key_Q: 'Q',
        QtCore.Qt.Key_E: 'E',
        # 兼容按住 Shift (Powerslide) 时按下/松开 '.' (变成 '>') 或 '/' (变成 '?')
        QtCore.Qt.Key_Period: 'Period',
        QtCore.Qt.Key_Greater: 'Period',
        QtCore.Qt.Key_Slash: 'Slash',
        QtCore.Qt.Key_Question: 'Slash',
    }
except (ImportError, ModuleNotFoundError):
    pass
