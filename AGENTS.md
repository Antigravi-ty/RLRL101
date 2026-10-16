# RLRL101 Agent 指南与运行最佳实践 (纯 PPO 架构)

### 1. 极简环境配置 (~200MB 预编译安装，无需 CMake / C++ 编译器)
在 Linux / M1 Mac / Headless 环境下，仅需安装核心轻量依赖即可运行完整强化学习训练与 CLI 评估：
```bash
# 1. 核心物理引擎仿真 (RocketSim 预编译 wheel，秒级完成)
pip install rocketsim

# 2. PyTorch 极简轻量版 (Linux CPU 约 190MB，Mac M1 原生自带 MPS 支持)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. 数据分析基础库
pip install pandas
```
> 若本地设备需要启动 3D 渲染可视化窗口 (`python run.py tuner`)，额外安装 `pip install pyqt5 pyqtgraph PyOpenGL` 即可。

---

### 2. 核心设计规范
1. **首帧输入即刻释放模拟**：
   - `BallReachEnv.reset()` 生成随机场景后保持静止初始状态，无论是模型还是人类玩家，在给出第一个输入时立即开始真正的物理模拟 (`arena.step`)。
2. **120Hz (Tuner) 与 30Hz (Trainer) 帧级奖励 100% 严格对齐 (`ticks_elapsed` + 浮点 `step`)**：
   - `compute_step_reward` 接收 `ticks_elapsed` 与精确帧折算步数 `step = ep_ticks / 4.0`，使 30Hz 训练器与 120Hz 面板算出的 `r_cum` 分毫不差。
3. **6 通道键盘独立输出 (`PPO_ACTION_DIMS = [3, 3, 3, 2, 2, 2]`)**：
   - `ch0_ws` (`0=S, 1=None, 2=W`)：同时控制油门与空中俯仰 (`pitch = -throttle`)；
   - `ch1_ad` (`0=A, 1=None, 2=D`)：同时控制转向与空中偏航；
   - `ch2_qe` (`0=Q, 1=None, 2=E`)：空中横滚；
   - `ch3_jump`, `ch4_boost`, `ch5_slide` (`0/1`)。

---

### 3. 统一运行入口与 CLI 工具
- **CLI 多模型排行榜对比与单模型评估 (`python run.py cli` / `python run.py eval`)**：
  ```bash
  # 自动发现 ./checkpoints/ 下全部模型，同题评测 Ground 分层场景并打印排行榜
  python run.py cli
  python run.py cli --batches 3

  # 评估指定 prefix 模型
  python run.py cli --prefix ground_v4
  ```
- **PPO 模型训练（默认无限循环，按 `Ctrl+C` 安全保存并退出）**：
  ```bash
  # 从指定最佳权重继续训练 Ground 全场地面拿球
  python run.py train --mode ground --prefix ground_v4 --save-prefix ground_v5 --from best --lr-max 1.5e-4 --ent-coef 0.005

  # 从零开始训练 Ground 模式
  python run.py train --mode ground --save-prefix ground_v0 --from none --lr-max 3e-4 --ent-coef 0.005
  ```
- **3D 交互式 GUI Tuner（支持 `R` 同场景重放、`N` 新随机场景、`M` 同场景切换 HUMAN/MODEL）**：
  ```bash
  python run.py tuner --prefix ground_v4
  ```
