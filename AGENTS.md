# RLRL101 Agent 指南与运行最佳实践 (8-Stage 渐进课程 & 软钳制保险丝 PPO 架构)

### 1. 极简环境配置 (~200MB 预编译安装，无需 CMake / C++ 编译器)
```bash
pip install rocketsim
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas
```
> 若本地设备需要启动 3D 渲染可视化窗口 (`python run.py tuner`)，额外安装 `pip install pyqt5 pyqtgraph PyOpenGL`。

---

### 2. 核心设计规范
1. **187 维车体中心观测空间 (`core/obs_v3.py`, `OBS_V3_DIM = 187`)**：
   - 包含车体局部坐标系下的球位置/速度、自身速度/角速度、墙面/地面/顶棚距离、动作历史，以及**全场 34 个 Boost Pads（6 大 + 28 小）**的车体局部坐标与激活冷却状态 (`[rel_fwd, rel_right, is_active, cooldown_norm] * 34`)。
   - 首层正交初始化针对 136 维 Boost Pads (`0.15x`) 与垂直高度 Z 通道 (`3.0x`) 进行了信噪比均衡，确保冷启动与空中球阶段特征清晰。
2. **动作掩码、保险丝 1 (Projected STE 软钳制) 与回合初 30 帧后退探索 (`core/nn.py`, `core/env.py`)**：
   - **地面喷气强制前驱掩码**：仅在 `on_ground & boost==1 & has_nitro` 时屏蔽 `S` 与 `None`（强制 `W`），空中完全放开俯仰与喷气组合；
   - **保险丝 1（Projected STE 软钳制）**：训练采样 (`deterministic=False`) 时对每个动作头的去均值 Logits 施加 `±2.25` 钳制（最大跨度 `4.5`，保底约 `1.1%` 探索活口），并通过单向投影直通估计器 (`_ProjectedClampSTE`) 保证只有改善回报的探索方向反传梯度；评估路径 (`deterministic=True`) 显式采用 `argmax / temperature=0`（不施加钳制）；
   - **回合初 30 帧后退探索 (`--sticky-prob 0.02`)**：仅在 `env.reset()` 开局有 `2%` 概率触发一次 30 物理帧（0.25s）的随机后退动作 `{S, S+A, S+D}`，并在第 30 帧有 50% 概率触发 `Jump`，随后完全交由模型自主决策。
3. **三维矢量距离奖励与 Boost Pad 回调 (`core/reward.py`, `core/env.py`)**：
   - **三维矢量距离差分**：按 `(|dx_prev|-|dx_now|) + (|dy_prev|-|dy_now|) + (|dz_prev|-|dz_now|)` 分别计算三轴接近量之和（缩放系数 `0.01`），无论静止球或动态球，单局可获距离总分恒由初始三轴距离决定；
   - **Boost Pad 拾取奖励 (`+0.12`)**：严格通过 `arena.set_boost_pickup_callback` 监听（绝不使用 `cur_boost > prev_boost` 判定），每拾取任意一个大小氮气垫奖励 `+0.12`。
4. **8-Stage 渐进课程与高空静止球阶梯 (`core/env.py`, `train/checkpoint.py`)**：
   - **8 个渐进阶段**：`S1(地面球静车静)` -> `S2(地面球动车静)` -> `S3(地面球静车动)` -> `S4(地面球动车动)` -> `S5(空中悬停球-车静)` -> `S6(空中悬停球-车动)` -> `S7(空中动态球-车静)` -> `S8(空中动态球-车动)`；
   - **按阶准入与异步 Best 评估**：Stage $K$ 激活时，前 $K$ 个阶段平分训练采样权重（`Stage 1..6` 要求前 $K$ 格全部 $\ge 70\%$ 才触发后台异步 `Best` 评估，`Stage 7` 豁免第 7 格检查前 6 格，`Stage 8` 要求全部 8 格 $\ge 70\%$）；
   - **高空悬停球 (`!touched` 零重力锁定) 与高度/氮气子状态码 (`air=...`)**：
     - `Z0400_DJ`：高度 `<= 400 uu`（纯 Double Jump 可达），70% 概率初始 Boost `< 33`；
     - `Z0500_B100` .. `Z2000_B100`：球高 `> 400 uu` 覆写为满氮气 `100`，每次专门考核顶部 `[max_z-100, max_z]` 区间 100 个随机位置，命中率 $\ge 70\%$ 则上限 `+100 uu` 直至 `2000 uu`；
     - `Z2000_M+30` -> `M+18` -> `M+06` -> `M+00`：满高 `2000 uu` 达标后，对 `> 500 uu` 高空球按经验 CSV (`AERIAL_HEIGHT_BOOST_REF`) 对应高度氮气量加上裕量（从 `+30` 起步，每次满 70% 递减 `12` 直至 `+0`），全部完成后自动解锁动态高空球 (`S7`, `S8`)。

---

### 3. 统一运行入口与推荐终端命令
- **8-Stage 渐进课程全流程训练 (`python run.py train`)**：
  ```bash
  # 从零启动 8-Stage 自动晋级课程训练 (推荐 4 核并行 + 异步评估)
  python run.py train --mode curriculum --save-prefix cur_v1 --from none --workers 4 --sticky-prob 0.02 --logit-clamp 2.25

  # 断点续训最新权重
  python run.py train --mode curriculum --prefix cur_v1 --from latest --workers 4 --sticky-prob 0.02 --logit-clamp 2.25
  ```
- **多核吞吐量与物理帧率基准测试 (`python run.py bench`)**：
  ```bash
  python run.py bench --workers 1,2,4,8 --steps 2048
  ```
- **CLI 8-Stage 矩阵评估与排行榜 (`python run.py cli`)**：
  ```bash
  python run.py cli --prefix cur_v1 --from best
  ```
- **3D 交互式 GUI Tuner (`python run.py tuner`)**：
  ```bash
  python run.py tuner --prefix cur_v1
  ```
