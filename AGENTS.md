# RLRL101 Agent 指南与运行最佳实践 (纯 PPO 全变体架构)

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
   - 支持从旧版 75 维权重 (`ground_v4_best.pt`) 自动零填充无损升级迁移至 187 维网络 (`adapt_state_dict_input_dim`)。
2. **零先验偏置 + 无强制 W 掩码 + 30 帧后退方向保持探索 (`core/nn.py`, `train/train_ppo.py`)**：
   - 清空 `ActorCritic` 输出层人为动作偏置，全 0 初始化；
   - 移除地面喷气强制按 `W` 掩码，释放 Speed Flip（Boost 同时按 S 取消前翻俯仰）等高阶操作空间；
   - **30 帧方向保持探索机制**：默认概率 `5%` (`--sticky-prob 0.05`)，触发后连续 30 帧（8 个决策步）锁定后退方向动作空间 `{S, S+A, S+D}`，且最后一个决策帧有 50% 概率输出一次 `Jump`（诱发半翻转 Half-Flip / 后翻加速探索）。
3. **纯粹奖励函数 (`core/reward.py`)**：
   - **无触球速度奖励**：触球固定奖励 `+80.0`，通过帧间距离缩减奖励 (`0.01 * Δdist`) 与每帧时间惩罚 (`-0.04/tick`) 自然驱动模型追求最快触球；
   - **微弱喷气惩罚**：每消耗 1 点 Boost 扣 `0.01` 分（满管 100 点最多扣 `1.0` 分），70% 概率初始 Boost `< 33`；
   - **速度相关翻滚触发惩罚**：触发翻滚瞬间按起翻前车速扣分（`>=1800 uu/s` 扣 `0.001`，每降低 `100 uu/s` 多扣 `0.0001`，`0 uu/s` 扣 `0.0028`），抑制起步 0 速盲目原地翻滚。
4. **16-Bin (35 样本/格) Base-36 紧凑评测编码 (`train/checkpoint.py`, `vis/cli_tuner.py`)**：
   - 4 个距离段 × 4 个相对方位角（正前 F、正右 R、正左 L、正后 B）共 16 个分位格，每格 35 个固定种子样本（共 560 题）；
   - 单行输出 16 字符进度码（`0-9` 代表 0~9 次命中，`a-z` 代表 10~35 次命中，全部 35/35 满分输出 `zzzzzzzzzzzzzzzz`）及平均用时。

---

### 3. 统一运行入口与 CLI 工具
- **PPO 多核并行全变体训练 (`python run.py train`)**：
  ```bash
  # 多核并行训练全变体场景 (继承 ground_v4 最佳权重自动扩维至 187d，开启 5% 30帧后退探索)
  python run.py train --mode all --prefix ground_v4 --save-prefix all_v1 --from best --workers 4 --sticky-prob 0.05

  # 从零开始多核训练
  python run.py train --mode all --save-prefix all_v0 --from none --workers 4 --sticky-prob 0.05
  ```
- **动态 Focus Region 靶向训练与 Tuner 实时联动 (`python run.py focus`)**：
  ```bash
  # 开启正后方球专项聚焦 (50% 概率在车正后方生成，50% 全场随机，训练进程与 3D Tuner 自动热加载)
  python run.py focus --preset back --prob 0.5 --dist 300 2500

  # 自定义距离与角度范围并热更新
  python run.py focus --prob 0.5 --dist 500 2000 --angle 135 180

  # 关闭专项聚焦，恢复 100% 全场随机
  python run.py focus --off
  ```
- **CLI 16-Bin 排行榜对比与单模型评估 (`python run.py cli`)**：
  ```bash
  python run.py cli
  python run.py cli --prefix ground_v4
  ```
- **3D 交互式 GUI Tuner（支持 `P` 键切换 REWARD/dist 与 OBS 的一阶差分 Delta 视图，实时显示 Focus Region 扇区）**：
  ```bash
  python run.py tuner --prefix ground_v4
  ```
