# E3 共同 checkpoint 指导诊断：未完成

原始历史归档主要包含日志、cloud H5、模型缓存与评估结果；没有可确认的 REAL_MATCHED 更新前、10、50 的模型 checkpoint。服务器当前不可登录，无法在**同一** `w` 上对 REAL_MATCHED、PUB_RANDOM、PUB_NLI、PRIVATE_ZERO 与 DP 计算 cloud 梯度 norm、cosine、relative error。不能把不同训练轨迹的不同 `w` 直接比较成指导数据差异，也不能把 rank-1 guided 方向的 cosine 当作任意 V 子空间不变量。

本次仅有代码级对应：cloud BP 生成归一化方向；客户端使用 rank-1 cloud 方向与归一化局部 Gaussian 混合。未记录新的参数、梯度或私有记录。组件位置与仅有的历史 pack 用时列在 `COMPONENT_COSTS.csv`；没有受控时延、真实边端设备、GPU 峰值、通信字节实测，因此不支持速度或内存节省结论。
