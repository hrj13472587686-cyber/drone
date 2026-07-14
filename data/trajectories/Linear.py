import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ===================== 超参数 =====================
obs_steps = 8    # 观测窗口帧数
pred_steps = 5   # 预测帧数
dims = 3         # xyz三维坐标
# ===================== 中文乱码修复 =====================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 读取CSV轨迹数据 =====================
df = pd.read_csv("mmaud_mavic3_gt_relative.csv")
t_all = df["timestamp"].values
xyz = df[["x", "y", "z"]].to_numpy()
x_all = xyz[:, 0]
y_all = xyz[:, 1]
z_all = xyz[:, 2]
total_len = len(df)
print(f"数据集总帧数：{total_len}")

# 存储三类误差
rmse_record = []
ade_record = []
fde_record = []
all_window_pred = []

# 合法滑动窗口最大起始下标
max_start = total_len - obs_steps - pred_steps
if max_start <= 0:
    raise ValueError("数据长度不足，无法构建观测+预测窗口，请扩充数据！")

# ===================== 全窗口遍历冒烟测试 =====================
for start in range(max_start):
    obs_end = start + obs_steps
    # 观测集
    train_t = t_all[start:obs_end].reshape(-1, 1)
    train_xyz = xyz[start:obs_end]
    # 预测真值区间
    pred_start_idx = obs_end
    pred_end_idx = obs_end + pred_steps
    true_xyz = xyz[pred_start_idx:pred_end_idx]

    # 线性回归训练
    model = LinearRegression()
    model.fit(train_t, train_xyz)
    pred_t_arr = t_all[pred_start_idx:pred_end_idx].reshape(-1, 1)
    pred_xyz = model.predict(pred_t_arr)

    # ========== 核心误差计算 RMSE / ADE / FDE ==========
    # 1. RMSE
    mse = mean_squared_error(true_xyz, pred_xyz)
    rmse = np.sqrt(mse)
    # 2. ADE 逐帧欧式距离均值
    dists = np.linalg.norm(pred_xyz - true_xyz, axis=1)
    ade = np.mean(dists)
    # 3. FDE 最后一帧终点误差
    fde = np.linalg.norm(pred_xyz[-1] - true_xyz[-1])

    rmse_record.append(rmse)
    ade_record.append(ade)
    fde_record.append(fde)
    all_window_pred.append({"xyz_pred": pred_xyz})

# ===================== 全局误差统计输出 =====================
rmse_arr = np.array(rmse_record)
ade_arr = np.array(ade_record)
fde_arr = np.array(fde_record)
print("="*70)
print(f"【全窗口冒烟测试 obs={obs_steps} pred={pred_steps}】")
print(f"有效窗口总数：{len(rmse_arr)}")
print(f"RMSE 均值:{np.mean(rmse_arr):.4f} | 最大:{np.max(rmse_arr):.4f} | 最小:{np.min(rmse_arr):.4f} | 标准差:{np.std(rmse_arr):.4f}")
print(f"ADE  均值:{np.mean(ade_arr):.4f} | 最大:{np.max(ade_arr):.4f} | 最小:{np.min(ade_arr):.4f} | 标准差:{np.std(ade_arr):.4f}")
print(f"FDE  均值:{np.mean(fde_arr):.4f} | 最大:{np.max(fde_arr):.4f} | 最小:{np.min(fde_arr):.4f} | 标准差:{np.std(fde_arr):.4f}")
print("="*70)

# 写入本地文本，百分百能看到结果
with open("../../results/tables/mmaud_mavic3_gt_relative_summary.csv", "w", encoding="utf-8") as f:
    f.write("="*70 + "\n")
    f.write(f"【全窗口冒烟测试 obs={obs_steps} pred={pred_steps}】\n")
    f.write(f"有效窗口总数：{len(rmse_arr)}\n")
    f.write(f"RMSE 均值:{np.mean(rmse_arr):.4f} | 最大:{np.max(rmse_arr):.4f} | 最小:{np.min(rmse_arr):.4f} | 标准差:{np.std(rmse_arr):.4f}\n")
    f.write(f"ADE  均值:{np.mean(ade_arr):.4f} | 最大:{np.max(ade_arr):.4f} | 最小:{np.min(ade_arr):.4f} | 标准差:{np.std(ade_arr):.4f}\n")
    f.write(f"FDE  均值:{np.mean(fde_arr):.4f} | 最大:{np.max(fde_arr):.4f} | 最小:{np.min(fde_arr):.4f} | 标准差:{np.std(fde_arr):.4f}\n")
    f.write("="*70 + "\n")
print("指标已保存至 mmaud_mavic3_gt_relative_summary.csv", flush=True)

# 全局整体线性外推（用全部数据预测未来5帧）
full_model = LinearRegression()
full_t = t_all.reshape(-1, 1)
full_model.fit(full_t, xyz)
dt_mean = np.mean(np.diff(t_all))
future_t = np.array([t_all[-1] + i*dt_mean for i in range(1, pred_steps+1)]).reshape(-1, 1)
future_xyz = full_model.predict(future_t)
fx = future_xyz[:, 0]
fy = future_xyz[:, 1]
fz = future_xyz[:, 2]

# ===================== 3D绘图：历史橙色、预测蓝色，无渐变 =====================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")
# 真实完整轨迹 橙色
ax.plot(x_all, y_all, z_all, c="#ff5a36", lw=2, label="真实轨迹")
ax.scatter(x_all, y_all, z_all, color="#ff5a36", s=12, alpha=0.6)
# 所有窗口短时预测 蓝细线
for item in all_window_pred:
    xp, yp, zp = item["xyz_pred"].T
    ax.plot(xp, yp, zp, c="#2367eb", lw=0.7, alpha=0.35)

# 全局线性预测
ax.plot(fx, fy, fz, c="#2367eb", linestyle="--", lw=3, label=f"全局线性预测{pred_steps}帧")

ax.set_xlabel("X 坐标")
ax.set_ylabel("Y 坐标")
ax.set_zlabel("Z 高度")
ax.set_title(f"全窗口冒烟测试 obs={obs_steps} pred={pred_steps}", fontsize=15)
ax.legend(loc="upper right")
ax.view_init(elev=26, azim=-62)

# 保存3D轨迹图
save_3d = r"./smoke_3d_obs8_pred5.png"
plt.savefig(save_3d, dpi=300, bbox_inches="tight")
print(f"\3D轨迹图已保存：{save_3d}")
plt.show()
