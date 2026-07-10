import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ===================== 固定超参数 =====================
obs_steps = 8    # 观测窗口长度
pred_steps = 5   # 预测步数
dims = 3         # x/y/z三维
# ===================== 解决中文乱码 =====================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 读取CSV数据 =====================
df = pd.read_csv("mmaud_mavic3_gt_relative.csv")
t_all = df["timestamp"].values
xyz = df[["x", "y", "z"]].to_numpy()
# 拆分xyz一维数组（提前定义，解决NameError）
x_all = xyz[:, 0]
y_all = xyz[:, 1]
z_all = xyz[:, 2]
total_len = len(df)
print(f"数据集总长度：{total_len} 帧")

# 存储所有窗口误差、预测结果
rmse_record = []
all_window_pred = []

# 合法窗口最大起始索引
max_start = total_len - obs_steps - pred_steps
if max_start <= 0:
    raise ValueError("数据长度不足，无法执行观测+预测窗口测试！")

# ===================== 全窗口遍历冒烟测试 =====================
for start in range(max_start):
    obs_end = start + obs_steps
    # 统一切片，保证样本数量一致
    train_t = t_all[start:obs_end].reshape(-1, 1)
    train_xyz = xyz[start:obs_end]

    # 训练多输出线性模型
    model = LinearRegression()
    model.fit(train_t, train_xyz)

    # 预测区间
    pred_start_idx = obs_end
    pred_end_idx = obs_end + pred_steps
    pred_t_arr = t_all[pred_start_idx:pred_end_idx].reshape(-1, 1)
    pred_xyz = model.predict(pred_t_arr)

    # 计算RMSE
    true_xyz = xyz[pred_start_idx:pred_end_idx]
    mse = mean_squared_error(true_xyz, pred_xyz)
    rmse = np.sqrt(mse)
    rmse_record.append(rmse)

    all_window_pred.append({
        "t_pred": pred_t_arr.flatten(),
        "xyz_pred": pred_xyz,
        "rmse": rmse,
        "win_start": start,
        "win_obs_end": obs_end
    })

# ===================== 冒烟测试误差统计 =====================
rmse_arr = np.array(rmse_record)
print("="*50)
print(f"观测窗口={obs_steps}帧，预测{pred_steps}帧，三维{dims}")
print(f"有效测试窗口总数：{len(rmse_arr)}")
print(f"平均RMSE: {np.mean(rmse_arr):.4f}")
print(f"最大RMSE: {np.max(rmse_arr):.4f}")
print(f"最小RMSE: {np.min(rmse_arr):.4f}")
print(f"RMSE标准差: {np.std(rmse_arr):.4f}")
print("="*50)

# 全局整体线性外推
full_model = LinearRegression()
full_t = t_all.reshape(-1, 1)
full_model.fit(full_t, xyz)
dt_mean = np.mean(np.diff(t_all))
future_t = np.array([t_all[-1] + i*dt_mean for i in range(1, pred_steps+1)]).reshape(-1, 1)
future_xyz = full_model.predict(future_t)
# 拆分全局预测坐标
fx = future_xyz[:, 0]
fy = future_xyz[:, 1]
fz = future_xyz[:, 2]

# ===================== 3D轨迹绘图 =====================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# 真实轨迹（x_all/y_all/z_all已提前定义，无报错）
ax.plot(x_all, y_all, z_all, c="#ff5a36", lw=2, label="真实轨迹")
scatter_real = ax.scatter(x_all, y_all, z_all, color="#ff5a36", s=12, alpha=0.6)

# 所有滑动窗口预测（蓝色细线）
for item in all_window_pred:
    xp, yp, zp = item["xyz_pred"].T
    ax.plot(xp, yp, zp, c="#2367eb", lw=0.7, alpha=0.35)

# 全局线性预测
ax.plot(fx, fy, fz, c="#2367eb", linestyle="--", lw=3, label=f"全局线性预测{pred_steps}帧")
ax.scatter(fx, fy, fz, marker="*", s=40, c="#2367eb")

ax.set_xlabel("X 坐标")
ax.set_ylabel("Y 坐标")
ax.set_zlabel("Z 高度")
ax.set_title(f"全窗口冒烟测试 | 观测{obs_steps}帧 预测{pred_steps}帧 三维轨迹", fontsize=15)
ax.legend(loc="upper right")
cbar = fig.colorbar(scatter_real, shrink=0.52)
cbar.set_label("时间戳 timestamp")
ax.view_init(elev=26, azim=-62)

# 保存3D图
save_3d = r"./smoke_obs8_pred5_3d.png"
plt.savefig(save_3d, dpi=300, bbox_inches="tight")
print(f"\n3D轨迹图保存路径：{save_3d}")
plt.show()

# ===================== RMSE误差曲线 =====================
plt.figure(figsize=(10, 3.8))
plt.plot(rmse_arr, color="#d63031", linewidth=1.0)
plt.xlabel("滑动窗口序号")
plt.ylabel("预测RMSE误差")
plt.title(f"obs={obs_steps} pred={pred_steps} 全窗口误差")
plt.grid(alpha=0.3)
save_err = r"./smoke_rmse_obs8_pred5.png"
plt.savefig(save_err, dpi=300, bbox_inches="tight")
print(f"误差曲线图保存路径：{save_err}")
plt.show()