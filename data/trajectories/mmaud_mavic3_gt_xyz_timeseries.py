import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 解决中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# 读取csv
df = pd.read_csv("../mmaud_mavic3_gt_relative.csv")
x = df["x"]
y = df["y"]
z = df["z"]
t = df["timestamp"]  # 时间戳，用于渐变上色

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# 轨迹线
ax.plot(x, y, z, color="#2277dd", lw=1.8, label="Trajectory")
# 散点按时间上色，越晚越黄
scatter = ax.scatter(x, y, z, c=t, cmap="plasma", s=15, alpha=0.8)

ax.set_xlabel("X 坐标")
ax.set_ylabel("Y 坐标")
ax.set_zlabel("Z 高度")
ax.set_title("Mavic3 MMAUD DJI Mavic3 Leica Relative GT: 3D timestamp Trajectory")
ax.legend()

# 色条代表时间戳
cbar = fig.colorbar(scatter, shrink=0.6)
cbar.set_label("时间戳 timestamp")
ax.view_init(elev=26, azim=-58)

# ========== 保存路径（可自行修改）=========
plt.savefig(r"C:\Users\86134\Desktop\drone\results\figures/mmaud_mavic3_gt_xyz_timeseries.png", dpi=300, bbox_inches="tight")
plt.show()