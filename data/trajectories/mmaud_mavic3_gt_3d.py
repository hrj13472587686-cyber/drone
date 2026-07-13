import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========== 核心：修复中文乱码 ==========
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
# ========================================

# 读取CSV
df = pd.read_csv("mmaud_mavic3_gt_relative.csv")
x = df["x"]
y = df["y"]
z = df["z"]

# 创建3D画布
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# 绘制轨迹与散点
ax.plot(x, y, z, color="#1f77b4", linewidth=2, label="Mavic3 真值轨迹")
scatter = ax.scatter(x, y, z, c=z, cmap="viridis", s=12, alpha=0.7)

# 中文标签（现在不会乱码）
ax.set_xlabel("X 坐标", fontsize=11)
ax.set_ylabel("Y 坐标", fontsize=11)
ax.set_zlabel("Z 高度", fontsize=1)
ax.set_title("MMAUD DJI Mavic3 Leica Relative GT: 3D Trajectory", fontsize=14, pad=20)
ax.legend()

# 颜色条
fig.colorbar(scatter, ax=ax, shrink=0.6, label="高度 Z")
ax.view_init(elev=25, azim=-60)

# 4. 展示/保存
plt.tight_layout()
plt.savefig(r"C:\Users\86134\Desktop\drone\results\figures\mmaud_mavic3_gt_3d.png", dpi=300, bbox_inches="tight")
plt.show()