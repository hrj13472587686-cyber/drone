import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====================== 3D CV卡尔曼滤波器 ======================
class CV3DKalmanFilter:
    def __init__(self, std_pos, std_vel):
        self.x = np.zeros((6, 1))
        self.P = np.diag(np.ones(6) * 1.0)
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.dt = None

    def update_F_Q(self, dt):
        self.dt = dt
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        q1 = self.std_vel ** 2 * dt4 / 4
        q2 = self.std_vel ** 2 * dt3 / 2
        q3 = self.std_vel ** 2 * dt2
        self.Q = np.array([
            [q1, 0, 0, q2, 0, 0],
            [0, q1, 0, 0, q2, 0],
            [0, 0, q1, 0, 0, q2],
            [q2, 0, 0, q3, 0, 0],
            [0, q2, 0, 0, q3, 0],
            [0, 0, q2, 0, 0, q3]
        ])
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        self.R = np.diag([self.std_pos**2, self.std_pos**2, self.std_pos**2])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, z):
        z = np.array(z).reshape(3,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_pos(self):
        return self.x[:3,0]

# ====================== 评估指标计算 ======================
def compute_metrics(gt, pred):
    disp_err = np.linalg.norm(gt - pred, axis=1)
    rmse = np.sqrt(np.mean(np.sum((gt - pred)**2, axis=1)))
    ade = np.mean(disp_err)
    fde = disp_err[-1]
    rmse_x = np.sqrt(np.mean((gt[:,0]-pred[:,0])**2))
    rmse_y = np.sqrt(np.mean((gt[:,1]-pred[:,1])**2))
    rmse_z = np.sqrt(np.mean((gt[:,2]-pred[:,2])**2))
    return rmse, ade, fde, rmse_x, rmse_y, rmse_z

# ====================== 滑动窗口多步预测函数 ======================
def sliding_cv_predict(seq, time_seq, obs_steps, pred_steps, dims, std_pos=0.05, std_vel=0.1):
    """
    seq: (N,3) 完整xyz真值序列
    time_seq: 对应时间戳
    obs_steps: 观测窗口长度
    pred_steps: 预测步数
    dims=3 三维
    return: 完整滤波+多步预测轨迹
    """
    N = len(seq)
    full_pred = np.zeros_like(seq)
    for start in range(0, N, obs_steps + pred_steps):
        end_obs = start + obs_steps
        end_pred = end_obs + pred_steps
        if end_obs > N:
            break
        # 初始化KF，用前obs帧观测
        kf = CV3DKalmanFilter(std_pos, std_vel)
        kf.x[0,0] = seq[start,0]
        kf.x[1,0] = seq[start,1]
        kf.x[2,0] = seq[start,2]
        full_pred[start] = seq[start]
        # 1. 观测窗口滤波更新
        for i in range(start+1, end_obs):
            dt = time_seq[i] - time_seq[i-1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.correct(seq[i])
            full_pred[i] = kf.get_pos()
        # 2. 无观测，纯多步预测pred_steps帧
        current_t = time_seq[end_obs-1]
        for j in range(end_obs, min(end_pred, N)):
            dt = time_seq[j] - current_t
            kf.update_F_Q(dt)
            kf.predict()
            full_pred[j] = kf.get_pos()
            current_t = time_seq[j]
    # 末尾不足窗口的直接单步滤波补齐
    last_start = (N // (obs_steps+pred_steps)) * (obs_steps+pred_steps)
    if last_start < N:
        kf = CV3DKalmanFilter(std_pos, std_vel)
        kf.x[0,0] = seq[last_start,0]
        kf.x[1,0] = seq[last_start,1]
        kf.x[2,0] = seq[last_start,2]
        full_pred[last_start] = seq[last_start]
        for i in range(last_start+1, N):
            dt = time_seq[i] - time_seq[i-1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.correct(seq[i])
            full_pred[i] = kf.get_pos()
    return full_pred

# ====================== 超参数设置 ======================
obs_steps = 8    # 观测窗口帧数
pred_steps = 5   # 预测帧数
dims = 3         # xyz三维坐标

# ====================== 主程序 ======================
if __name__ == "__main__":
    csv_path = "mmaud_mavic3_gt_relative.csv"
    df = pd.read_csv(csv_path)
    timestamps = df["timestamp"].values
    gt_all = df[["x", "y", "z"]].values

    # 滑动窗口CV-KF 观测8帧+预测5帧
    kf_traj = sliding_cv_predict(gt_all, timestamps, obs_steps, pred_steps, dims)

    # 计算指标
    rmse, ade, fde, rmse_x, rmse_y, rmse_z = compute_metrics(gt_all, kf_traj)
    disp_err = np.linalg.norm(gt_all - kf_traj, axis=1)

    print("=" * 40)
    print(f"滑动窗口CV-KF | 观测{obs_steps}帧 预测{pred_steps}帧")
    print("=" * 40)
    print(f"整体RMSE : {rmse:.6f}")
    print(f"ADE      : {ade:.6f}")
    print(f"FDE      : {fde:.6f}")
    print("-" * 40)
    print(f"RMSE_X   : {rmse_x:.6f}")
    print(f"RMSE_Y   : {rmse_y:.6f}")
    print(f"RMSE_Z   : {rmse_z:.6f}")
    print("=" * 40)

    # 保存指标CSV
    metrics_df = pd.DataFrame({
        "metric": ["RMSE_3D", "ADE", "FDE", "RMSE_X", "RMSE_Y", "RMSE_Z"],
        "value": [rmse, ade, fde, rmse_x, rmse_y, rmse_z],
        "unit": ["m", "m", "m", "m", "m", "m"]
    })
    metrics_df.to_csv(r"C:\Users\86134\Desktop\drone\results\tables\cv_kf_slide_summary.csv", index=False)
    print(f"指标文件保存: cv_kf_summary.csv")

    # ====================== 单独图像保存 ======================
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1 3D轨迹对比图
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(gt_all[:, 0], gt_all[:, 1], gt_all[:, 2],
             c="#ff4444", lw=1.8, label="真值轨迹")
    ax1.plot(kf_traj[:, 0], kf_traj[:, 1], kf_traj[:, 2],
             c="#0088ff", lw=0.7, label=f"CV-KF(obs={obs_steps}, pred={pred_steps})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D轨迹对比")
    ax1.legend()
    ax1.grid(True)
    fig1.savefig(r"C:\Users\86134\Desktop\drone\results\figures\cv_kf_3d.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 2 X轴时序对比
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(timestamps, gt_all[:, 0], "r-", label="真值X")
    ax2.plot(timestamps, kf_traj[:, 0], "b--", label="滤波X")
    ax2.set_title("X轴时序对比")
    ax2.set_xlabel("时间戳")
    ax2.set_ylabel("X坐标")
    ax2.legend()
    ax2.grid(True)
    fig2.savefig("cv_kf_plot_x_time.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 3 Y轴时序对比
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(timestamps, gt_all[:, 1], "r-", label="真值Y")
    ax3.plot(timestamps, kf_traj[:, 1], "b--", label="滤波Y")
    ax3.set_title("Y轴时序对比")
    ax3.set_xlabel("时间戳")
    ax3.set_ylabel("Y坐标")
    ax3.legend()
    ax3.grid(True)
    fig3.savefig("cv_kf_plot_y_time.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 4 Z轴时序对比
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    ax4.plot(timestamps, gt_all[:, 2], "r-", label="真值Z")
    ax4.plot(timestamps, kf_traj[:, 2], "b--", label="滤波Z")
    ax4.set_title("Z轴时序对比")
    ax4.set_xlabel("时间戳")
    ax4.set_ylabel("Z坐标")
    ax4.legend()
    ax4.grid(True)
    fig4.savefig("cv_kf_plot_z_time.png", dpi=150, bbox_inches="tight")
    plt.show()

