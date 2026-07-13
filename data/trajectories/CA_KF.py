import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====================== 3D CA-KF 恒加速度卡尔曼滤波类 ======================
class CA3DKalmanFilter:
    def __init__(self, std_pos, std_acc):
        """
        std_pos: 位置观测噪声标准差
        std_acc: 加速度过程噪声标准差
        状态维度9: [x,y,z, vx,vy,vz, ax,ay,az].T
        """
        self.x = np.zeros((9, 1))
        self.P = np.diag(np.ones(9) * 1.0)  # 初始协方差
        self.std_pos = std_pos
        self.std_acc = std_acc
        self.dt = None

    def update_F_Q(self, dt):
        """根据当前时间间隔更新状态转移F、过程噪声Q"""
        self.dt = dt
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        dt5 = dt ** 5


        # 单轴CA转移块 3x3
        F_1d = np.array([
            [1, dt, dt2/2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        # 9维F 分块对角 x/y/z三轴独立
        self.F = np.block([
                [F_1d, np.zeros((3, 3)), np.zeros((3, 3))],
                [np.zeros((3, 3)), F_1d, np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), F_1d]
               ])

        # 单轴CA过程噪声Q块 (白加速度噪声)
        q1 = self.std_acc**2 * dt5 / 20
        q2 = self.std_acc**2 * dt4 / 8
        q3 = self.std_acc**2 * dt3 / 6
        q4 = self.std_acc**2 * dt2 / 2
        q5 = self.std_acc**2 * dt3 / 3
        q6 = self.std_acc**2 * dt / 1
        Q_1d = np.array([
            [q1, q2, q3],
            [q2, q5, q4],
            [q3, q4, q6]
        ])
        # 9维Q 分块对角
        self.Q = np.block([
            [Q_1d, np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), Q_1d, np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), Q_1d]
        ])

        # 观测矩阵 H 仅观测x,y,z位置 (3×9)
        self.H = np.zeros((3, 9))
        self.H[0, 0] = 1  # 观测x位置
        self.H[1, 3] = 1  # 观测y位置
        self.H[2, 6] = 1  # 观测z位置
        # 观测噪声 R (3×3)
        self.R = np.diag([self.std_pos**2, self.std_pos**2, self.std_pos**2])

    def predict(self):
        """预测步"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """更新校正步 z=[x,y,z]"""
        z = np.array(z).reshape(3, 1)
        y = z - self.H @ self.x               # 残差
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        self.x = self.x + K @ y
        I = np.eye(9)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        return self.x.copy()

    def get_pos(self):
        """获取三维位置 [x, y, z]"""
        x = self.x[0, 0]
        y = self.x[3, 0]
        z = self.x[6, 0]
        return np.array([x, y, z])

    def get_vel(self):
        """获取三维速度 [vx, vy, vz]"""
        vx = self.x[1, 0]
        vy = self.x[4, 0]
        vz = self.x[7, 0]
        return np.array([vx, vy, vz])

    def get_acc(self):
        """获取三维加速度 [ax, ay, az]"""
        ax = self.x[2, 0]
        ay = self.x[5, 0]
        az = self.x[8, 0]
        return np.array([ax, ay, az])

# ====================== 评估指标 RMSE / ADE / FDE ======================
def compute_metrics(gt, pred):
    disp_err = np.linalg.norm(gt - pred, axis=1)
    rmse = np.sqrt(np.mean(np.sum((gt - pred) ** 2, axis=1)))
    ade = np.mean(disp_err)
    fde = disp_err[-1]
    rmse_x = np.sqrt(np.mean((gt[:,0]-pred[:,0])**2))
    rmse_y = np.sqrt(np.mean((gt[:,1]-pred[:,1])**2))
    rmse_z = np.sqrt(np.mean((gt[:,2]-pred[:,2])**2))
    return rmse, ade, fde, rmse_x, rmse_y, rmse_z

# ====================== 滑动窗口参数 ======================
obs_steps = 8    # 观测帧数
pred_steps = 5   # 预测帧数
dims = 3

def sliding_ca_predict(seq, time_seq, obs_steps, pred_steps, std_pos=0.04, std_acc=0.01):
    N = len(seq)
    full_pred = np.zeros_like(seq)
    window_len = obs_steps + pred_steps
    for start in range(0, N, window_len):
        end_obs = start + obs_steps
        end_pred = end_obs + pred_steps
        if end_obs > N:
            break
        kf = CA3DKalmanFilter(std_pos, std_acc)
        kf.x[0,0] = seq[start,0]
        kf.x[1,0] = seq[start,1]
        kf.x[2,0] = seq[start,2]
        full_pred[start] = seq[start]
        # 观测窗口更新
        for i in range(start+1, end_obs):
            dt = time_seq[i] - time_seq[i-1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.update(seq[i])
            full_pred[i] = kf.get_pos()
        # 纯预测无观测
        current_t = time_seq[end_obs - 1]
        for j in range(end_obs, min(end_pred, N)):
            dt = time_seq[j] - current_t
            kf.update_F_Q(dt)
            kf.predict()
            full_pred[j] = kf.get_pos()
            current_t = time_seq[j]
    # 尾部剩余数据单独滤波
    last_start = (N // window_len) * window_len
    if last_start < N:
        kf = CA3DKalmanFilter(std_pos, std_acc)
        kf.x[0,0] = seq[last_start,0]
        kf.x[1,0] = seq[last_start,1]
        kf.x[2,0] = seq[last_start,2]
        full_pred[last_start] = seq[last_start]
        for i in range(last_start+1, N):
            dt = time_seq[i] - time_seq[i-1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.update(seq[i])
            full_pred[i] = kf.get_pos()
    return full_pred

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    csv_path = "mmaud_mavic3_gt_relative.csv"
    df = pd.read_csv(csv_path)
    timestamps = df["timestamp"].values
    gt_all = df[["x", "y", "z"]].values

    # 运行CA-KF滑动窗口滤波
    kf_traj = sliding_ca_predict(gt_all, timestamps, obs_steps, pred_steps, std_pos=0.05, std_acc=0.01)

    # 计算指标
    rmse, ade, fde, rmse_x, rmse_y, rmse_z = compute_metrics(gt_all, kf_traj)
    disp_err = np.linalg.norm(gt_all - kf_traj, axis=1)

    print("=" * 42)
    print(f"三维 CA-KF 匀加速滤波 | obs={obs_steps}, pred={pred_steps}")
    print("=" * 42)
    print(f"整体RMSE : {rmse:.6f} m")
    print(f"ADE      : {ade:.6f} m")
    print(f"FDE      : {fde:.6f} m")
    print("-" * 42)
    print(f"RMSE_X   : {rmse_x:.6f} m")
    print(f"RMSE_Y   : {rmse_y:.6f} m")
    print(f"RMSE_Z   : {rmse_z:.6f} m")
    print("=" * 42)


    # 保存指标CSV
    metrics_df = pd.DataFrame({
        "metric": ["RMSE_3D", "ADE", "FDE", "RMSE_X", "RMSE_Y", "RMSE_Z"],
        "value": [rmse, ade, fde, rmse_x, rmse_y, rmse_z],
        "unit": ["m", "m", "m", "m", "m", "m"]
    })
    metrics_df.to_csv(r"C:\Users\86134\Desktop\drone\results\tables\ca_kf_metrics.csv", index=False)
    print("指标文件已保存: ca_kf_summary.csv")

    # ====================== 独立分开绘图 ======================
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图1：3D轨迹
    fig1 = plt.figure(figsize=(10,8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(gt_all[:,0], gt_all[:,1] , gt_all[:,2], c="#ff3333", lw=1.8, label="真值")
    ax1.plot(kf_traj[:,0], kf_traj[:,1], kf_traj[:,2], c="#0066ff", lw=0.7, label="CA-KF匀加速滤波")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_title("3D轨迹 CA-KF")
    ax1.legend(); ax1.grid()
    fig1.savefig(r"C:\Users\86134\Desktop\drone\results\figures\ca_kf_3d.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 图2 X时序
    fig2 = plt.figure(figsize=(10,6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(timestamps, gt_all[:,0], "r-", label="真值X")
    ax2.plot(timestamps, kf_traj[:,0], "b--", label="CA滤波X")
    ax2.set_title("X轴时序")
    ax2.legend(); ax2.grid()
    fig2.savefig("ca_kf_x.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 图3 Y时序
    fig3 = plt.figure(figsize=(10,6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(timestamps, gt_all[:,1], "r-", label="真值Y")
    ax3.plot(timestamps, kf_traj[:,1], "b--", label="CA滤波Y")
    ax3.set_title("Y轴时序")
    ax3.legend(); ax3.grid()
    fig3.savefig("ca_kf_y.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 图4 Z时序
    fig4 = plt.figure(figsize=(10,6))
    ax4 = fig4.add_subplot(111)
    ax4.plot(timestamps, gt_all[:,2], "r-", label="真值Z")
    ax4.plot(timestamps, kf_traj[:,2], "b--", label="CA滤波Z")
    ax4.set_title("Z轴时序")
    ax4.legend(); ax4.grid()
    fig4.savefig("ca_kf_z.png", dpi=150, bbox_inches="tight")
    plt.show()

