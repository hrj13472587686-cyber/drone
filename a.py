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


# ====================== 滑动窗口参数 ======================
obs_steps = 8    # 观测帧数
pred_steps = 5   # 预测帧数
dims = 3
stride = 1    # 重叠滑动步长
target_win_id = 0
save_window_metrics_csv = True

def window_list(results_window, pred_steps):
    ade_all = []
    fde_all = []
    horizon_err = [[] for _ in range(pred_steps)]
    for win in results_window:
        gt = win["gt_pred"]
        pred = win["est_pred"]
        L = len(gt)
        if L < pred_steps:
            continue
        disp = np.linalg.norm(gt - pred, axis=1)
        ade_all.append(np.mean(disp))
        fde_all.append(disp[-1])
        for h in range(pred_steps):
            horizon_err[h].append(disp[h])
    mean_ade = np.mean(ade_all)
    mean_fde = np.mean(fde_all)
    horizon_mean = [np.mean(arr) for arr in horizon_err]
    return mean_ade, mean_fde, horizon_mean


def sliding_ca_predict_overlap(seq, time_seq, obs_steps, pred_steps, std_pos=0.05, std_acc=0.1):
    N = len(seq)
    output_wins = []
    stride = 1
    start = 0
    while True:
        end_obs = start + obs_steps
        end_pred = start + obs_steps + pred_steps
        if end_obs >= N:
            break

        kf = CA3DKalmanFilter(std_pos, std_acc)
        kf.x[0, 0] = seq[start, 0]
        kf.x[3, 0] = seq[start, 1]
        kf.x[6, 0] = seq[start, 2]

        # 观测段
        for i in range(start + 1, end_obs):
            dt = time_seq[i] - time_seq[i - 1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.update(seq[i])

        pred_length = min(end_pred, N) - end_obs
        est_piece = np.zeros((pred_length, 3))
        gt_piece = seq[end_obs:min(end_pred, N)].copy()

        current_t = time_seq[end_obs - 1]
        ptr = 0
        for j in range(end_obs, min(end_pred, N)):
            dt = time_seq[j] - current_t
            kf.update_F_Q(dt)
            kf.predict()
            est_piece[ptr] = kf.get_pos()
            ptr += 1
            current_t = time_seq[j]

        output_wins.append({
            "gt_pred": gt_piece,
            "est_pred": est_piece
        })
        start += stride
    # return 必须顶格，在while循环外面！！
    return output_wins


def calc_metrics_from_windows(window_collection, pred_steps):
    ade_all = []
    fde_all = []
    horizon_err = [[] for _ in range(pred_steps)]

    all_gt = []
    all_pred = []

    for win in window_collection:
        gt = win["gt_pred"]
        pred = win["est_pred"]
        L = len(gt)
        # 只保留完整pred_steps长度窗口
        if L < pred_steps:
            continue

        disp = np.linalg.norm(gt - pred, axis=1)
        ade_all.append(np.mean(disp))
        fde_all.append(disp[-1])

        for h in range(pred_steps):
            horizon_err[h].append(disp[h])

        all_gt.append(gt)
        all_pred.append(pred)

    all_gt = np.vstack(all_gt)
    all_pred = np.vstack(all_pred)

    rmse = np.sqrt(np.mean(np.sum((all_gt - all_pred) ** 2, axis=1)))
    rmse_x = np.sqrt(np.mean((all_gt[:, 0] - all_pred[:, 0]) ** 2))
    rmse_y = np.sqrt(np.mean((all_gt[:, 1] - all_pred[:, 1]) ** 2))
    rmse_z = np.sqrt(np.mean((all_gt[:, 2] - all_pred[:, 2]) ** 2))

    mean_ade = np.mean(ade_all)
    mean_fde = np.mean(fde_all)
    horizon_mean = [np.mean(arr) for arr in horizon_err]

    # =========一共7个返回值！顺序不能乱=========
    return rmse, mean_ade, mean_fde, rmse_x, rmse_y, rmse_z, horizon_mean

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "./data/mmaud_mavic3_gt_relative.csv")
    df = pd.read_csv(csv_path)
    timestamps = df["timestamp"].values
    gt_all = df[["x", "y", "z"]].values

    # 接收窗口列表
    out_wins = sliding_ca_predict_overlap(gt_all, timestamps, obs_steps, pred_steps, std_pos=0.05, std_acc=0.01)
    print("=====验证输出类型=====")
    print(type(out_wins))
    print("窗口数量:", len(out_wins))

    rmse, mean_ade, mean_fde, rmse_x, rmse_y, rmse_z, horizon_mean = calc_metrics_from_windows(out_wins, pred_steps)

    print("=" * 42)
    print(f"三维 CA-KF | obs={obs_steps}, pred={pred_steps}")
    print("=" * 42)
    print(f"ADE      : {mean_ade:.6f} m")
    print(f"FDE      : {mean_fde:.6f} m")
    for step, val in enumerate(horizon_mean, start=1):
        print(f"Horizon {step} : {val:.6f} m")
    print("=" * 42)




    # 保存汇总指标csv
    metrics_df = pd.DataFrame({
        "metric": ["RMSE_3D", "ADE", "FDE", "RMSE_X", "RMSE_Y", "RMSE_Z"],
        "value": [rmse, mean_ade, mean_fde, rmse_x, rmse_y, rmse_z],
        "unit": ["m", "m", "m", "m", "m", "m"]
    })
    metrics_df.to_csv(r"C:\Users\86134\Desktop\drone\results\tables\ca_kf_summary.csv", index=False)
    print("指标文件已保存: ca_kf_summary.csv")

    # ====================== 独立分开绘图 ======================
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图1：3D轨迹
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    win0 = out_wins[236]

    # 真值：散点（圆点）
    ax1.scatter(gt_all[:, 0], gt_all[:, 1], gt_all[:, 2], c="#ff3333", s=6, alpha=0.7, label="真值")
    # CA-KF轨迹：实线
    ax1.plot(win0["est_pred"][:, 0], win0["est_pred"][:, 1], win0["est_pred"][:, 2], c="#0066ff", lw=0.9, label="CA_KF匀加速滤波")

    ax1.set_xlabel("X (m)", labelpad=8)
    ax1.set_ylabel("Y (m)", labelpad=8)
    ax1.set_zlabel("Z (m)", labelpad=8)
    ax1.set_title("3D轨迹 CA_KF", fontsize=13)
    ax1.legend();
    ax1.grid(alpha=0.3)

    fig1.savefig(r"C:\Users\86134\Desktop\drone\results\figures\ca_kf_window236.png", dpi=150, bbox_inches="tight")
    plt.show()

