import numpy as np
import matplotlib.pyplot as plt
import os

# ====================== 3D CV卡尔曼滤波器 ======================
class CV3DKalmanFilter:
    def __init__(self, std_pos, std_vel):
        """
        std_pos: 位置观测噪声标准差
        std_vel: 速度过程噪声标准差(CV模型驱动噪声)
        状态向量 x = [x,y,z,vx,vy,vz]^T
        """
        self.x = np.zeros((6, 1))
        self.P = np.diag(np.ones(6) * 1.0)
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.dt = None

    def init_state(self, pos, vel=None):
        """初始化状态向量"""
        pos = np.array(pos).reshape(3, 1)
        if vel is None:
            vel = np.zeros((3, 1))
        else:
            vel = np.array(vel).reshape(3, 1)
        self.x = np.vstack([pos, vel])

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
        z = np.array(z).reshape(3, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        S += 1e-8 * np.eye(3)  # 防止奇异无法求逆
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_pos(self):
        return self.x[:3, 0]


# -------------------------- 2. 指标计算函数 --------------------------
def compute_pred_only_metrics(gt_win_full, pred_win_full, obs_steps):
    if gt_win_full.ndim != 2 or gt_win_full.shape[-1] != 3:
        raise ValueError(f"gt shape expect [T,3], get {gt_win_full.shape}")
    if pred_win_full.shape != gt_win_full.shape:
        raise ValueError("gt and pred length mismatch!")

    gt_pred = gt_win_full[obs_steps:]
    pred_pred = pred_win_full[obs_steps:]

    if len(gt_pred) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    err = gt_pred - pred_pred
    disp_err = np.linalg.norm(err, axis=1)

    rmse = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    ade = np.mean(disp_err)
    fde = disp_err[-1]

    rmse_x = np.sqrt(np.mean(err[:, 0]**2))
    rmse_y = np.sqrt(np.mean(err[:, 1]**2))
    rmse_z = np.sqrt(np.mean(err[:, 2]**2))

    return rmse, ade, fde, rmse_x, rmse_y, rmse_z

# -------------------------- 3. 滑动窗口评测函数 --------------------------
def sliding_cv_evaluate_windows(seq, time_seq, obs_steps, pred_steps,
                                stride=1, std_pos=0.05, std_vel=0.1):
    N = len(seq)
    win_total = obs_steps + pred_steps
    metric_records = []

    start = 0
    while True:
        end_window = start + win_total
        if end_window > N:
            break

        gt_win = seq[start:end_window].copy()
        pred_win = np.zeros_like(gt_win)

        kf = CV3DKalmanFilter(std_pos, std_vel)
        # 初始化状态
        if start + 1 < N:
            dt0 = time_seq[start+1] - time_seq[start]
            v0 = (seq[start+1] - seq[start]) / dt0
            kf.init_state(pos=seq[start], vel=v0)
        else:
            kf.init_state(pos=seq[start])

        pred_win[0] = seq[start]

        for idx_win in range(1, obs_steps):
            g = start + idx_win
            dt = time_seq[g] - time_seq[g-1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.correct(seq[g])
            pred_win[idx_win] = kf.get_pos()

        current_t = time_seq[start + obs_steps - 1]
        for idx_win in range(obs_steps, win_total):
            g = start + idx_win
            dt = time_seq[g] - current_t
            kf.update_F_Q(dt)
            kf.predict()
            pred_win[idx_win] = kf.get_pos()
            current_t = time_seq[g]

        metrics = compute_pred_only_metrics(gt_win, pred_win, obs_steps)
        metric_records.append(metrics)
        start += stride

    return np.array(metric_records)

# -------------------------- 4. 滑动窗口绘图函数 --------------------------
def sliding_cv_predict_overlap(seq, time_seq, obs_steps, pred_steps,
                               stride=1, std_pos=0.05, std_vel=0.1):
    N = len(seq)
    full_pred = np.zeros_like(seq)
    pred_cnt = np.zeros_like(seq)

    start = 0
    while True:
        end_obs = start + obs_steps
        end_pred = start + obs_steps + pred_steps
        if end_obs >= N:
            break

        kf = CV3DKalmanFilter(std_pos, std_vel)
        if start + 1 < N:
            dt0 = time_seq[start+1] - time_seq[start]
            v0 = (seq[start+1] - seq[start]) / dt0
            kf.init_state(pos=seq[start], vel=v0)
        else:
            kf.init_state(pos=seq[start])

        for i in range(start, end_obs):
            if i == start:
                full_pred[i] += seq[i]
                pred_cnt[i] += 1
                continue
            dt = time_seq[i] - time_seq[i-1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.correct(seq[i])
            full_pred[i] += kf.get_pos()
            pred_cnt[i] += 1

        current_t = time_seq[end_obs - 1]
        pred_end = min(end_pred, N)
        for j in range(end_obs, pred_end):
            dt = time_seq[j] - current_t
            kf.update_F_Q(dt)
            kf.predict()
            full_pred[j] += kf.get_pos()
            pred_cnt[j] += 1
            current_t = time_seq[j]

        start += stride

    mask = pred_cnt > 0
    full_pred[mask] = full_pred[mask] / pred_cnt[mask]
    full_pred[~mask] = seq[~mask]

    return full_pred

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 超参配置
    obs_steps = 8
    pred_steps = 5
    stride = 1
    std_pos = 0.05
    std_vel = 0.1   # 原std_vel重命名，语义更准确

    # 创建输出目录
    os.makedirs("results/tables", exist_ok=True)
    fig_save_path = r"C:\Users\86134\Desktop\drone\results\figures\cv_kf_3d.png"
    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)

    csv_path = "data/mmaud_mavic3_gt_relative.csv"
    csv_data = np.genfromtxt(
        csv_path,
        delimiter=',',
        skip_header=1,
        usecols=(0, 4, 5, 6)
    )

    t_all = csv_data[:, 0]
    gt_all = csv_data[:, 1:4]

    mask = ~np.isnan(gt_all).any(axis=1)
    t_all = t_all[mask]
    gt_all = gt_all[mask]

    print(f"成功加载轨迹数据：共 {len(gt_all)} 个有效点")

    kf_traj = sliding_cv_predict_overlap(gt_all, t_all, obs_steps=obs_steps, pred_steps=pred_steps,
                                          stride=stride, std_pos=std_pos, std_vel=std_vel)
    metrics = sliding_cv_evaluate_windows(gt_all, t_all, obs_steps=obs_steps, pred_steps=pred_steps,
                                          stride=stride, std_pos=std_pos, std_vel=std_vel)

    if metrics.shape[0] > 0:
        mean_rmse, mean_ade, mean_fde, rx, ry, rz = metrics.mean(axis=0)
        print("==== CV-KF 评测结果 ====")
        print(f"窗口数量: {metrics.shape[0]}")
        print(f"RMSE = {mean_rmse:.4f}")
        print(f"ADE  = {mean_ade:.4f}")
        print(f"FDE  = {mean_fde:.4f}")
        summary_data = np.array([
            ["窗口数量", metrics.shape[0]],
            ["RMSE", round(mean_rmse, 4)],
            ["ADE", round(mean_ade, 4)],
            ["FDE", round(mean_fde, 4)],
            ["RMSE_X", round(rx,4)],
            ["RMSE_Y", round(ry,4)],
            ["RMSE_Z", round(rz,4)]
        ])
        np.savetxt("results/tables/cv_kf_metrics_summary.csv", summary_data, delimiter=",", fmt="%s", encoding="utf-8")

    # 3D绘图
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gt_all[:,0], gt_all[:,1], gt_all[:,2], c="#ff3333", s=8, alpha=0.7, label=" Truth")
    ax.plot(kf_traj[:,0], kf_traj[:,1], kf_traj[:,2], c="#0066ff", lw=1.0, label="CV-KF")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("CV-KF  Trajectory")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    plt.show()