import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 1. CA卡尔曼滤波器 --------------------------
class CA3DKalmanFilter:
    def __init__(self, std_pos, std_acc):
        self.x = np.zeros((9, 1))
        self.P = np.diag(np.ones(9) * 1.0)
        self.std_pos = std_pos
        self.std_acc = std_acc
        self.dt = None

    def init_state(self, pos, vel=None, acc=None):
        self.x[0, 0] = pos[0]
        self.x[3, 0] = pos[1]
        self.x[6, 0] = pos[2]
        if vel is not None:
            self.x[1, 0] = vel[0]
            self.x[4, 0] = vel[1]
            self.x[7, 0] = vel[2]
        if acc is not None:
            self.x[2, 0] = acc[0]
            self.x[5, 0] = acc[1]
            self.x[8, 0] = acc[2]

    def update_F_Q(self, dt):
        self.dt = dt
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        dt5 = dt ** 5

        F1d = np.array([
            [1, dt, dt2/2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        self.F = np.block([
            [F1d, np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), F1d, np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), F1d]
        ])

        sa2 = self.std_acc ** 2
        Q1d = sa2 * np.array([
            [dt5/20, dt4/8,  dt3/6],
            [dt4/8,  dt3/3,  dt2/2],
            [dt3/6,  dt2/2,  dt]
        ])
        self.Q = np.block([
            [Q1d, np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), Q1d, np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), Q1d]
        ])

        self.H = np.zeros((3, 9))
        self.H[0,0] = 1.0
        self.H[1,3] = 1.0
        self.H[2,6] = 1.0
        self.R = np.diag([self.std_pos**2]*3)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array(z).reshape(3,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(9)
        self.P = (I - K@self.H) @ self.P @ (I - K@self.H).T + K @ self.R @ K.T

    def get_pos(self):
        return np.array([self.x[0,0], self.x[3,0], self.x[6,0]])

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

    rmse_x = np.sqrt(np.mean(err[:,0]**2))
    rmse_y = np.sqrt(np.mean(err[:,1]**2))
    rmse_z = np.sqrt(np.mean(err[:,2]**2))

    return rmse, ade, fde, rmse_x, rmse_y, rmse_z

# -------------------------- 3. 滑动窗口评测函数 --------------------------
def sliding_ca_evaluate_windows(seq, time_seq, obs_steps, pred_steps,
                                stride=1, std_pos=0.05, std_acc=0.1):
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

        kf = CA3DKalmanFilter(std_pos, std_acc)
        kf.init_state(pos=seq[start])
        if start + 1 < N:
            dt0 = time_seq[start+1] - time_seq[start]
            v0 = (seq[start+1] - seq[start]) / dt0
            kf.init_state(pos=seq[start], vel=v0)

        pred_win[0] = seq[start]

        for idx_win in range(1, obs_steps):
            g = start + idx_win
            dt = time_seq[g] - time_seq[g-1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.update(seq[g])
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
def sliding_ca_predict_overlap(seq, time_seq, obs_steps, pred_steps,
                               stride=1, std_pos=0.05, std_acc=0.1):
    N = len(seq)
    full_pred = np.zeros_like(seq)
    pred_cnt = np.zeros_like(seq)

    start = 0
    while True:
        end_obs = start + obs_steps
        end_pred = start + obs_steps + pred_steps
        if end_obs >= N:
            break

        kf = CA3DKalmanFilter(std_pos, std_acc)
        kf.init_state(pos=seq[start])
        if start + 1 < N:
            dt0 = time_seq[start+1] - time_seq[start]
            v0 = (seq[start+1] - seq[start]) / dt0
            kf.init_state(pos=seq[start], vel=v0)

        for i in range(start, end_obs):
            if i == start:
                full_pred[i] += seq[i]
                pred_cnt[i] += 1
                continue
            dt = time_seq[i] - time_seq[i-1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.update(seq[i])
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

# -------------------------- 主函数（直接读取CSV真值轨迹） --------------------------
if __name__ == "__main__":
    # 超参配置
    obs_steps = 8
    pred_steps = 5
    stride = 1
    std_pos = 0.05
    std_acc = 0.1

    # ========== 核心修改：直接读取CSV真值轨迹 ==========
    csv_path = "data/mmaud_mavic3_gt_relative.csv"
    csv_data = np.genfromtxt(
        csv_path,
        delimiter=',',
        skip_header=1,  # 跳过第一行表头
        usecols=(0, 4, 5, 6)  # 只读取需要的列：时间、x、y、z
    )

    # 提取时间轴和真值轨迹
    t_all = csv_data[:, 0]  # 时间戳（秒级，直接用原始时间）
    gt_all = csv_data[:, 1:4]  # x/y/z 3D轨迹

    # 去除含空值的行（如果有）
    mask = ~np.isnan(gt_all).any(axis=1)
    t_all = t_all[mask]
    gt_all = gt_all[mask]

    print(f"成功加载轨迹数据：共 {len(gt_all)} 个有效点")

    # 生成整条预测轨迹
    kf_traj = sliding_ca_predict_overlap(gt_all, t_all, obs_steps=obs_steps, pred_steps=pred_steps, stride=stride, std_pos=std_pos, std_acc=std_acc)
    # 计算标准ADE/FDE/RMSE指标
    metrics = sliding_ca_evaluate_windows(gt_all, t_all, obs_steps=obs_steps, pred_steps=pred_steps, stride=stride, std_pos=std_pos, std_acc=std_acc)

    # 输出指标
    if metrics.shape[0] > 0:
        mean_rmse, mean_ade, mean_fde, rx, ry, rz = metrics.mean(axis=0)
        print("==== CA-KF  ====")
        print(f"窗口数量: {metrics.shape[0]}")
        print(f"RMSE = {mean_rmse:.4f}")
        print(f"ADE = {mean_ade:.4f}")
        print(f"FDE = {mean_fde:.4f}")
        summary_data = np.array([
            ["窗口数量", metrics.shape[0]],
            ["RMSE", round(mean_rmse, 4)],
            ["ADE", round(mean_ade, 4)],
            ["FDE", round(mean_fde, 4)],
        ])
        np.savetxt("results/tables/ca_kf_summary.csv", summary_data, delimiter=",", fmt="%s", encoding="utf-8")



    # 3D绘图
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gt_all[:,0], gt_all[:,1], gt_all[:,2], c="#ff3333", s=8, alpha=0.7, label="true")
    ax.plot(kf_traj[:,0], kf_traj[:,1], kf_traj[:,2], c="#0066ff", lw=1.0, label="CA-KF")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("CA-KF Trajectory")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(r"C:\Users\86134\Desktop\drone\results\figures\ca_kf_3d.png", dpi=300, bbox_inches="tight")
    plt.show()