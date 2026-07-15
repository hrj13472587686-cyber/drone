import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------- 1. CA3D 恒加速度卡尔曼(原有) --------------------------
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
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(3))
        self.x = self.x + K @ y
        I = np.eye(9)
        self.P = (I - K@self.H) @ self.P @ (I - K@self.H).T + K @ self.R @ K.T

    def get_pos(self):
        return np.array([self.x[0,0], self.x[3,0], self.x[6,0]])

# -------------------------- 2. CV3D 恒速度卡尔曼 --------------------------
class CV3DKalmanFilter:
    def __init__(self, std_pos, std_acc):
        self.x = np.zeros((6, 1))
        self.P = np.diag(np.array([40.0,20.0,20.0,1.0,1.0,1.0]))
        self.std_pos = std_pos
        self.std_acc = std_acc
        self.dt = None

    def init_state(self, pos, vel=None, acc=None):
        self.x[0,0] = pos[0]
        self.x[1,0] = pos[1]
        self.x[2,0] = pos[2]
        if vel is not None:
            self.x[3,0] = vel[0]
            self.x[4,0] = vel[1]
            self.x[5,0] = vel[2]

    def update_F_Q(self, dt):
        self.dt = dt
        F1d = np.array([[1, dt],[0, 1]])
        self.F = np.block([
            [F1d, np.zeros((2,2)), np.zeros((2,2))],
            [np.zeros((2,2)), F1d, np.zeros((2,2))],
            [np.zeros((2,2)), np.zeros((2,2)), F1d]
        ])
        sa2 = self.std_acc ** 2
        Q1d = sa2 * np.array([[dt**3/3, dt**2/2],[dt**2/2, dt]])
        self.Q = np.block([
            [Q1d, np.zeros((2,2)), np.zeros((2,2))],
            [np.zeros((2,2)), Q1d, np.zeros((2,2))],
            [np.zeros((2,2)), np.zeros((2,2)), Q1d]
        ])
        self.H = np.zeros((3,6))
        self.H[0,0]=1.0
        self.H[1,1]=1.0
        self.H[2,2]=1.0
        self.R = np.diag([self.std_pos**2]*3)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array(z).reshape(3,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(3))
        self.x = self.x + K @ y
        I = np.eye(6)
        KH = K @ self.H
        self.P = (I-KH)@self.P@(I-KH).T + K@self.R@K.T

    def get_pos(self):
        return np.array([self.x[0,0], self.x[1,0], self.x[2,0]])

# -------------------------- 3. 线性基准：滑动窗口最小二乘LR（匀速模型） --------------------------
class LinearBaseline3D:
    def __init__(self):
        self.buffer = []  # list [t, x,y,z]
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.x0 = 0.0
        self.y0 = 0.0
        self.z0 = 0.0

    def init_state(self, pos, vel=None, acc=None):
        """对齐KF统一接口，acc参数占位兼容"""
        self.buffer.clear()

    def add_obs(self, t, pos):
        self.buffer.append([float(t), pos[0], pos[1], pos[2]])

    def fit(self):
        n = len(self.buffer)
        if n < 2:
            return
        data = np.array(self.buffer)
        t_arr = data[:,0:1]
        xyz = data[:,1:]
        A = np.hstack([t_arr, np.ones_like(t_arr)])
        theta,_,_,_ = np.linalg.lstsq(A, xyz, rcond=None)
        self.vx, self.x0 = theta[:,0]
        self.vy, self.y0 = theta[:,1]
        self.vz, self.z0 = theta[:,2]

    def predict_at_time(self, t_query):
        x = self.vx * t_query + self.x0
        y = self.vy * t_query + self.y0
        z = self.vz * t_query + self.z0
        return np.array([x,y,z])

# -------------------------- 4. 指标计算函数（完全沿用原版） --------------------------
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

# -------------------------- 5. 统一滑动窗口评测入口 --------------------------
def sliding_evaluate_windows(seq, time_seq, obs_steps, pred_steps,
                             stride=1, model_type="CAKF", std_pos=0.05, std_acc=0.1):
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

        # ========= 模型初始化分支 =========
        if model_type == "CAKF":
            tracker = CA3DKalmanFilter(std_pos, std_acc)
        elif model_type == "CVKF":
            tracker = CV3DKalmanFilter(std_pos, std_acc)
        elif model_type == "LR":
            tracker = LinearBaseline3D()
        else:
            raise NotImplementedError("model_type: CAKF / CVKF / LR")

        tracker.init_state(pos=seq[start])
        pred_win[0] = seq[start]

        # 计算初始速度用于KF初始化（LR不需要）
        v0 = None
        if start + 1 < N:
            dt0 = time_seq[start+1] - time_seq[start]
            v0 = (seq[start+1] - seq[start]) / dt0
            if model_type in ["CAKF","CVKF"]:
                tracker.init_state(pos=seq[start], vel=v0)

        # 观测窗口前obs_steps帧
        for idx_win in range(1, obs_steps):
            g = start + idx_win
            t_curr = time_seq[g]
            if model_type in ["CAKF","CVKF"]:
                dt = time_seq[g] - time_seq[g-1]
                tracker.update_F_Q(dt)
                tracker.predict()
                tracker.update(seq[g])
                pred_win[idx_win] = tracker.get_pos()
            else: # LR基准
                tracker.add_obs(t_curr, seq[g])
                tracker.fit()
                pred_win[idx_win] = tracker.predict_at_time(t_curr)

        # 多步预测阶段
        current_t = time_seq[start + obs_steps - 1]
        for idx_win in range(obs_steps, win_total):
            g = start + idx_win
            t_future = time_seq[g]
            if model_type in ["CAKF","CVKF"]:
                dt = t_future - current_t
                tracker.update_F_Q(dt)
                tracker.predict()
                pred_win[idx_win] = tracker.get_pos()
                current_t = t_future
            else: # LR线性外推
                pred_win[idx_win] = tracker.predict_at_time(t_future)

        metrics = compute_pred_only_metrics(gt_win, pred_win, obs_steps)
        metric_records.append(metrics)
        start += stride
    return np.array(metric_records)

# -------------------------- 6. 全局重叠平均预测轨迹 --------------------------
def sliding_predict_overlap(seq, time_seq, obs_steps, pred_steps,
                            stride=1, model_type="CAKF", std_pos=0.05, std_acc=0.1):
    N = len(seq)
    full_pred = np.zeros_like(seq)
    pred_cnt = np.zeros_like(seq)

    start = 0
    while True:
        end_obs = start + obs_steps
        end_pred = start + obs_steps + pred_steps
        if end_obs >= N:
            break

        if model_type == "CAKF":
            tracker = CA3DKalmanFilter(std_pos, std_acc)
        elif model_type == "CVKF":
            tracker = CV3DKalmanFilter(std_pos, std_acc)
        elif model_type == "LR":
            tracker = LinearBaseline3D()
        else:
            raise NotImplementedError

        tracker.init_state(pos=seq[start])
        if start + 1 < N and model_type in ["CAKF","CVKF"]:
            dt0 = time_seq[start+1] - time_seq[start]
            v0 = (seq[start+1] - seq[start]) / dt0
            tracker.init_state(pos=seq[start], vel=v0)

        # 观测段
        for i in range(start, end_obs):
            if i == start:
                full_pred[i] += seq[i]
                pred_cnt[i] += 1
                continue
            t_i = time_seq[i]
            if model_type in ["CAKF","CVKF"]:
                dt = time_seq[i] - time_seq[i-1]
                tracker.update_F_Q(dt)
                tracker.predict()
                tracker.update(seq[i])
                full_pred[i] += tracker.get_pos()
            else:
                tracker.add_obs(t_i, seq[i])
                tracker.fit()
                full_pred[i] += tracker.predict_at_time(t_i)
            pred_cnt[i] += 1

        # 预测段
        pred_end = min(end_pred, N)
        current_t = time_seq[end_obs - 1]
        for j in range(end_obs, pred_end):
            t_j = time_seq[j]
            if model_type in ["CAKF","CVKF"]:
                dt = t_j - current_t
                tracker.update_F_Q(dt)
                tracker.predict()
                full_pred[j] += tracker.get_pos()
                current_t = t_j
            else:
                full_pred[j] += tracker.predict_at_time(t_j)
            pred_cnt[j] += 1
        start += stride

    mask = pred_cnt > 0
    full_pred[mask] = full_pred[mask] / pred_cnt[mask]
    full_pred[~mask] = seq[~mask]
    return full_pred

# -------------------------- 主运行入口 --------------------------
if __name__ == "__main__":
    # ========== 配置 ==========
    obs_steps = 8
    pred_steps = 5
    stride = 1
    std_pos = 0.05
    std_acc = 0.1

    # 可选："CAKF" / "CVKF" / "LR"
    model_type = "CVKF"

    csv_path = "data/mmaud_mavic3_gt_relative.csv"
    csv_data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(0,4,5,6))
    t_all = csv_data[:, 0]
    gt_all = csv_data[:, 1:4]
    mask = ~np.isnan(gt_all).any(axis=1)
    t_all = t_all[mask]
    gt_all = gt_all[mask]
    print(f"总轨迹点数：{len(gt_all)} | Model = {model_type}")

    # 全局预测轨迹 & 指标
    kf_traj = sliding_predict_overlap(gt_all, t_all, obs_steps=obs_steps, pred_steps=pred_steps,
                                      stride=stride, model_type=model_type, std_pos=std_pos, std_acc=std_acc)
    metrics = sliding_evaluate_windows(gt_all, t_all, obs_steps=obs_steps, pred_steps=pred_steps,
                                       stride=stride, model_type=model_type, std_pos=std_pos, std_acc=std_acc)

    # 输出汇总指标
    if metrics.shape[0] > 0:
        mean_rmse, mean_ade, mean_fde, rx, ry, rz = metrics.mean(axis=0)
        print("==== 全局平均指标 ====")
        print(f"窗口总数: {metrics.shape[0]}")
        print(f"RMSE = {mean_rmse:.4f}")
        print(f"ADE = {mean_ade:.4f}")
        print(f"FDE = {mean_fde:.4f}")
        print(f"RMSE_X = {rx:.4f}, RMSE_Y = {ry:.4f}, RMSE_Z = {rz:.4f}")

        out_dir = "results/tables"
        os.makedirs(out_dir, exist_ok=True)
        save_name = f"{model_type}_metrics_summary.csv"
        summary_data = np.array([
            ["窗口数量", metrics.shape[0]],
            ["RMSE", round(mean_rmse, 4)],
            ["ADE", round(mean_ade, 4)],
            ["FDE", round(mean_fde, 4)],
            ["RMSE_X", round(rx, 4)],
            ["RMSE_Y", round(ry, 4)],
            ["RMSE_Z", round(rz, 4)]
        ])
        np.savetxt(os.path.join(out_dir, save_name), summary_data, delimiter=",", fmt="%s", encoding="utf-8")
        print(f"指标保存: {os.path.join(out_dir, save_name)}")

    # ========== 单窗口可视化（完全沿用你原有绘图逻辑） ==========
    target_win_idx = 700
    if 0 <= target_win_idx < len(metrics):
        win_rmse, win_ade, win_fde, _, _, _ = metrics[target_win_idx]
        print(f"\n==== 第{target_win_idx}窗口指标 ====")
        print(f"RMSE={win_rmse:.4f}, ADE={win_ade:.4f}, FDE={win_fde:.4f}")

        win_total_len = obs_steps + pred_steps
        start = target_win_idx * stride
        end = start + win_total_len
        win_gt = gt_all[start:end]
        win_time = t_all[start:end]
        win_pred = np.zeros_like(win_gt)
        win_pred[0] = win_gt[0]

        # 初始化跟踪器
        if model_type == "CAKF":
            tracker = CA3DKalmanFilter(std_pos, std_acc)
        elif model_type == "CVKF":
            tracker = CV3DKalmanFilter(std_pos, std_acc)
        else:
            tracker = LinearBaseline3D()
        tracker.init_state(pos=win_gt[0])
        if len(win_gt)>1 and model_type in ["CAKF","CVKF"]:
            dt0 = win_time[1] - win_time[0]
            v0 = (win_gt[1] - win_gt[0]) / dt0
            tracker.init_state(pos=win_gt[0], vel=v0)

        # 观测段推演
        for idx_win in range(1, obs_steps):
            t_curr = win_time[idx_win]
            if model_type in ["CAKF","CVKF"]:
                dt = win_time[idx_win] - win_time[idx_win-1]
                tracker.update_F_Q(dt)
                tracker.predict()
                tracker.update(win_gt[idx_win])
                win_pred[idx_win] = tracker.get_pos()
            else:
                tracker.add_obs(t_curr, win_gt[idx_win])
                tracker.fit()
                win_pred[idx_win] = tracker.predict_at_time(t_curr)

        # 预测段外推
        current_t = win_time[obs_steps - 1]
        for idx_win in range(obs_steps, win_total_len):
            t_future = win_time[idx_win]
            if model_type in ["CAKF","CVKF"]:
                dt = t_future - current_t
                tracker.update_F_Q(dt)
                tracker.predict()
                win_pred[idx_win] = tracker.get_pos()
                current_t = t_future
            else:
                win_pred[idx_win] = tracker.predict_at_time(t_future)

        # 保存窗口指标csv
        win_metric_data = np.array([["RMSE", win_rmse],["ADE", win_ade],["FDE", win_fde]])
        np.savetxt(f"{model_type}_window_{target_win_idx}_metrics.csv", win_metric_data, delimiter=",", fmt="%s")

        # 绘图逻辑完全不变
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(gt_all[:, 0], gt_all[:, 1], gt_all[:, 2], c="#999922", lw=0.7, alpha=0.6, label="True Trajectory")
        hist_gt = win_gt[:obs_steps]
        ax.plot(hist_gt[:, 0], hist_gt[:, 1], hist_gt[:, 2], c="orange", lw=1, label="History")
        ax.scatter(hist_gt[:, 0], hist_gt[:, 1], hist_gt[:, 2], c="orange", s=20)
        fut_pred = win_pred[obs_steps:]
        ax.plot(fut_pred[:, 0], fut_pred[:, 1], fut_pred[:, 2], c="#0066ff", lw=1, label="Future Pred")
        ax.scatter(fut_pred[:, 0], fut_pred[:, 1], fut_pred[:, 2], c="#0066ff", s=20)
        ax.set_xlabel("X 坐标(m)")
        ax.set_ylabel("Y 坐标(m)")
        ax.set_zlabel("Z 坐标(m)")
        ax.set_title(f"{model_type} | 窗口{target_win_idx} | obs={obs_steps}, pred={pred_steps}")
        ax.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{model_type}_window_{target_win_idx}_3d_plot.png", dpi=300, bbox_inches='tight')
        plt.show()