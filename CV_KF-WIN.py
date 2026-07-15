import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== CV常速度3维卡尔曼滤波器（修复矩阵维度bug） =====================
import numpy as np

class CV3DKalmanFilter:
    def __init__(self, std_pos, std_vel):
        # 位置协方差放大到20，速度保留1，允许观测快速修正位置偏移
        self.x = np.zeros((6, 1))
        self.P = np.diag(np.array([20.0, 20.0, 20.0, 1.0, 1.0, 1.0]))
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.dt = None

    def init_state(self, pos, vel=None):
        """
        初始化状态
        pos: [x,y,z] 位置
        vel: [vx,vy,vz] 初始速度，可选
        """
        self.x[0, 0] = pos[0]
        self.x[1, 0] = pos[1]
        self.x[2, 0] = pos[2]
        if vel is not None:
            self.x[3, 0] = vel[0]
            self.x[4, 0] = vel[1]
            self.x[5, 0] = vel[2]

    def update_F_Q(self, dt):
        """根据时间间隔更新F状态转移矩阵、Q过程噪声、R观测噪声"""
        self.dt = dt
        dt2 = dt ** 2
        dt3 = dt ** 3

        # 构造单轴F块 [[1,dt],[0,1]]
        F_block = np.array([[1, dt], [0, 1]], dtype=np.float64)
        # 对角拼接三轴F
        self.F = np.block([
            [F_block, np.zeros((2, 2)), np.zeros((2, 2))],
            [np.zeros((2, 2)), F_block, np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2)), F_block]
        ])

        # 单轴Q块 (速度驱动匀速模型噪声)
        def get_q_block(sv):
            sv2 = sv ** 2
            return sv2 * np.array([
                [dt3 / 3, dt2 / 2],
                [dt2 / 2, dt]
            ])
        qx = get_q_block(self.std_vel)
        qy = get_q_block(self.std_vel)
        qz = get_q_block(self.std_vel)
        self.Q = np.block([
            [qx, np.zeros((2, 2)), np.zeros((2, 2))],
            [np.zeros((2, 2)), qy, np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2)), qz]
        ])

        # 观测矩阵：仅观测位置，不观测速度
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # 三轴独立观测噪声
        self.R = np.diag([
            self.std_pos ** 2,
            self.std_pos ** 2,
            self.std_pos ** 2
        ])

    def predict(self):
        # 预测步 x=F·x  P=F·P·F^T + Q
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        观测更新
        z: [x,y,z] 观测位置
        """
        z = np.array(z, dtype=np.float64).reshape(3, 1)
        # 残差
        y = z - self.H @ self.x
        # 残差协方差
        S = self.H @ self.P @ self.H.T + self.R
        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # 状态更新
        self.x = self.x + K @ y
        # Joseph稳定协方差更新（无维度错误）
        I = np.eye(6, dtype=np.float64)
        KH = K @ self.H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ self.R @ K.T

    def get_pos(self):
        """输出当前滤波位置 [x,y,z]"""
        return np.array([
            self.x[0, 0],
            self.x[1, 0],
            self.x[2, 0]
        ], dtype=np.float64)

# ===================== 指标计算函数 =====================
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

    rmse = np.sqrt(np.mean(np.sum(err ** 2, axis=1)))
    ade = np.mean(disp_err)
    fde = disp_err[-1]

    rmse_x = np.sqrt(np.mean(err[:, 0] ** 2))
    rmse_y = np.sqrt(np.mean(err[:, 1] ** 2))
    rmse_z = np.sqrt(np.mean(err[:, 2] ** 2))

    return rmse, ade, fde, rmse_x, rmse_y, rmse_z

# ===================== 滑动窗口批量评测 =====================
def sliding_cv_evaluate_windows(seq, time_seq, obs_steps, pred_steps, stride=1, std_pos=0.05, std_vel=0.1):
    N = len(seq)
    win_total = obs_steps + pred_steps
    metric_records = []
    start = 0
    while True:
        end_window = start + win_total
        if end_window > N:
            break
        gt_win = seq[start:end_window].copy()
        pred_win = np.zeros_like(seq[start:end_window])

        kf = CV3DKalmanFilter(std_pos, std_vel)
        kf.init_state(pos=seq[start])
        if start + 1 < N:
            dt0 = time_seq[start + 1] - time_seq[start]
            v0 = (seq[start + 1] - seq[start]) / dt0
            kf.init_state(pos=seq[start], vel=v0)
        pred_win[0] = seq[start]

        for idx_win in range(1, obs_steps):
            g = start + idx_win
            dt = time_seq[g] - time_seq[g - 1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.update(seq[g])
            pred_win[idx_win] = kf.get_pos()

        # 多步预测
        for idx_win in range(obs_steps, win_total):
            g = start + idx_win
            dt = time_seq[g] - time_seq[g - 1]
            kf.update_F_Q(dt)
            kf.predict()
            pred_win[idx_win] = kf.get_pos()

        metrics = compute_pred_only_metrics(gt_win, pred_win, obs_steps)
        metric_records.append(metrics)
        start += stride
    return np.array(metric_records)

# ===================== 全局重叠预测=====================
def sliding_cv_predict_overlap(seq, time_seq, obs_steps, pred_steps, stride=1, std_pos=0.05, std_vel=0.1):
    N = len(seq)
    full_pred = np.zeros_like(seq)
    pred_cnt = np.zeros_like(seq)
    start = 0
    while True:
        end_obs = start + obs_steps
        end_pred = start + obs_steps + pred_steps
        if end_obs >= N:
            break
        end_pred = min(end_pred, N)
        win_len = end_pred - start

        kf = CV3DKalmanFilter(std_pos, std_vel)
        kf.init_state(pos=seq[start])
        if start + 1 < N:
            dt0 = time_seq[start + 1] - time_seq[start]
            v0 = (seq[start + 1] - seq[start]) / dt0
            kf.init_state(pos=seq[start], vel=v0)

        pred_win = np.zeros((win_len, 3))
        pred_win[0] = seq[start]
        # 观测更新
        for idx_win in range(1, obs_steps):
            g = start + idx_win
            dt = time_seq[g] - time_seq[g - 1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.update(seq[g])
            pred_win[idx_win] = kf.get_pos()
        # 无更新纯预测
        for idx_win in range(obs_steps, win_len):
            g = start + idx_win
            dt = time_seq[g] - time_seq[g - 1]
            kf.update_F_Q(dt)
            kf.predict()
            pred_win[idx_win] = kf.get_pos()

        full_pred[start:end_pred] += pred_win
        pred_cnt[start:end_pred] += 1
        start += stride

    mask = pred_cnt > 0
    full_pred[mask] = full_pred[mask] / pred_cnt[mask]
    full_pred[~mask] = seq[~mask]
    return full_pred

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 超参数
    obs_steps = 8
    pred_steps = 5
    stride = 1
    std_pos = 0.05
    std_vel = 0.01

    # 读取轨迹csv
    csv_path = "data/mmaud_mavic3_gt_relative.csv"
    csv_data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(0, 4, 5, 6))
    t_all = csv_data[:, 0]
    gt_all = csv_data[:, 1:4]
    mask = ~np.isnan(gt_all).any(axis=1)
    t_all = t_all[mask]
    gt_all = gt_all[mask]
    print(f"加载有效轨迹点数量：{len(gt_all)}")

    # CV滤波全局预测 + 窗口指标
    cv_traj = sliding_cv_predict_overlap(gt_all, t_all, obs_steps, pred_steps, stride, std_pos, std_vel)
    metrics = sliding_cv_evaluate_windows(gt_all, t_all, obs_steps, pred_steps, stride, std_pos, std_vel)

    # 输出并保存全局平均指标
    if metrics.shape[0] > 0:
        mean_rmse, mean_ade, mean_fde, rx, ry, rz = metrics.mean(axis=0)
        print("==== CV-KF 常速度卡尔曼 全局平均指标 ====")
        print(f"窗口总数：{metrics.shape[0]}")
        print(f"RMSE = {mean_rmse:.4f}")
        print(f"ADE = {mean_ade:.4f}")
        print(f"FDE = {mean_fde:.4f}")
        print(f"RMSE_X = {rx:.4f}, RMSE_Y = {ry:.4f}, RMSE_Z = {rz:.4f}")

        summary_data = np.array([
            ["窗口数量", metrics.shape[0]],
            ["RMSE", round(mean_rmse, 4)],
            ["ADE", round(mean_ade, 4)],
            ["FDE", round(mean_fde, 4)],
            ["RMSE_X", round(rx, 4)],
            ["RMSE_Y", round(ry, 4)],
            ["RMSE_Z", round(rz, 4)]
        ])
        np.savetxt("cv_kf_metrics_summary.csv", summary_data, delimiter=",", fmt="%s", encoding="utf-8")
        header = "RMSE,ADE,FDE,RMSE_X,RMSE_Y,RMSE_Z"
        print("全局指标文件已保存：cv_kf_metrics_summary.csv\n")

    # ---------------- 指定单窗口可视化与保存 ----------------
    # 解决matplotlib中文乱码
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    # 图片保存路径
    img_dir = r"C:\Users\86134\Desktop\drone\window_figs"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    target_win_idx = 236  # 修改此处切换窗口
    if 0 <= target_win_idx < len(metrics):
        win_rmse, win_ade, win_fde, _, _, _ = metrics[target_win_idx]
        print(f"==== 第{target_win_idx}窗口 CV-KF 指标 ====")
        print(f"RMSE={win_rmse:.4f}, ADE={win_ade:.4f}, FDE={win_fde:.4f}")

        win_total_len = obs_steps + pred_steps
        start = target_win_idx * stride
        end = start + win_total_len
        win_gt = gt_all[start:end]
        win_time = t_all[start:end]

        # 重建当前窗口KF
        kf = CV3DKalmanFilter(std_pos, std_vel)
        first_pos = win_gt[0]
        kf.init_state(pos=first_pos)
        if len(win_gt) > 1:
            dt0 = win_time[1] - win_time[0]
            v0 = (win_gt[1] - win_gt[0]) / dt0
            kf.init_state(pos=first_pos, vel=v0)
        win_pred = np.zeros_like(win_gt)
        win_pred[0] = win_gt[0]

        for idx_win in range(1, obs_steps):
            dt = win_time[idx_win] - win_time[idx_win - 1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.update(win_gt[idx_win])
            win_pred[idx_win] = kf.get_pos()
        for idx_win in range(obs_steps, win_total_len):
            dt = win_time[idx_win] - win_time[idx_win - 1]
            kf.update_F_Q(dt)
            kf.predict()
            win_pred[idx_win] = kf.get_pos()

        # 仅保存窗口RMSE/ADE/FDE指标csv
        win_metric_data = np.array([["RMSE", win_rmse], ["ADE", win_ade], ["FDE", win_fde]])
        save_csv_name = f"cv_window_{target_win_idx}_metrics.csv"
        np.savetxt(save_csv_name, win_metric_data, delimiter=",", fmt="%s", encoding="utf-8")
        print(f"窗口指标CSV已保存：{save_csv_name}")

        # 绘图配色：全局深红色、观测橙色、预测蓝色
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(gt_all[:, 0], gt_all[:, 1], gt_all[:, 2], c="#992222", lw=0.8, alpha=0.6, label="全局真值")
        hist_gt = win_gt[:obs_steps]
        ax.plot(hist_gt[:, 0], hist_gt[:, 1], hist_gt[:, 2], c="orange", lw=2, label="观测History")
        ax.scatter(hist_gt[:, 0], hist_gt[:, 1], hist_gt[:, 2], c="orange", s=25)
        fut_pred = win_pred[obs_steps:]
        ax.plot(fut_pred[:, 0], fut_pred[:, 1], fut_pred[:, 2], c="#0066ff", lw=2, label="预测Future")
        ax.scatter(fut_pred[:, 0], fut_pred[:, 1], fut_pred[:, 2], c="#0066ff", s=25)

        ax.set_xlabel("X 坐标(m)")
        ax.set_ylabel("Y 坐标(m)")
        ax.set_zlabel("Z 坐标(m)")
        ax.set_title(f"第{target_win_idx}窗口 | CV常速度滤波 观测{obs_steps}步 预测{pred_steps}步")
        ax.legend()
        plt.grid(True, alpha=0.3)
        save_img_name = os.path.join(img_dir, f"cv_window_{target_win_idx}_3d_plot.png")
        plt.savefig(save_img_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"窗口3D图保存至：{save_img_name}\n")
    else:
        print(f"窗口索引越界！可用范围 0 ~ {len(metrics)-1}")

    # 全局轨迹总图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gt_all[:, 0], gt_all[:, 1], gt_all[:, 2], c="#ff3333", s=8, alpha=0.7, label="全局真值")
    ax.plot(cv_traj[:, 0], cv_traj[:, 1], cv_traj[:, 2], c="#0066ff", lw=1.0, label="CV-KF预测")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("CV常速度卡尔曼 全局3D预测轨迹")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()