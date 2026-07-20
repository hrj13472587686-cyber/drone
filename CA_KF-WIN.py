import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------- 1. CA卡尔曼滤波器 --------------------------
class CA3DKalmanFilter:
    def __init__(self, std_pos, std_acc):
        # 状态向量：x,y,z, vx,vy,vz, ax,ay,az (9,1)
        self.x = np.zeros((9, 1))
        self.P = np.diag(np.ones(9) * 1.0)   # 初始协方差
        self.std_pos = std_pos                # 位置观测噪声标准差 [m]
        self.std_acc = std_acc                # 加速度过程噪声（jerk）标准差 [m/s²?]
        self.dt = None

    def init_state(self, pos, vel=None, acc=None):
        """初始化状态，pos, vel, acc 均为长度3的序列"""
        self.x[0:3, 0] = np.array(pos).reshape(3)      # x,y,z
        if vel is not None:
            self.x[3:6, 0] = np.array(vel).reshape(3)  # vx,vy,vz
        if acc is not None:
            self.x[6:9, 0] = np.array(acc).reshape(3)  # ax,ay,az

    def update_F_Q(self, dt):
        self.dt = dt
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        dt5 = dt ** 5


        I3 = np.eye(3)
        Z3 = np.zeros((3,3))
        self.F = np.block([
            [I3, dt * I3, 0.5 * dt2 * I3],
            [Z3, I3,      dt * I3],
            [Z3, Z3,      I3]
        ])

        # 单轴过程噪声协方差 Q1d（来自 jerk 白噪声模型）
        sa2 = self.std_acc ** 2
        # 单轴系数
        q11 = sa2 * dt5 / 20   # 位置方差
        q12 = sa2 * dt4 / 8    # 位置-速度协方差
        q13 = sa2 * dt3 / 6    # 位置-加速度协方差
        q22 = sa2 * dt3 / 3    # 速度方差
        q23 = sa2 * dt2 / 2    # 速度-加速度协方差
        q33 = sa2 * dt         # 加速度方差

        # Q 矩阵：块形式，每个块 = 系数 × I3
        self.Q = np.block([
            [q11 * I3, q12 * I3, q13 * I3],
            [q12 * I3, q22 * I3, q23 * I3],
            [q13 * I3, q23 * I3, q33 * I3]
        ])

        # 观测矩阵：只测量位置，H = [I3, 0, 0]
        self.H = np.hstack([np.eye(3), np.zeros((3,6))])

        # 观测噪声协方差（各轴独立、等方差）
        self.R = np.diag([self.std_pos**2] * 3)

    def predict(self):
        """预测步：状态与协方差传播"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, z):
        """更新步：用观测z修正状态和协方差（原 update 方法）"""
        z = np.array(z).reshape(3, 1)
        y = z - self.H @ self.x               # 新息
        S = self.H @ self.P @ self.H.T + self.R  # 新息协方差
        S += 1e-8 * np.eye(3)                 # 防止奇异
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        self.x = self.x + K @ y               # 状态更新
        I = np.eye(9)
        # Joseph形式协方差更新，保证对称正定
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

    def get_pos(self):
        """返回当前估计的位置"""
        return self.x[0:3, 0]

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

# -------------------------- 主函数（直接读取CSV真值轨迹） --------------------------
if __name__ == "__main__":
    # 超参配置
    obs_steps = 8
    pred_steps = 5
    stride = 1
    std_pos = 0.05
    std_acc = 0.1


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

    # 全局预测 + 全部窗口指标
    kf_traj = sliding_ca_predict_overlap(gt_all, t_all, obs_steps=obs_steps, pred_steps=pred_steps, stride=stride, std_pos=std_pos, std_acc=std_acc)
    metrics = sliding_ca_evaluate_windows(gt_all, t_all, obs_steps=obs_steps, pred_steps=pred_steps, stride=stride, std_pos=std_pos, std_acc=std_acc)



        # ====================== 【指定窗口切片功能】 ======================
        # 修改这里更换你要查看的窗口索引（从0开始，0=第一个窗口）
    target_win_idx = 700

    if 0 <= target_win_idx < len(metrics):
        # 取出当前窗口三项误差指标
        win_rmse, win_ade, win_fde, _, _, _ = metrics[target_win_idx]
        print(f"==== 第{target_win_idx}个窗口 单独指标 ====")
        print(f"窗口RMSE={win_rmse:.4f}, ADE={win_ade:.4f}, FDE={win_fde:.4f}")

        win_total_len = obs_steps + pred_steps
        start = target_win_idx * stride
        end = start + win_total_len
        # 窗口内真值与时间
        win_gt = gt_all[start:end]
        win_time = t_all[start:end]

        # 重建KF窗口预测（修复pos传参bug）
        kf = CA3DKalmanFilter(std_pos, std_acc)
        first_pos = win_gt[0]
        kf.init_state(pos=first_pos)
        if len(win_gt) > 1:
            dt0 = win_time[1] - win_time[0]
            v0 = (win_gt[1] - win_gt[0]) / dt0
            kf.init_state(pos=first_pos, vel=v0)
        win_pred = np.zeros_like(win_gt)
        win_pred[0] = win_gt[0]

        # 观测段滤波
        for idx_win in range(1, obs_steps):
            dt = win_time[idx_win] - win_time[idx_win - 1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.correct(win_gt[idx_win])
            win_pred[idx_win] = kf.get_pos()
        # 预测段多步外推
        current_t = win_time[obs_steps - 1]
        for idx_win in range(obs_steps, win_total_len):
            dt = win_time[idx_win] - current_t
            kf.update_F_Q(dt)
            kf.predict()
            win_pred[idx_win] = kf.get_pos()
            current_t = win_time[idx_win]

        # 1. CSV只保存当前窗口RMSE,ADE,FDE，不再输出时序数据
        win_metric_data = np.array([
            ["RMSE", win_rmse],
            ["ADE", win_ade],
            ["FDE", win_fde]
        ])
        save_csv_name = f"CA_window_{target_win_idx}_metrics.csv"
        np.savetxt(save_csv_name, win_metric_data, delimiter=",", fmt="%s", encoding="utf-8")
        print(f"窗口指标CSV已保存：{save_csv_name}")

        # 2. 3D图：全轨迹灰色 + 切片history橙色 + future蓝色
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # 整条完整真值：浅灰色背景
        ax.plot(gt_all[:, 0], gt_all[:, 1], gt_all[:, 2], c="#999922", lw=0.7, alpha=0.6, label="True Trajectory")

        # 窗口观测历史段 history：橙色
        hist_gt = win_gt[:obs_steps]
        ax.plot(hist_gt[:, 0], hist_gt[:, 1], hist_gt[:, 2], c="orange", lw=1, label="History")
        ax.scatter(hist_gt[:, 0], hist_gt[:, 1], hist_gt[:, 2], c="orange", s=20)

        # 窗口预测未来段 future：蓝色
        fut_pred = win_pred[obs_steps:]
        fut_gt_true = win_gt[obs_steps:]
        ax.plot(fut_pred[:, 0], fut_pred[:, 1], fut_pred[:, 2], c="#0066ff", lw=1, label="Future")
        ax.scatter(fut_pred[:, 0], fut_pred[:, 1], fut_pred[:, 2], c="#0066ff", s=20)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"win{target_win_idx} | obs={obs_steps} pred={pred_steps}")
        ax.legend()
        plt.grid(True, alpha=0.3)

        save_img_name = f"CA_window_{target_win_idx}_3d_plot.png"
        plt.savefig(save_img_name, dpi=300, bbox_inches='tight')

        plt.show()
        print(f"窗口3D对比图已保存：{save_img_name}\n")
    else:
        print(f"窗口索引超出范围！总窗口数：{len(metrics)}，请修改target_win_idx 取值 0~{len(metrics) - 1}")