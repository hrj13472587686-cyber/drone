import numpy as np
import os
import matplotlib.pyplot as plt

# ====================== 3D CV卡尔曼滤波器 ======================

class CV3DKalmanFilter:
    def __init__(self, std_pos, std_vel):
        self.x = np.zeros((6, 1))
        self.P = np.diag(np.ones(6) * 1.0)          #状态估计误差
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.dt = None

    def init_state(self, pos, vel=None):
        """初始化状态向量 [x,y,z,vx,vy,vz]^T"""
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
        ])                                      #状态转移矩阵

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
        ])                                              #过程噪声协方差矩阵
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])                                      #观测矩阵
        self.R = np.diag([self.std_pos**2, self.std_pos**2, self.std_pos**2])           #观察噪音

    def predict(self):
        """预测步：状态+协方差完整传播"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, z):
        z = np.array(z).reshape(3, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        S += 1e-8 * np.eye(3)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I6 = np.eye(6)
        self.P = (I6 - K @ self.H) @ self.P @ (I6 - K @ self.H).T + K @ self.R @ K.T

    def get_pos(self):
        return self.x[:3,0]

# ====================== 3D CA-KF ======================
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
# =================================================================================

# ===================== IMM Filter (CV + CA 双模型) =====================
class IMM3DFilter:
    def __init__(self, std_pos, std_vel_cv, std_acc_ca, trans_matrix=None):
        self.std_pos = std_pos
        self.filters = [
            CV3DKalmanFilter(std_pos, std_vel_cv),
            CA3DKalmanFilter(std_pos, std_acc_ca)
        ]
        self.mu = np.array([0.5, 0.5])           # 初始模型概率
        if trans_matrix is None:            # Markov 转移概率矩阵
            self.trans = np.array([
                [0.9995, 0.0005],
                [0.0005, 0.9995]
            ])
        else:
            self.trans = trans_matrix
        self.dt = None

    def init_state(self, pos, vel=None):
        for f in self.filters:
            if isinstance(f, CV3DKalmanFilter):
                f.init_state(pos=pos, vel=vel)
            else:
                f.init_state(pos=pos, vel=vel)

    def update_F_Q(self, dt):
        for f in self.filters:
            f.update_F_Q(dt)

    def _mix_state(self):
        mu = self.mu.copy()          # 上一时刻的模型概率 [μ1, μ2]
        trans = self.trans          # 转移矩阵 Π
        c_bar = trans.T @ mu        # 预测的模型概率（归一化常数）
        c_bar[c_bar < 1e-12] = 1e-12        # 防除零，极小值兜底
        mu_ij = np.zeros_like(trans)        #计算混合权重
        for j in range(2):
            for i in range(2):
                mu_ij[i,j] = trans[i,j] * mu[i] / c_bar[j]
        x_cv = self.filters[0].x.copy()
        P_cv = self.filters[0].P.copy()
        x_ca = self.filters[1].x.copy()
        P_ca = self.filters[1].P.copy()
        x_ca_6 = x_ca[:6, :].copy()
        P_ca_6 = P_ca[:6, :6].copy()
        # 混合CV
        x0_mix = mu_ij[0,0] * x_cv + mu_ij[1,0] * x_ca_6
        P0_mix = mu_ij[0,0] * (P_cv + (x_cv - x0_mix) @ (x_cv - x0_mix).T) + \
                 mu_ij[1,0] * (P_ca_6 + (x_ca_6 - x0_mix) @ (x_ca_6 - x0_mix).T)
        # CV扩9维
        x_cv_9 = np.zeros((9,1))
        x_cv_9[:6,0] = x_cv[:,0]
        P_cv_9 = np.zeros((9,9))
        P_cv_9[:6,:6] = P_cv
        # 混合CA
        x1_mix = mu_ij[0,1] * x_cv_9 + mu_ij[1,1] * x_ca
        P1_mix = mu_ij[0,1] * (P_cv_9 + (x_cv_9 - x1_mix) @ (x_cv_9 - x1_mix).T) + \
                 mu_ij[1,1] * (P_ca + (x_ca - x1_mix) @ (x_ca - x1_mix).T)
        return [x0_mix, x1_mix], [P0_mix, P1_mix], c_bar

    def predict(self):
        mixed_x, mixed_P, self.c_bar = self._mix_state()
        self.filters[0].x = mixed_x[0]
        self.filters[0].P = mixed_P[0]
        self.filters[1].x = mixed_x[1]
        self.filters[1].P = mixed_P[1]
        self.filters[0].predict()
        self.filters[1].predict()

    def correct(self, z):
        def calc_log_like(filter_inst, z_meas):
            z_pred = filter_inst.H @ filter_inst.x
            v = z_meas - z_pred
            S = filter_inst.H @ filter_inst.P @ filter_inst.H.T + filter_inst.R
            S += 1e-10 * np.eye(3)
            detS = np.linalg.det(S)
            detS = max(detS, 1e-15)
            invS = np.linalg.inv(S)
            quad = (v.T @ invS @ v).item()
            log_det = np.log(detS)
            log_lik = -0.5 * (quad + 3 * np.log(2 * np.pi) + log_det)
            return log_lik

        z = np.array(z).reshape(3, 1)
        self.filters[0].correct(z)
        self.filters[1].correct(z)
        L0 = calc_log_like(self.filters[0], z)
        L1 = calc_log_like(self.filters[1], z)
        # 数值稳定处理
        L_max = max(L0, L1)
        L0 = np.exp(L0 - L_max)
        L1 = np.exp(L1 - L_max)
        # 注意：需要获取 c_bar（应在 predict 中保存为 self.c_bar）
        self.mu = np.array([L0 * self.c_bar[0], L1 * self.c_bar[1]])
        self.mu /= self.mu.sum()

    def get_pos(self):
        pos0 = self.filters[0].get_pos()
        pos1 = self.filters[1].get_pos()
        pos_fuse = self.mu[0] * pos0 + self.mu[1] * pos1
        # 兜底：防止mu全0时返回原始CV位置
        if np.isnan(pos_fuse).any() or np.sum(self.mu) < 1e-12:
            return pos0
        return pos_fuse

    def get_model_prob(self):
        return self.mu.copy()

# -------------------------- 指标函数（不变） --------------------------
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

# -------------------------- 滑动窗口评测函数（泛化版） --------------------------
# -------------------------- 重叠预测轨迹生成函数（仅IMM专用） --------------------------
def sliding_predict_overlap(seq, time_seq, obs_steps, pred_steps,
                            filter_cls, std_pos, std_proc, stride=1):
    N = len(seq)
    full_pred = np.zeros_like(seq)
    pred_cnt = np.zeros_like(seq)

    start = 0
    while True:
        end_obs = start + obs_steps
        end_pred = start + obs_steps + pred_steps
        if end_obs >= N:
            break

        # 固定IMM构造，std_proc=[std_vel_cv, std_acc_ca]
        kf = IMM3DFilter(std_pos, std_proc[0], std_proc[1])

        # 初始化速度
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
            # IMM统一correct
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

# -------------------------- 滑动窗口评测函数（仅IMM专用） --------------------------
def sliding_evaluate_windows(seq, time_seq, obs_steps, pred_steps,
                            filter_cls, std_pos, std_proc, stride=1):
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

        # 直接实例IMM
        kf = IMM3DFilter(std_pos, std_proc[0], std_proc[1])

        # 状态初始化
        if start + 1 < N:
            dt0 = time_seq[start+1] - time_seq[start]
            v0 = (seq[start+1] - seq[start]) / dt0
            kf.init_state(pos=seq[start], vel=v0)
        else:
            kf.init_state(pos=seq[start])

        pred_win[0] = seq[start]

        # ========= 观测段（滤波+校正） =========
        for idx_win in range(1, obs_steps):
            g = start + idx_win
            dt = time_seq[g] - time_seq[g-1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.correct(seq[g])
            pred_win[idx_win] = kf.get_pos()

        # ========= 预测段（仅预测无校正） =========
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
# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 超参配置
    obs_steps = 8
    pred_steps = 5
    stride = 1
    std_pos = 0.05
    std_vel_cv = 0.1
    std_acc_ca = 0.1

    # 创建输出目录
    os.makedirs("results/tables", exist_ok=True)
    fig_save_path = r"C:\Users\86134\Desktop\drone\results\figures\IMM_3d.png"
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

    traj_imm = sliding_predict_overlap(
        seq=gt_all,
        time_seq=t_all,
        obs_steps=8,
        pred_steps=5,
        filter_cls=IMM3DFilter,
        std_pos=std_pos,
        std_proc=[std_vel_cv, std_acc_ca],  # IMM需要两个噪声参数
        stride=1
    )

    metric_imm = sliding_evaluate_windows(
        seq=gt_all,
        time_seq=t_all,
        obs_steps=8,
        pred_steps=5,
        filter_cls=IMM3DFilter,
        std_pos=std_pos,
        std_proc=[std_vel_cv, std_acc_ca],
        stride=1
    )

    if metric_imm.shape[0] > 0:
        rmse, ade, fde, rx, ry, rz = metric_imm.mean(axis=0)
        print("==== IMM 评测结果 ====")
        print(f"窗口数量: {metric_imm.shape[0]}")
        print(f"RMSE = {rmse:.4f}")
        print(f"ADE  = {ade:.4f}")
        print(f"FDE  = {fde:.4f}")
        summary_data = np.array([
            ["窗口数量", metric_imm.shape[0]],
            ["RMSE", round(rmse, 4)],
            ["ADE", round(ade, 4)],
            ["FDE", round(fde, 4)],
            ["RMSE_X", round(rx,4)],
            ["RMSE_Y", round(ry,4)],
            ["RMSE_Z", round(rz,4)]
        ])
        np.savetxt("results/tables/imm_summary.csv", summary_data, delimiter=",", fmt="%s", encoding="utf-8")

    # 3D绘图
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gt_all[:,0], gt_all[:,1], gt_all[:,2], c="#ff3333", s=8, alpha=0.7, label=" Truth")
    ax.plot(traj_imm[:,0], traj_imm[:,1], traj_imm[:,2], c="#0066ff", lw=1.0, label="IMM")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("IMM  Trajectory")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # 图2 X时序
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(t_all, gt_all[:, 0], "r.", label="true X")
    ax2.plot(t_all, traj_imm[:, 0], "b-", label="IMM X")
    ax2.set_title("X")
    ax2.legend()
    ax2.grid()
    fig2.savefig(r"C:\Users\86134\Desktop\drone\results\figures\imm_kf_x.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 图3 Y时序
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(t_all, gt_all[:, 1], "r.", label="true Y")
    ax3.plot(t_all, traj_imm[:, 1], "b-", label="IMM Y")
    ax3.set_title("Y")
    ax3.legend()
    ax3.grid()
    fig3.savefig(r"C:\Users\86134\Desktop\drone\results\figures\imm_kf_y.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 图4 Z时序
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    ax4.plot(t_all, gt_all[:, 2], "r.", label="true Z")
    ax4.plot(t_all, traj_imm[:, 2], "b-", label="IMM Z")
    ax4.set_title("Z")
    ax4.legend()
    ax4.grid()
    fig4.savefig(r"C:\Users\86134\Desktop\drone\results\figures\imm_kf_z.png", dpi=150, bbox_inches="tight")
    plt.show()