import numpy as np
import os
import matplotlib.pyplot as plt

# ====================== 3D CV卡尔曼滤波器 ======================

class CV3DKalmanFilter:
    def __init__(self, std_pos, std_vel):
        self.x = np.zeros((6, 1))
        self.P = np.diag(np.ones(6) * 1.0)
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.dt = None

    def init_state(self, pos, vel=None):
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
        self.x = np.zeros((9, 1))
        self.P = np.diag(np.ones(9) * 1.0)
        self.std_pos = std_pos
        self.std_acc = std_acc
        self.dt = None

    def init_state(self, pos, vel=None, acc=None):
        pos = np.array(pos).reshape(3,1)
        if vel is None:
            vel = np.zeros((3,1))
        else:
            vel = np.array(vel).reshape(3,1)
        if acc is None:
            acc = np.zeros((3,1))
        else:
            acc = np.array(acc).reshape(3,1)
        self.x = np.vstack([pos, vel, acc])

    def update_F_Q(self, dt):
        self.dt = dt
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        dt5 = dt ** 5
        F_1d = np.array([
            [1, dt, dt2/2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        self.F = np.block([
                [F_1d, np.zeros((3, 3)), np.zeros((3, 3))],
                [np.zeros((3, 3)), F_1d, np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), F_1d]
               ])
        q1 = self.std_acc**2 * dt5 / 20
        q2 = self.std_acc**2 * dt4 / 8
        q3 = self.std_acc**2 * dt3 / 6
        q4 = self.std_acc**2 * dt2 / 2
        q5 = self.std_acc**2 * dt3 / 3
        q6 = self.std_acc**2 * dt
        Q_1d = np.array([
            [q1, q2, q3],
            [q2, q5, q4],
            [q3, q4, q6]
        ])
        self.Q = np.block([
            [Q_1d, np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), Q_1d, np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), Q_1d]
        ])
        self.H = np.zeros((3, 9))
        self.H[0, 0] = 1
        self.H[1, 3] = 1
        self.H[2, 6] = 1
        self.R = np.diag([self.std_pos**2, self.std_pos**2, self.std_pos**2])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array(z).reshape(3, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        S += 1e-8 * np.eye(3)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(9)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

    def get_pos(self):
        x = self.x[0, 0]
        y = self.x[3, 0]
        z = self.x[6, 0]
        return np.array([x, y, z])
# =================================================================================

# ===================== IMM Filter (CV + CA 双模型) =====================
class IMM3DFilter:
    def __init__(self, std_pos, std_vel_cv, std_acc_ca, trans_matrix=None):
        """
        :param std_pos: 观测噪声标准差
        :param std_vel_cv: CV模型过程噪声
        :param std_acc_ca: CA模型过程噪声
        :param trans_matrix: 马尔可夫模型转移概率矩阵
        model 0: CV, model 1: CA
        """
        self.std_pos = std_pos
        # 初始化两个子滤波器
        self.filters = [
            CV3DKalmanFilter(std_pos, std_vel_cv),
            CA3DKalmanFilter(std_pos, std_acc_ca)
        ]
        # 初始模型概率
        self.mu = np.array([0.5, 0.5])
        # 模型转移概率矩阵 [2x2]
        if trans_matrix is None:
            # 自转移概率高，互相跳转概率低
            self.trans = np.array([
                [0.995, 0.005],
                [0.005, 0.995]
            ])
        else:
            self.trans = trans_matrix
        self.dt = None

    def init_state(self, pos, vel=None):
        """IMM统一初始化两个子滤波器"""
        for f in self.filters:
            if isinstance(f, CV3DKalmanFilter):
                f.init_state(pos=pos, vel=vel)
            else:
                f.init_state(pos=pos, vel=vel, acc=None)

    def update_F_Q(self, dt):
        self.dt = dt
        for f in self.filters:
            f.update_F_Q(dt)

    def _mix_state(self):
        """IMM Step1: 交互混合（Mixing）"""
        mu = self.mu
        trans = self.trans
        n_model = len(self.filters)

        # 计算预测混合概率 mu_ij
        c_bar = trans.T @ mu
        mu_ij = np.zeros((n_model, n_model))
        for j in range(n_model):
            for i in range(n_model):
                mu_ij[i,j] = trans[i,j] * mu[i] / c_bar[j]

        mixed_states = []
        mixed_covs = []
        # 模型0 CV(6维), 模型1 CA(9维)
        x_cv = self.filters[0].x
        P_cv = self.filters[0].P
        x_ca = self.filters[1].x
        P_ca = self.filters[1].P

        # ======= 混合得到CV的初始状态（来自CV、CA映射到6维）=======
        x_ca_6 = x_ca[:6,:]
        # CA协方差映射到6维
        P_ca_6 = P_ca[:6, :6]
        x0_mix = mu_ij[0,0] * x_cv + mu_ij[1,0] * x_ca_6
        P0_mix = mu_ij[0,0]*(P_cv + (x_cv - x0_mix)@(x_cv - x0_mix).T) + \
                 mu_ij[1,0]*(P_ca_6 + (x_ca_6 - x0_mix)@(x_ca_6 - x0_mix).T)

        # ======= 混合得到CA初始状态（CV扩展9维 + CA原始）=======
        x_cv_9 = np.vstack([x_cv, np.zeros((3,1))])
        P_cv_9 = np.zeros((9,9))
        P_cv_9[:6,:6] = P_cv
        x1_mix = mu_ij[0,1] * x_cv_9 + mu_ij[1,1] * x_ca
        P1_mix = mu_ij[0,1]*(P_cv_9 + (x_cv_9 - x1_mix)@(x_cv_9 - x1_mix).T) + \
                 mu_ij[1,1]*(P_ca + (x_ca - x1_mix)@(x_ca - x1_mix).T)

        mixed_states.append(x0_mix)
        mixed_covs.append(P0_mix)
        mixed_states.append(x1_mix)
        mixed_covs.append(P1_mix)
        return mixed_states, mixed_covs, c_bar

    def predict(self):
        """IMM Step2: 各模型独立预测（使用混合后的初值）"""
        mixed_x, mixed_P, _ = self._mix_state()
        # 载入混合初始值
        self.filters[0].x = mixed_x[0]
        self.filters[0].P = mixed_P[0]
        self.filters[1].x = mixed_x[1]
        self.filters[1].P = mixed_P[1]
        # 各自predict
        self.filters[0].predict()
        self.filters[1].predict()

    def correct(self, z):
        """IMM Step3: 滤波更新 + Step4: 更新模型概率"""
        z = np.array(z).reshape(3,1)
        # 子滤波器分别校正
        self.filters[0].correct(z)
        self.filters[1].update(z)

        # ========= 计算每个模型似然 =========
        def calc_likelihood(filter_inst, z_meas):
            z_pred = filter_inst.H @ filter_inst.x
            v = z_meas - z_pred
            S = filter_inst.H @ filter_inst.P @ filter_inst.H.T + filter_inst.R
            detS = np.linalg.det(S + 1e-10*np.eye(3))
            invS = np.linalg.inv(S + 1e-10*np.eye(3))
            ndim = z_meas.shape[0]
            lik = np.exp(-0.5 * v.T @ invS @ v) / np.sqrt((2*np.pi)**ndim * detS)
            return float(lik.item())

        lik0 = calc_likelihood(self.filters[0], z)
        lik1 = calc_likelihood(self.filters[1], z)
        lik = np.array([lik0, lik1])

        # 更新模型概率
        _, _, c_bar = self._mix_state()
        norm = np.sum(c_bar * lik) + 1e-12
        self.mu = (c_bar * lik) / norm

    def get_pos(self):
        """概率加权融合输出位置"""
        pos0 = self.filters[0].get_pos()
        pos1 = self.filters[1].get_pos()
        pos_fuse = self.mu[0] * pos0 + self.mu[1] * pos1
        return pos_fuse

    def get_model_prob(self):
        """返回当前模型概率 [mu_cv, mu_ca]"""
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