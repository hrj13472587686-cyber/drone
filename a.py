import numpy as np
import os
import matplotlib.pyplot as plt

# ====================== LinearPredictor 不变 ======================
class LinearPredictor:
    def __init__(self, std_pos=None):
        self.t0 = None
        self.tau_obs = []
        self.pos_obs = []
        self.coeff = None

    def init_state(self, pos, vel=None):
        self.t0 = None
        self.tau_obs.clear()
        self.pos_obs.clear()
        self.coeff = None

    def add_observation(self, time_stamp, pos):
        if self.t0 is None:
            self.t0 = time_stamp
        tau = time_stamp - self.t0
        self.tau_obs.append(tau)
        self.pos_obs.append(np.array(pos))

    def fit_model(self):                #最小二乘拟合
        if len(self.tau_obs) < 2:
            return False
        tau_arr = np.array(self.tau_obs)
        pos_mat = np.array(self.pos_obs)
        self.coeff = np.zeros((3, 2))
        X = np.vstack([tau_arr, np.ones_like(tau_arr)]).T
        for dim in range(3):
            y = pos_mat[:, dim]     # 提取第 dim 维的观测值，形状 (N,)
            sol, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self.coeff[dim] = sol       # sol = [k, b]
        return True

    def predict_at_time(self, target_t):
        success = self.fit_model()
        if not success:
            return self.pos_obs[-1].copy()
        tau_pred = target_t - self.t0
        pred = np.zeros(3)
        for dim in range(3):
            k, b = self.coeff[dim]
            pred[dim] = k * tau_pred + b
        return pred

    def get_pos(self):
        if len(self.pos_obs) == 0:
            return np.zeros(3)
        return self.pos_obs[-1].copy()

# ====================== CV3DKalmanFilter  ======================
class CV3DKalmanFilter:
    def __init__(self, std_pos, std_vel):
        self.x = np.zeros((6, 1))                   #状态向量
        self.P = np.diag(np.ones(6) * 1.0)          #状态估计误差协方差
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.dt = None

    def init_state(self, pos, vel=None):
        """初始化状态向量 [x,y,z,vx,vy,vz]^T"""
        pos = np.array(pos).reshape(3, 1)
        if vel is None:
            vel = np.zeros((3, 1))
        else:
            vel = np.array(vel).reshape(3, 1) # 关键：一维速度转列向量
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
        self.Q = np.array([                          #过程噪声协方差矩阵
            [q1, 0, 0, q2, 0, 0],
            [0, q1, 0, 0, q2, 0],
            [0, 0, q1, 0, 0, q2],
            [q2, 0, 0, q3, 0, 0],
            [0, q2, 0, 0, q3, 0],
            [0, 0, q2, 0, 0, q3]
        ])
        self.H = np.array([                         #观测矩阵
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        self.R = np.diag([self.std_pos**2, self.std_pos**2, self.std_pos**2])           #观察噪音

    def predict(self):
        """预测步：状态+协方差完整传播"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, z):
        z = np.array(z).reshape(3, 1)               #三维位置观测值
        y = z - self.H @ self.x                     #观察残差
        S = self.H @ self.P @ self.H.T + self.R     #新息协方差矩阵
        S += 1e-8 * np.eye(3)                       #防止奇异
        K = self.P @ self.H.T @ np.linalg.inv(S)    #卡尔曼增益计算
        self.x = self.x + K @ y                     #状态更新
        I6 = np.eye(6)
        self.P = (I6 - K @ self.H) @ self.P @ (I6 - K @ self.H).T + K @ self.R @ K.T
                                                    #状态估计误差协方差更新

    def get_pos(self):
        return self.x[:3,0]

# ====================== CA3DKalmanFilter =====================

class CA3DKalmanFilter:
    def __init__(self, std_pos, std_acc):
        # 状态向量：x,y,z, vx,vy,vz, ax,ay,az (9,1)
        self.x = np.zeros((9, 1))
        self.P = np.diag(np.ones(9) * 1.0)   # 初始协方差
        self.std_pos = std_pos                # 位置观测噪声标准差 [m]
        self.std_acc = std_acc                # 加速度过程噪声（jerk）标准差 [m/s^2.5]
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

        I3 = np.eye(3)  # 3×3 单位矩阵
        Z3 = np.zeros((3, 3))  # 3×3 零矩阵
        self.F = np.block([
            [I3, dt * I3, 0.5 * dt2 * I3],  # 位置
            [Z3, I3, dt * I3],  # 速度
            [Z3, Z3, I3]  # 加速度
        ])

        # 单轴过程噪声协方差 （来自 jerk(加加速度) 白噪声模型）
        sa2 = self.std_acc ** 2         #加速度过程噪声的方差强度
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
        self.H =np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])

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

# ====================== IMM3DFilter 转移概率对齐IMM_WIN ======================
class IMM3DFilter:
    def __init__(self, std_pos, std_vel_cv, std_acc_ca, trans_matrix=None):
        self.std_pos = std_pos
        self.filters = [
            CV3DKalmanFilter(std_pos, std_vel_cv),
            CA3DKalmanFilter(std_pos, std_acc_ca)
        ]
        self.mu = np.array([0.5, 0.5])       # 初始模型概率
        if trans_matrix is None:
            self.trans = np.array([         #马尔可夫转移矩阵
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
        mu = self.mu.copy()
        trans = self.trans
        c_bar = trans.T @ mu
        # 防除零，极小值兜底
        c_bar[c_bar < 1e-12] = 1e-12
        mu_ij = np.zeros_like(trans)        #计算混合权重
        for j in range(2):
            for i in range(2):
                mu_ij[i,j] = trans[i,j] * mu[i] / c_bar[j]
        # 提取各滤波器状态和协方差
        x_cv = self.filters[0].x.copy()         # CV 滤波器上一时刻状态 (6,1)
        P_cv = self.filters[0].P.copy()
        x_ca = self.filters[1].x.copy()         # CA 滤波器状态 (9,1)
        P_ca = self.filters[1].P.copy()
        x_ca_6 = x_ca[:6, :].copy()         # 取CA的前6维（位置+速度），丢弃加速度
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
        self.filters[0].x = mixed_x[0]      #cv状态混合
        self.filters[0].P = mixed_P[0]      #cv协方差混合
        self.filters[1].x = mixed_x[1]      #ca状态混合
        self.filters[1].P = mixed_P[1]      #ca协方差混合
        self.filters[0].predict()           #独立预测
        self.filters[1].predict()

    def correct(self, z):
        def calc_log_like(filter_inst, z_meas):       #计算给定滤波器在观测计算给定滤波器在观测z下的对数似然
            z_pred = filter_inst.H @ filter_inst.x
            v = z_meas - z_pred                                                     #新息
            S = filter_inst.H @ filter_inst.P @ filter_inst.H.T + filter_inst.R     #新息协方差矩阵
            S += 1e-10 * np.eye(3)
            detS = np.linalg.det(S)                                                 #计算行列式S
            detS = max(detS, 1e-15)                                            #防止detS为0
            invS = np.linalg.inv(S)                                             #计算S^-1
            quad = (v.T @ invS @ v).item()                                  #计算二次型 v^T S^-1 v,马氏距离的平方
            log_det = np.log(detS)                                          #logS
            log_lik = -0.5 * (quad + 3 * np.log(2 * np.pi) + log_det)       #对数似然值，表征在当前模型预测下，观测 z 出现的合理程度
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
        self.mu = np.array([L0 * self.c_bar[0], L1 * self.c_bar[1]])     #c_bar是预测边缘概率,mu更新边缘概率
        self.mu /= self.mu.sum()        #归一化处理



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
# ====================== 指标函数不变 ======================
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
    rmse = np.sqrt(np.mean(np.sum(err**2, axis=1)))     #均方根误差
    ade = np.mean(disp_err)                             #平均位移误差
    fde = disp_err[-1]                                  #最终位移误差
    rmse_x = np.sqrt(np.mean(err[:,0]**2))
    rmse_y = np.sqrt(np.mean(err[:,1]**2))
    rmse_z = np.sqrt(np.mean(err[:,2]**2))
    return rmse, ade, fde, rmse_x, rmse_y, rmse_z

# ====================== 核心修改：rebuild_win 对齐IMM_WIN逻辑 ======================
def rebuild_win(win_gt, win_time, obs_steps, std_pos, std_vel, std_acc, mode):
    N = len(win_gt)
    pred = np.zeros_like(win_gt)
    pred[0] = win_gt[0]

    if mode == "LR":
        f = LinearPredictor()
        f.init_state(win_gt[0])
        f.add_observation(win_time[0], win_gt[0])
    elif mode == "CV":
        f = CV3DKalmanFilter(std_pos, std_vel)
        # 计算初始速度
        if len(win_gt) > 1:
            dt0 = win_time[1] - win_time[0]
            v0 = (win_gt[1] - win_gt[0]) / dt0
            f.init_state(pos=win_gt[0], vel=v0)
        else:
            f.init_state(pos=win_gt[0])
    elif mode == "CA":
        f = CA3DKalmanFilter(std_pos, std_acc)
        if len(win_gt) > 1:
            dt0 = win_time[1] - win_time[0]
            v0 = (win_gt[1] - win_gt[0]) / dt0
            f.init_state(pos=win_gt[0], vel=v0)
        else:
            f.init_state(pos=win_gt[0])
    else: # IMM
        f = IMM3DFilter(std_pos, std_vel, std_acc)
        # 和IMM_WIN完全一致：窗口前两帧求初始速度
        if len(win_gt) > 1:
            dt0 = win_time[1] - win_time[0]
            v0 = (win_gt[1] - win_gt[0]) / dt0
            f.init_state(pos=win_gt[0], vel=v0)
        else:
            f.init_state(pos=win_gt[0])

    # 观测段循环：先update_F_Q，再predict无参，再校正
    for i in range(1, obs_steps):
        dt = win_time[i] - win_time[i-1]
        if mode == "LR":
            f.add_observation(win_time[i], win_gt[i])
        else:
            f.update_F_Q(dt)
            f.predict()
            if mode == "CV":
                f.correct(win_gt[i])
            elif mode == "CA":
                f.correct(win_gt[i])
            elif mode == "IMM":
                f.correct(win_gt[i])
        pred[i] = f.get_pos()

    # 预测段：先更新F/Q，再predict无参
    current_t = win_time[obs_steps - 1]
    for i in range(obs_steps, N):

        t_target = win_time[i]
        if mode == "LR":
            pred[i] = f.predict_at_time(t_target)
        else:
            dt = win_time[i] - current_t
            f.update_F_Q(dt)
            f.predict()
            pred[i] = f.get_pos()
            current_t = win_time[i]
    return pred

# ====================== 主函数不变 ======================
if __name__ == "__main__":
    obs_steps = 8
    pred_steps = 5
    stride = 1
    std_pos = 0.05
    std_vel_cv = 0.1
    std_acc_ca = 0.1

    os.makedirs("results/tables", exist_ok=True)
    fig_save_path = r"results/figures/mmaud_mavic3_gt_window_straight.png"
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

    target_win_idx = 236
    win_total_len = obs_steps + pred_steps
    start = target_win_idx * stride
    end = start + win_total_len

    if end > len(gt_all):
        print("窗口超出数据范围！")
        exit()

    win_gt = gt_all[start:end]
    win_time = t_all[start:end]

    win_lr  = rebuild_win(win_gt, win_time, obs_steps, std_pos, std_vel_cv, std_acc_ca, "LR")
    win_cv  = rebuild_win(win_gt, win_time, obs_steps, std_pos, std_vel_cv, std_acc_ca, "CV")
    win_ca  = rebuild_win(win_gt, win_time, obs_steps, std_pos, std_vel_cv, std_acc_ca, "CA")
    win_imm = rebuild_win(win_gt, win_time, obs_steps, std_pos, std_vel_cv, std_acc_ca, "IMM")

    met_lr  = compute_pred_only_metrics(win_gt, win_lr, obs_steps)
    met_cv  = compute_pred_only_metrics(win_gt, win_cv, obs_steps)
    met_ca  = compute_pred_only_metrics(win_gt, win_ca, obs_steps)
    met_imm = compute_pred_only_metrics(win_gt, win_imm, obs_steps)

    print("===== 单窗口对比指标 =====")
    names = ["LR","CV","CA","IMM"]
    mets  = [met_lr, met_cv, met_ca, met_imm]
    for n,m in zip(names,mets):
        print(f"{n:4s} | RMSE:{m[0]:.4f} ADE:{m[1]:.4f} FDE:{m[2]:.4f}")

    # 绘图透明度保留原有设置
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    obs_gt = win_gt[:obs_steps]
    fut_gt = win_gt[obs_steps:]

    ax.plot(obs_gt[:, 0], obs_gt[:, 1], obs_gt[:, 2], c="orange", lw=3.7, label="history")
    ax.scatter(obs_gt[:, 0], obs_gt[:, 1], obs_gt[:, 2], c="orange", s=35)
    ax.plot(fut_gt[:, 0], fut_gt[:, 1], fut_gt[:, 2], c="blue", lw=0.7, label="future")
    ax.scatter(fut_gt[:, 0], fut_gt[:, 1], fut_gt[:, 2], c="blue", s=35)

    ax.plot(win_lr[:, 0], win_lr[:, 1], win_lr[:, 2], "r-", lw=0.5 , alpha=0.5)
    ax.scatter(win_lr[:, 0], win_lr[:, 1], win_lr[:, 2], c="red", s=5, label="LR ")
    ax.plot(win_cv[:, 0], win_cv[:, 1], win_cv[:, 2], "g-", lw=1.5 , alpha=0.5)
    ax.scatter(win_cv[:, 0], win_cv[:, 1], win_cv[:, 2], c="green", s=5, label="CV ")
    ax.plot(win_ca[:, 0], win_ca[:, 1], win_ca[:, 2], "y-", lw=1.5 , alpha=0.5)
    ax.scatter(win_ca[:, 0], win_ca[:, 1], win_ca[:, 2], c="yellow", s=5, label="CA ")
    ax.plot(win_imm[:, 0], win_imm[:, 1], win_imm[:, 2], "#660099", lw=1.5, alpha=0.5)
    ax.scatter(win_imm[:, 0], win_imm[:, 1], win_imm[:, 2], c="#660099", s=5, label="IMM ")

    ax.set_xlabel("X(m)")
    ax.set_ylabel("Y(m)")
    ax.set_zlabel("Z(m)")
    ax.set_title(f" | obs={obs_steps} pred={pred_steps} | window={target_win_idx}")
    ax.legend(fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    plt.show()