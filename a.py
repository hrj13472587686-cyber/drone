
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 线性回归 LR（原版无BUG） ======================
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

    def fit_model(self):
        if len(self.tau_obs) < 2:
            return False
        tau_arr = np.array(self.tau_obs)
        pos_mat = np.array(self.pos_obs)
        self.coeff = np.zeros((3, 2))
        X = np.vstack([tau_arr, np.ones_like(tau_arr)]).T
        for dim in range(3):
            y = pos_mat[:, dim]
            sol, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self.coeff[dim] = sol
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

# ====================== CV 匀速模型 6维（无BUG） ======================
class CVKalman:
    def __init__(self, std_pos, std_vel):
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.x = None
        self.P = None
        self.H = np.block([np.eye(3), np.zeros((3,3))])

    def init_state(self, pos, vel=None):
        if vel is None:
            vel = np.zeros(3)
        self.x = np.hstack([pos, vel]).reshape(6,1)
        self.P = np.diag([self.std_pos**2]*3 + [self.std_vel**2]*3)

    def predict(self, dt):
        F = np.block([
            [np.eye(3), dt*np.eye(3)],
            [np.zeros((3,3)), np.eye(3)]
        ])
        Q = np.diag([0]*3 + [self.std_vel**2 * dt**2]*3)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        z = z.reshape(3,1)
        R = np.eye(3) * self.std_pos**2
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_pos(self):
        return self.x[:3,0].copy()

# ====================== CA 匀加速模型【维度BUG彻底修复：严格9维】 ======================
class CAKalman:
    def __init__(self, std_pos, std_vel, std_acc):
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.std_acc = std_acc
        self.x = None
        self.P = None
        # 观测仅位置：3x9
        self.H = np.block([np.eye(3), np.zeros((3,6))])

    def init_state(self, pos, vel=None, acc=None):
        if vel is None: vel = np.zeros(3)
        if acc is None: acc = np.zeros(3)
        # 固定9维状态 [x,y,z,vx,vy,vz,ax,ay,az]
        self.x = np.hstack([pos, vel, acc]).reshape(9,1)
        self.P = np.diag([self.std_pos**2]*3 + [self.std_vel**2]*3 + [self.std_acc**2]*3)

    def predict(self, dt):
        # 严格9维状态转移矩阵
        dt2 = 0.5 * dt**2
        F_block = np.array([
            [1, dt, dt2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        F = np.kron(F_block, np.eye(3))
        # 过程噪声
        Q_block = np.array([
            [dt**4/4, dt**3/2, dt**2/2],
            [dt**3/2, dt**2, dt],
            [dt**2/2, dt, 1]
        ]) * self.std_acc**2
        Q = np.kron(Q_block, np.eye(3))

        # 核心修复：9维 × 9维，维度完全匹配
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        z = z.reshape(3,1)
        R = np.eye(3) * self.std_pos**2
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P

    def get_pos(self):
        return self.x[:3,0].copy()

# ====================== IMM 融合模型（适配修复后的CA/CV维度） ======================
class IMMFilter:
    def __init__(self, std_pos, std_vel, std_acc):
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.std_acc = std_acc
        self.cv = CVKalman(std_pos, std_vel)
        self.ca = CAKalman(std_pos, std_vel, std_acc)
        self.mu = np.array([0.5, 0.5])
        self.trans_mat = np.array([
            [0.95, 0.05],
            [0.05, 0.95]
        ])

    def init_state(self, pos, vel=None):
        self.cv.init_state(pos, vel)
        self.ca.init_state(pos, vel)

    def predict(self, dt):
        # IMM预测步：先更新模型概率
        self.mu = self.trans_mat.T @ self.mu
        # 子滤波器各自独立向前预测，不做状态混合（规避6维/9维冲突）
        self.cv.predict(dt)
        self.ca.predict(dt)

    def update(self, z):
        self.cv.update(z)
        self.ca.update(z)
        # 简易似然更新权重
        err_cv = z - self.cv.get_pos()
        err_ca = z - self.ca.get_pos()
        l1 = np.exp(-0.5 * np.sum(err_cv**2))
        l2 = np.exp(-0.5 * np.sum(err_ca**2))
        self.mu *= np.array([l1, l2])
        self.mu = self.mu / (np.sum(self.mu)+1e-10)

    def get_pos(self):
        return self.mu[0] * self.cv.get_pos() + self.mu[1] * self.ca.get_pos()


# ====================== 指标函数 ======================
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

# ====================== 窗口重建通用函数（适配所有模型） ======================
def rebuild_win(win_gt, win_time, obs_steps, std_pos, std_vel, std_acc, mode):
    N = len(win_gt)
    pred = np.zeros_like(win_gt)
    pred[0] = win_gt[0]

    if mode == "LR":
        f = LinearPredictor()
    elif mode == "CV":
        f = CVKalman(std_pos, std_vel)
    elif mode == "CA":
        f = CAKalman(std_pos, std_vel, std_acc)
    else:
        f = IMMFilter(std_pos, std_vel, std_acc)

    f.init_state(win_gt[0])
    if mode == "LR":
        f.add_observation(win_time[0], win_gt[0])

    # 观测段
    for i in range(1, obs_steps):
        dt = win_time[i] - win_time[i-1]
        if mode == "LR":
            f.add_observation(win_time[i], win_gt[i])
        else:
            f.predict(dt)
            f.update(win_gt[i])
        pred[i] = f.get_pos()

    # 预测段（重点分支修复）
    for i in range(obs_steps, N):
        t_target = win_time[i]
        if mode == "LR":
            pred[i] = f.predict_at_time(t_target)
        else:
            dt = win_time[i] - win_time[i-1]
            f.predict(dt)
            pred[i] = f.get_pos()
    return pred
# ====================== 主函数 ======================
if __name__ == "__main__":
    # 超参
    obs_steps = 8
    pred_steps = 5
    stride = 1
    std_pos = 0.05
    std_vel_cv = 0.1
    std_acc_ca = 0.02

    os.makedirs("results/tables", exist_ok=True)
    fig_save_path = r"/results/figures/window.png"
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

    # 选择可视化窗口
    target_win_idx = 700
    win_total_len = obs_steps + pred_steps
    start = target_win_idx * stride
    end = start + win_total_len

    if end > len(gt_all):
        print("窗口超出数据范围！")
        exit()

    win_gt = gt_all[start:end]
    win_time = t_all[start:end]

    # 四类模型预测
    win_lr  = rebuild_win(win_gt, win_time, obs_steps, std_pos, std_vel_cv, std_acc_ca, "LR")
    win_cv  = rebuild_win(win_gt, win_time, obs_steps, std_pos, std_vel_cv, std_acc_ca, "CV")
    win_ca  = rebuild_win(win_gt, win_time, obs_steps, std_pos, std_vel_cv, std_acc_ca, "CA")
    win_imm = rebuild_win(win_gt, win_time, obs_steps, std_pos, std_vel_cv, std_acc_ca, "IMM")

    # 指标计算
    met_lr  = compute_pred_only_metrics(win_gt, win_lr, obs_steps)
    met_cv  = compute_pred_only_metrics(win_gt, win_cv, obs_steps)
    met_ca  = compute_pred_only_metrics(win_gt, win_ca, obs_steps)
    met_imm = compute_pred_only_metrics(win_gt, win_imm, obs_steps)

    print("===== 单窗口对比指标 =====")
    names = ["LR","CV","CA","IMM"]
    mets  = [met_lr, met_cv, met_ca, met_imm]
    for n,m in zip(names,mets):
        print(f"{n:4s} | RMSE:{m[0]:.4f} ADE:{m[1]:.4f} FDE:{m[2]:.4f}")

    # 绘图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')



    # 拆分当前窗口观测/预测段
    obs_gt = win_gt[:obs_steps]
    fut_gt = win_gt[obs_steps:]

    # 观测段真值（橙色）
    ax.plot(obs_gt[:, 0], obs_gt[:, 1], obs_gt[:, 2], c="orange", lw=2.7, label="观测段真值")
    ax.scatter(obs_gt[:, 0], obs_gt[:, 1], obs_gt[:, 2], c="orange", s=35)

    # 未来预测段真值（黑色）
    ax.plot(fut_gt[:, 0], fut_gt[:, 1], fut_gt[:, 2], c="black", lw=0.7, label="预测段真值")
    ax.scatter(fut_gt[:, 0], fut_gt[:, 1], fut_gt[:, 2], c="black", s=35)

    # 四类模型预测曲线
    ax.plot(win_lr[:, 0], win_lr[:, 1], win_lr[:, 2], "r-", lw=0.5, )
    ax.scatter(win_lr[:, 0], win_lr[:, 1], win_lr[:, 2], c="red", s=5, label="LR 线性回归")
    ax.plot(win_cv[:, 0], win_cv[:, 1], win_cv[:, 2], "g-", lw=1.5, )
    ax.scatter(win_cv[:, 0], win_cv[:, 1], win_cv[:, 2], c="green", s=5, label="CV 恒速卡尔曼")
    ax.plot(win_ca[:, 0], win_ca[:, 1], win_ca[:, 2], "b-", lw=1.5)
    ax.scatter(win_ca[:, 0], win_ca[:, 1], win_ca[:, 2], c="blue", s=5, label="CA ")
    ax.plot(win_imm[:, 0], win_imm[:, 1], win_imm[:, 2], "p-", lw=1.5, label="IMM ")
    ax.scatter(win_imm[:, 0], win_imm[:, 1], win_imm[:, 2], c="#9922bb", s=5)


    ax.set_xlabel("X(m)")
    ax.set_ylabel("Y(m)")
    ax.set_zlabel("Z(m)")
    ax.set_title(f"多模型轨迹预测对比 | obs={obs_steps} pred={pred_steps} | window={target_win_idx}")
    ax.legend(fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    plt.show()
