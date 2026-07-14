import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================== 3D CV卡尔曼滤波器 ======================
class CV3DKalmanFilter:
    def __init__(self, std_pos, std_vel):
        self.x = np.zeros((6, 1))
        self.P = np.diag(np.ones(6) * 1.0)
        self.std_pos = std_pos
        self.std_vel = std_vel
        self.dt = None

    def predict (self):
            self.x = self.F @ self.x

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

    def correct(self, z):
        z = np.array(z).reshape(3, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I6 = np.eye(6)
        self.P = (I6 - K @ self.H) @ self.P @ (I6 - K @ self.H).T + K @ self.R @ K.T

    def predict(self):
            self.x = self.F @ self.x

    def get_pos(self):
        return self.x[:3,0]

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

 #=================MIXING============================

class IMM_CVCA_3D:
        def __init__(self, std_pos, std_vel_cv, std_acc_ca, trans_mat=None):
            # 初始化两个独立滤波器
            self.cv_kf = CV3DKalmanFilter(std_pos, std_vel_cv)
            self.ca_kf = CA3DKalmanFilter(std_pos, std_acc_ca)

            # 模型数量
            self.M = 2
            # 马尔可夫转移矩阵 Pi[i,j] i→j
            if trans_mat is None:
                self.Pi = np.array([
                    [0.95, 0.05],
                    [0.05, 0.95]
                ])
            else:
                self.Pi = np.array(trans_mat)

            # 模型概率 mu[0]=CV, mu[1]=CA
            self.mu = np.array([0.5, 0.5])

            # 保存上一轮滤波结果
            self.x_cv6 = None
            self.P_cv6 = None
            self.x_ca9 = None
            self.P_ca9 = None

        def expand_cv_6to9(self, x6, P6):
            x9 = np.zeros((9, 1))
            x9[0:6] = x6
            P9 = np.zeros((9, 9))
            P9[0:6, 0:6] = P6
            P9[6:9, 6:9] = np.diag([3.0, 3.0, 3.0])
            return x9, P9

        def initialize(self, pos0, vel0):
            """
            初始化
            pos0: [x,y,z]
            vel0: [vx,vy,vz]
            """
            x6_0 = np.vstack([np.array(pos0).reshape(3, 1),
                              np.array(vel0).reshape(3, 1)])
            self.cv_kf.x = x6_0.copy()
            self.cv_kf.P = np.diag(np.ones(6) * 1.0)

            x9_0 = np.vstack([np.array(pos0).reshape(3, 1),
                              np.array(vel0).reshape(3, 1),
                              np.zeros((3, 1))])
            self.ca_kf.x = x9_0.copy()
            self.ca_kf.P = np.diag(np.ones(9) * 1.0)

            self.x_cv6 = self.cv_kf.x.copy()
            self.P_cv6 = self.cv_kf.P.copy()
            self.x_ca9 = self.ca_kf.x.copy()
            self.P_ca9 = self.ca_kf.P.copy()
            self.mu = np.array([0.5, 0.5])

        def step(self, dt, z):
            """
            dt: 当前帧时间间隔
            z: 观测 [x,y,z]
            return: pos_est, full9state, P9, mu
            """
            mu = self.mu
            # ========= 1. Mixing 交互融合（统一9维空间） =========
            # 上一轮状态升维
            x_cv9, P_cv9 = self.expand_cv_6to9(self.x_cv6, self.P_cv6)
            x_list9 = [x_cv9, self.x_ca9]
            P_list9 = [P_cv9, self.P_ca9]

            c_bar = self.Pi.T @ mu
            x_mix_9 = []
            P_mix_9 = []
            for j in range(self.M):
                x0j = np.zeros((9, 1))
                for i in range(self.M):
                    w = self.Pi[i, j] * mu[i] / c_bar[j]
                    x0j += w * x_list9[i]
                P0j = np.zeros((9, 9))
                for i in range(self.M):
                    w = self.Pi[i, j] * mu[i] / c_bar[j]
                    dx = x_list9[i] - x0j
                    P0j += w * (P_list9[i] + dx @ dx.T)
                x_mix_9.append(x0j)
                P_mix_9.append(P0j)

            # ========= 2. 分别送入CV、CA滤波预测更新 =========
            # CV滤波器：截取前6维
            x_mix_cv6 = x_mix_9[0][0:6, :]
            P_mix_cv6 = P_mix_9[0][0:6, 0:6]
            self.cv_kf.x = x_mix_cv6.copy()
            self.cv_kf.P = P_mix_cv6.copy()
            self.cv_kf.update_F_Q(dt)
            self.cv_kf.predict()
            self.cv_kf.correct(z)

            # CA滤波器：完整9维
            self.ca_kf.x = x_mix_9[1].copy()
            self.ca_kf.P = P_mix_9[1].copy()
            self.ca_kf.update_F_Q(dt)
            self.ca_kf.predict()
            self.ca_kf.update(z)

            # 保存本轮原始结果，供下一帧Mixing
            self.x_cv6 = self.cv_kf.x.copy()
            self.P_cv6 = self.cv_kf.P.copy()
            self.x_ca9 = self.ca_kf.x.copy()
            self.P_ca9 = self.ca_kf.P.copy()

            # ========= 3. 计算似然，更新模型概率 =========
            # 计算CV残差、S
            z_arr = np.array(z).reshape(3, 1)
            y_cv = z_arr - self.cv_kf.H @ self.cv_kf.x
            S_cv = self.cv_kf.H @ self.cv_kf.P @ self.cv_kf.H.T + self.cv_kf.R
            # 计算CA残差、S
            y_ca = z_arr - self.ca_kf.H @ self.ca_kf.x
            S_ca = self.ca_kf.H @ self.ca_kf.P @ self.ca_kf.H.T + self.ca_kf.R

            def gauss_likelihood(v, S):
                S_reg = S + 1e-6 * np.eye(3)
                detS = np.linalg.det(S_reg)
                invS = np.linalg.inv(S_reg)
                ll = np.exp(-0.5 * v.T @ invS @ v) / np.sqrt((2 * np.pi) ** 3 * detS)
                return float(ll.item())

            lik_cv = gauss_likelihood(y_cv, S_cv)
            lik_ca = gauss_likelihood(y_ca, S_ca)
            likelihood = np.array([lik_cv, lik_ca])

            mu_new = c_bar * likelihood
            mu_new = mu_new / np.sum(mu_new)
            self.mu = mu_new

            # ========= 4. 输出融合（9维统一加权） =========
            xcv9, _ = self.expand_cv_6to9(self.x_cv6, self.P_cv6)
            xca9 = self.x_ca9
            x_all9 = [xcv9, xca9]
            P_all9 = [self.expand_cv_6to9(self.x_cv6, self.P_cv6)[1], self.P_ca9]

            x_est9 = np.zeros((9, 1))
            for m in range(self.M):
                x_est9 += self.mu[m] * x_all9[m]
            P_est9 = np.zeros((9, 9))
            for m in range(self.M):
                dx = x_all9[m] - x_est9
                P_est9 += self.mu[m] * (P_all9[m] + dx @ dx.T)

            pos_est = x_est9[0:3, 0].ravel()
            return pos_est, x_est9, P_est9, self.mu

# ====================== 滑动窗口参数 ======================
obs_steps = 8    # 观测帧数
pred_steps = 5   # 预测帧数
dims = 3
stride= 1  # 重叠滑动步长
target_win_id = 0
save_window_metrics_csv = True

def sliding_imm_predict_overlap(seq, time_seq, obs_steps, pred_steps,
                                std_pos=0.05, std_vel_cv=0.2, std_acc_ca=0.1):
    N = len(seq)
    full_pred = np.zeros_like(seq)
    start = 0
    while True:
        end_obs = start + obs_steps
        end_pred = start + obs_steps + pred_steps

        if end_obs >= N:
            break

        # ========== 初始化IMM滤波器 ==========
        imm = IMM_CVCA_3D(std_pos=std_pos,
                          std_vel_cv=std_vel_cv,
                          std_acc_ca=std_acc_ca)
        init_pos = seq[start]
        init_vel = np.array([0.0, 0.0, 0.0])
        imm.initialize(pos0=init_pos, vel0=init_vel)
        full_pred[start] = seq[start]

        # ========== 观测段滤波 start ~ end_obs-1 (有观测z) ==========
        for i in range(start+1, end_obs):
            dt = time_seq[i] - time_seq[i-1]
            z = seq[i]
            pos_est, _, _, _ = imm.step(dt, z)
            full_pred[i] = pos_est

        # ========== 预测段 end_obs ~ end_pred-1 (无观测，仅预测外推) ==========
        current_t = time_seq[end_obs - 1]
        for j in range(end_obs, min(end_pred, N)):
            dt = time_seq[j] - current_t

            # =====关键===== IMM预测阶段：手动执行一轮Mixing + 各自predict，不做update
            mu = imm.mu
            x_cv9, P_cv9 = imm.expand_cv_6to9(imm.x_cv6, imm.P_cv6)
            x_list9 = [x_cv9, imm.x_ca9]
            P_list9 = [P_cv9, imm.P_ca9]

            c_bar = imm.Pi.T @ mu
            x_mix_9 = []
            P_mix_9 = []
            for jm in range(imm.M):
                x0j = np.zeros((9,1))
                for im in range(imm.M):
                    w = imm.Pi[im, jm] * mu[im] / c_bar[jm]
                    x0j += w * x_list9[im]
                P0j = np.zeros((9,9))
                for im in range(imm.M):
                    w = imm.Pi[im, jm] * mu[im] / c_bar[jm]
                    dx = x_list9[im] - x0j
                    P0j += w * (P_list9[im] + dx @ dx.T)
                x_mix_9.append(x0j)
                P_mix_9.append(P0j)

            # CV predict
            imm.cv_kf.x = x_mix_9[0][0:6, :].copy()
            imm.cv_kf.P = P_mix_9[0][0:6, 0:6].copy()
            imm.cv_kf.update_F_Q(dt)
            imm.cv_kf.predict()

            # CA predict
            imm.ca_kf.x = x_mix_9[1].copy()
            imm.ca_kf.P = P_mix_9[1].copy()
            imm.ca_kf.update_F_Q(dt)
            imm.ca_kf.predict()

            # 保存当前模型状态
            imm.x_cv6 = imm.cv_kf.x.copy()
            imm.P_cv6 = imm.cv_kf.P.copy()
            imm.x_ca9 = imm.ca_kf.x.copy()
            imm.P_ca9 = imm.ca_kf.P.copy()

            # 融合输出位置
            xcv9,_ = imm.expand_cv_6to9(imm.x_cv6, imm.P_cv6)
            x_est9 = imm.mu[0] * xcv9 + imm.mu[1] * imm.x_ca9
            pos_est = x_est9[0:3,0].ravel()
            full_pred[j] = pos_est

            current_t = time_seq[j]

        start += stride
        print(f"start={start}, 预测帧j={j}, 观测最后帧={end_obs - 1}, dt={dt:.3f}")
    return full_pred

def compute_pred_only_metrics(gt_win_full, pred_win_full, obs_steps):
    """
    gt_win_full: 当前窗口完整序列 [win_total,3]
    pred_win_full: 当前窗口完整序列 [win_total,3]
    obs_steps: 观测帧数，截取 obs_steps: 之后作为预测片段评估
    return rmse, ade, fde, rmse_x, rmse_y, rmse_z
    """
    # 截取【仅预测段】
    gt_pred = gt_win_full[obs_steps:]
    pred_pred = pred_win_full[obs_steps:]

    disp_err = np.linalg.norm(gt_pred - pred_pred, axis=1)
    rmse = np.sqrt(np.mean(np.sum((gt_pred - pred_pred) ** 2, axis=1)))
    ade = np.mean(disp_err)
    fde = disp_err[-1]

    rmse_x = np.sqrt(np.mean((gt_pred[:, 0] - pred_pred[:, 0]) ** 2))
    rmse_y = np.sqrt(np.mean((gt_pred[:, 1] - pred_pred[:, 1]) ** 2))
    rmse_z = np.sqrt(np.mean((gt_pred[:, 2] - pred_pred[:, 2]) ** 2))
    return rmse, ade, fde, rmse_x, rmse_y, rmse_z

# ====================== 主程序：CSV读取 + 全窗口遍历 ======================
if __name__ == "__main__":
    # -------------------------- 配置项（根据你的CSV修改）--------------------------
    csv_path = "data/mmaud_mavic3_gt_relative.csv"  # 你的CSV文件路径
    # 列名配置
    df = pd.read_csv(csv_path)
    timestamps = df["timestamp"].values
    gt_all = df[["x", "y", "z"]].values
    # 滤波参数
    std_pos = 0.3          # 位置观测噪声标准差（和你的CSV单位一致）
    std_vel_cv = 1.0       # CV模型加速度噪声标准差
    std_acc_ca = 0.1       # CA模型加加速度噪声标准差

    # -----------------------------------------------------------------------------

# IMM调用
    imm_traj = sliding_imm_predict_overlap(gt_all, timestamps, obs_steps, pred_steps,std_pos=0.05, std_vel_cv=1.0, std_acc_ca=0.1)

    # 计算指标
    rmse, ade, fde, rmse_x, rmse_y, rmse_z = compute_pred_only_metrics(gt_all, imm_traj, obs_steps)
    disp_err = np.linalg.norm(gt_all - imm_traj, axis=1)

    print("=" * 42)
    print(f"三维 IMM 滤波 | obs={obs_steps}, pred={pred_steps}")
    print("=" * 42)
    print(f"整体RMSE : {rmse:.6f} m")
    print(f"ADE      : {ade:.6f} m")
    print(f"FDE      : {fde:.6f} m")
    print("-" * 42)
    print(f"RMSE_X   : {rmse_x:.6f} m")
    print(f"RMSE_Y   : {rmse_y:.6f} m")
    print(f"RMSE_Z   : {rmse_z:.6f} m")
    print("=" * 42)

    # 保存汇总指标csv
    metrics_df = pd.DataFrame({
    "metric": ["RMSE_3D", "ADE", "FDE", "RMSE_X", "RMSE_Y", "RMSE_Z"],
    "value": [rmse, ade, fde, rmse_x, rmse_y, rmse_z],
    "unit": ["m", "m", "m", "m", "m", "m"]
})
    metrics_df.to_csv(r"C:\Users\86134\Desktop\drone\results\tables\IMM_kf_summary.csv", index=False)
    print("指标文件已保存: IMM_kf_summary.csv")

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

# 图1：3D轨迹
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')

    # 真值：散点（圆点）
    ax1.scatter(gt_all[:, 0], gt_all[:, 1], gt_all[:, 2], c="#ff3333", s=6, alpha=0.7, label="真值")
    # IMM轨迹：实线
    ax1.plot(imm_traj[:, 0], imm_traj[:, 1], imm_traj[:, 2], c="#0066ff", lw=0.9, label="IMM滤波")
    ax1.set_xlabel("X (m)", labelpad=8)
    ax1.set_ylabel("Y (m)", labelpad=8)
    ax1.set_zlabel("Z (m)", labelpad=8)
    ax1.set_title("3D轨迹 IMM", fontsize=13)
    ax1.legend();
    ax1.grid(alpha=0.3)

    fig1.savefig(r"C:\Users\86134\Desktop\drone\results\figures\imm_3d.png", dpi=150, bbox_inches="tight")
    plt.show()

  # 图2 X时序
    fig2 = plt.figure(figsize=(10,6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(timestamps, gt_all[:,0], "r-", label="真值X")
    ax2.plot(timestamps, imm_traj[:,0], "b--", label="IMM滤波X")
    ax2.set_title("X轴时序")
    ax2.legend(); ax2.grid()
    fig2.savefig(r"C:\Users\86134\Desktop\drone\results\figures\imm_x.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 图3 Y时序
    fig3 = plt.figure(figsize=(10,6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(timestamps, gt_all[:,1], "r-", label="真值Y")
    ax3.plot(timestamps, imm_traj[:,1], "b--", label="IMM滤波Y")
    ax3.set_title("Y轴时序")
    ax3.legend(); ax3.grid()
    fig3.savefig(r"C:\Users\86134\Desktop\drone\results\figures\imm_y.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 图4 Z时序
    fig4 = plt.figure(figsize=(10,6))
    ax4 = fig4.add_subplot(111)
    ax4.plot(timestamps, gt_all[:,2], "r-", label="真值Z")
    ax4.plot(timestamps, imm_traj[:,2], "b--", label="IMM滤波Z")
    ax4.set_title("Z轴时序")
    ax4.legend(); ax4.grid()
    fig4.savefig(r"C:\Users\86134\Desktop\drone\results\figures\imm_z.png", dpi=150, bbox_inches="tight")
    plt.show()