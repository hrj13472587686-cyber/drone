import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_single_pred_window(gt_all, kf_full_traj, win_start, obs_len, pred_len, view_3d=False):
    """
    gt_all: [T,3] 完整真值
    kf_full_traj: [T,3] KF整条输出轨迹
    win_start: 当前窗口起始索引
    obs_len: 历史观测帧数
    pred_len: 预测帧数
    view_3d: True绘制3D轨迹，False绘制XY平面2D
    配色规则：
        窗口外：灰色
        history观测段：橙色
        future预测段：蓝色
    """
    T_total = gt_all.shape[0]
    t_arr = np.arange(T_total)

    # 1. 区间边界
    hist_end = win_start + obs_len
    pred_end = hist_end + pred_len

    # 2. 生成三段掩码
    mask_outside = (t_arr < win_start) | (t_arr >= pred_end)  # 窗口外 → 灰色
    mask_history = (t_arr >= win_start) & (t_arr < hist_end)  # history → 橙色
    mask_future = (t_arr >= hist_end) & (t_arr < pred_end)    # future → 蓝色

    # 安全截断，防止越界
    pred_end = min(pred_end, T_total)

    if view_3d:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        # 灰色：窗口外完整真值
        ax.plot(gt_all[mask_outside, 0], gt_all[mask_outside, 1], gt_all[mask_outside, 2],
                c="gray", lw=1, alpha=0.35)
        # 橙色 History
        ax.plot(gt_all[mask_history, 0], gt_all[mask_history, 1], gt_all[mask_history, 2],
                c="orange", lw=2, label="GT History")
        ax.plot(kf_full_traj[mask_history, 0], kf_full_traj[mask_history, 1], kf_full_traj[mask_history, 2],
                c="darkorange", lw=1.6, ls="--", label="KF History")
        # 蓝色 Future
        ax.plot(gt_all[mask_future, 0], gt_all[mask_future, 1], gt_all[mask_future, 2],
                c="royalblue", lw=2, label="GT Future")
        ax.plot(kf_full_traj[mask_future, 0], kf_full_traj[mask_future, 1], kf_full_traj[mask_future, 2],
                c="blue", lw=1.6, ls="--", label="KF Predict")
        # 分界圆点标记
        ax.plot(gt_all[hist_end,0], gt_all[hist_end,1], gt_all[hist_end,2], "ko", markersize=5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # ax.view_init(elev=25, azim=50)
    else:
        fig, ax = plt.subplots(figsize=(9, 6))
        # 灰色：窗口外
        ax.plot(gt_all[mask_outside, 0], gt_all[mask_outside, 1],
                c="gray", lw=1, alpha=0.35)
        # History 橙色
        ax.plot(gt_all[mask_history, 0], gt_all[mask_history, 1],
                c="orange", lw=2, label="GT History")
        ax.plot(kf_full_traj[mask_history, 0], kf_full_traj[mask_history, 1],
                c="darkorange", lw=1.6, ls="--", label="KF History")
        # Future 蓝色
        ax.plot(gt_all[mask_future, 0], gt_all[mask_future, 1],
                c="royalblue", lw=2, label="GT Future")
        ax.plot(kf_full_traj[mask_future, 0], kf_full_traj[mask_future, 1],
                c="blue", lw=1.6, ls="--", label="KF Predict")
        # 分界圆点
        ax.plot(gt_all[hist_end,0], gt_all[hist_end,1], "ko", markersize=5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")

    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.title(f"Prediction Window start={win_start}")
    plt.tight_layout()
    plt.savefig(f"C:\\Users\\86134\\Desktop\\drone\\results\\figures\\window_{win_start}.png", dpi=300, bbox_inches="tight")
    plt.show()

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

# ====================== 评估指标 RMSE / ADE / FDE ======================
def compute_metrics(gt, pred):
    disp_err = np.linalg.norm(gt - pred, axis=1)
    rmse = np.sqrt(np.mean(np.sum((gt - pred) ** 2, axis=1)))
    ade = np.mean(disp_err)
    fde = disp_err[-1]
    rmse_x = np.sqrt(np.mean((gt[:,0]-pred[:,0])**2))
    rmse_y = np.sqrt(np.mean((gt[:,1]-pred[:,1])**2))
    rmse_z = np.sqrt(np.mean((gt[:,2]-pred[:,2])**2))
    return rmse, ade, fde, rmse_x, rmse_y, rmse_z

# ====================== 滑动窗口参数 ======================
obs_steps = 8    # 观测帧数
pred_steps = 5   # 预测帧数
dims = 3

def sliding_ca_predict(seq, time_seq, obs_steps, pred_steps, std_pos=0.05, std_acc=0.1):
    N = len(seq)
    full_pred = np.zeros_like(seq)
    window_len = obs_steps + pred_steps
    for start in range(0, N, window_len):
        end_obs = start + obs_steps
        end_pred = end_obs + pred_steps
        if end_obs > N:
            break
        kf = CA3DKalmanFilter(std_pos, std_acc)
        kf.x[0,0] = seq[start,0]
        kf.x[1,0] = seq[start,1]
        kf.x[2,0] = seq[start,2]
        full_pred[start] = seq[start]
        # 观测窗口更新
        for i in range(start+1, end_obs):
            dt = time_seq[i] - time_seq[i-1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.update(seq[i])
            full_pred[i] = kf.get_pos()
        # 纯预测无观测
        current_t = time_seq[end_obs - 1]
        for j in range(end_obs, min(end_pred, N)):
            dt = time_seq[j] - current_t
            kf.update_F_Q(dt)
            kf.predict()
            full_pred[j] = kf.get_pos()
            current_t = time_seq[j]
    # 尾部剩余数据单独滤波
    last_start = (N // window_len) * window_len
    if last_start < N:
        kf = CA3DKalmanFilter(std_pos, std_acc)
        kf.x[0,0] = seq[last_start,0]
        kf.x[1,0] = seq[last_start,1]
        kf.x[2,0] = seq[last_start,2]
        full_pred[last_start] = seq[last_start]
        for i in range(last_start+1, N):
            dt = time_seq[i] - time_seq[i-1]
            kf.update_F_Q(dt)
            kf.predict()
            kf.update(seq[i])
            full_pred[i] = kf.get_pos()
    return full_pred

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    csv_path = "mmaud_mavic3_gt_relative.csv"
    df = pd.read_csv(csv_path)
    timestamps = df["timestamp"].values
    gt_all = df[["x", "y", "z"]].values

    # 运行CA-KF滑动窗口滤波
    kf_traj = sliding_ca_predict(gt_all, timestamps, obs_steps, pred_steps, std_pos=0.05, std_acc=0.1)

    # 计算指标
    rmse, ade, fde, rmse_x, rmse_y, rmse_z = compute_metrics(gt_all, kf_traj)
    disp_err = np.linalg.norm(gt_all - kf_traj, axis=1)

    print("=" * 42)
    print(f"三维 CA-KF 匀加速滤波 | obs={obs_steps}, pred={pred_steps}")
    print("=" * 42)
    print(f"整体RMSE : {rmse:.6f} m")
    print(f"ADE      : {ade:.6f} m")
    print(f"FDE      : {fde:.6f} m")
    print("-" * 42)
    print(f"RMSE_X   : {rmse_x:.6f} m")
    print(f"RMSE_Y   : {rmse_y:.6f} m")
    print(f"RMSE_Z   : {rmse_z:.6f} m")
    print("=" * 42)


    # 保存指标CSV
    metrics_df = pd.DataFrame({
        "metric": ["RMSE_3D", "ADE", "FDE", "RMSE_X", "RMSE_Y", "RMSE_Z"],
        "value": [rmse, ade, fde, rmse_x, rmse_y, rmse_z],
        "unit": ["m", "m", "m", "m", "m", "m"]
    })
    metrics_df.to_csv(r"C:\Users\86134\Desktop\drone\results\tables\ca_kf_summary.csv", index=False)
    print("指标文件已保存: ca_kf_summary.csv")

    # ====================== 独立分开绘图 ======================
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


    # 指定窗口切片
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    def plot_single_pred_window(gt_all, kf_full_traj, win_start:{236}, obs_len, pred_len, view_3d=False):
        """
        gt_all: [T,3] 完整真值
        kf_full_traj: [T,3] KF整条输出轨迹
        win_start: 当前窗口起始索引
        obs_len: 历史观测帧数
        pred_len: 预测帧数
        view_3d: True绘制3D轨迹，False绘制XY平面2D
        """
        T_total = gt_all.shape[0]
        t_arr = np.arange(T_total)

        # 1. 区间边界
        hist_end = win_start + obs_len
        pred_end = hist_end + pred_len

        # 2. 生成三段掩码
        mask_outside = (t_arr < win_start) | (t_arr >= pred_end)  # 窗口外 → 灰色
        mask_history = (t_arr >= win_start) & (t_arr < hist_end)  # history → 橙色
        mask_future = (t_arr >= hist_end) & (t_arr < pred_end)  # future → 蓝色

        # 安全截断，防止越界
        pred_end = min(pred_end, T_total)

        if view_3d:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            # 灰色：窗口外完整真值
            ax.plot(gt_all[mask_outside, 0], gt_all[mask_outside, 1], gt_all[mask_outside, 2],
                    c="gray", lw=1, alpha=0.35)
            # 橙色 History
            ax.plot(gt_all[mask_history, 0], gt_all[mask_history, 1], gt_all[mask_history, 2],
                    c="orange", lw=2, label="GT History")
            ax.plot(kf_full_traj[mask_history, 0], kf_full_traj[mask_history, 1], kf_full_traj[mask_history, 2],
                    c="darkorange", lw=1.6, ls="--", label="KF History")
            # 蓝色 Future
            ax.plot(gt_all[mask_future, 0], gt_all[mask_future, 1], gt_all[mask_future, 2],
                    c="royalblue", lw=2, label="GT Future")
            ax.plot(kf_full_traj[mask_future, 0], kf_full_traj[mask_future, 1], kf_full_traj[mask_future, 2],
                    c="blue", lw=1.6, ls="--", label="KF Predict")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            # 如需固定视角取消下一行注释
            # ax.view_init(elev=25, azim=50)
        else:
            fig, ax = plt.subplots(figsize=(9, 6))
            # 灰色：窗口外
            ax.plot(gt_all[mask_outside, 0], gt_all[mask_outside, 1],
                    c="gray", lw=1, alpha=0.35)
            # History 橙色
            ax.plot(gt_all[mask_history, 0], gt_all[mask_history, 1],
                    c="orange", lw=2, label="GT History")
            ax.plot(kf_full_traj[mask_history, 0], kf_full_traj[mask_history, 1],
                    c="darkorange", lw=1.6, ls="--", label="KF History")
            # Future 蓝色
            ax.plot(gt_all[mask_future, 0], gt_all[mask_future, 1],
                    c="royalblue", lw=2, label="GT Future")
            ax.plot(kf_full_traj[mask_future, 0], kf_full_traj[mask_future, 1],
                    c="blue", lw=1.6, ls="--", label="KF Predict")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.axis("equal")

        # ===================== 调用窗口绘图（重点！）=====================
        WIN_START = 236  # 指定窗口起始索引
        # 绘制2D XY俯视图
        plot_single_pred_window(gt_all, kf_traj, win_start=WIN_START, obs_len=obs_steps, pred_len=pred_steps,
                                view_3d=False)
        # 如需3D轨迹打开下面一行
        plot_single_pred_window(gt_all, kf_traj, win_start=WIN_START, obs_len=obs_steps, pred_len=pred_steps, view_3d=True)
