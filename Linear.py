import numpy as np
import os
import matplotlib.pyplot as plt

# ====================== 线性回归预测器（仅保留此模型） ======================
class LinearPredictor:
    def __init__(self, std_pos=None):
        self.t0 = None        # 当前窗口第一条观测的时间（时间基准原点）
        self.tau_obs = []     # 存储相对时间 τ = t - t0
        self.pos_obs = []     # 存储对应观测位置 [x,y,z]
        self.coeff = None     # 拟合系数 shape(3,2) 每行 [k, b]

    def init_state(self, pos, vel=None):
        """每个滑动窗口开始时调用，清空缓存，初始化预测器"""
        self.t0 = None
        self.tau_obs.clear()
        self.pos_obs.clear()
        self.coeff = None

    def add_observation(self, time_stamp, pos):
        """新增一组观测 (绝对时间, 三维坐标)"""
        if self.t0 is None:
            self.t0 = time_stamp   # 第一条数据定为时间零点
        tau = time_stamp - self.t0  # 转为窗口局部相对时间
        self.tau_obs.append(tau)
        self.pos_obs.append(np.array(pos))

    def fit_model(self):
        """最小二乘求解 k,b；返回是否拟合成功"""
        if len(self.tau_obs) < 2:
            return False   # 至少2个点才能拟合直线

        tau_arr = np.array(self.tau_obs)
        pos_mat = np.array(self.pos_obs)  # [N,3]

        self.coeff = np.zeros((3, 2))
        # 构造设计矩阵 X = [τ , 1]
        X = np.vstack([tau_arr, np.ones_like(tau_arr)]).T

        # x/y/z 三个维度分别独立线性回归
        for dim in range(3):
            y = pos_mat[:, dim]
            # lstsq最小二乘：min||X·w - y||²
            sol, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self.coeff[dim] = sol  # sol=[k, b]
        return True

    def predict_at_time(self, target_t):
        """输入绝对时刻，输出该时刻预测坐标"""
        success = self.fit_model()
        if not success:
            # 数据不足无法拟合直线，直接返回最后观测值
            return self.pos_obs[-1].copy()

        tau_pred = target_t - self.t0
        pred = np.zeros(3)
        for dim in range(3):
            k, b = self.coeff[dim]
            pred[dim] = k * tau_pred + b
        return pred

    def get_pos(self):
        """兼容旧滤波器接口，返回最新观测位置（占位函数）"""
        if len(self.pos_obs) == 0:
            return np.zeros(3)
        return self.pos_obs[-1].copy()

# -------------------------- 指标函数 --------------------------
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

# -------------------------- 重叠预测轨迹生成函数（LR专用） --------------------------
def sliding_predict_overlap(seq, time_seq, obs_steps, pred_steps,
                            filter_cls, std_pos, stride=1):
    N = len(seq)
    full_pred = np.zeros_like(seq)
    pred_cnt = np.zeros_like(seq)

    start = 0
    while True:
        end_obs = start + obs_steps
        end_pred = start + obs_steps + pred_steps
        if end_obs >= N:
            break

        kf = LinearPredictor(std_pos)
        kf.init_state(pos=seq[start])

        for i in range(start, end_obs):
            current_time = time_seq[i]
            if i == start:
                full_pred[i] += seq[i]
                pred_cnt[i] += 1
                kf.add_observation(current_time, seq[i])
                continue

            kf.add_observation(current_time, seq[i])
            pred_pos = kf.predict_at_time(current_time)
            full_pred[i] += pred_pos
            pred_cnt[i] += 1

        pred_end = min(end_pred, N)
        for j in range(end_obs, pred_end):
            target_t = time_seq[j]
            pred_pos = kf.predict_at_time(target_t)
            full_pred[j] += pred_pos
            pred_cnt[j] += 1

        start += stride

    mask = pred_cnt > 0
    full_pred[mask] = full_pred[mask] / pred_cnt[mask]
    full_pred[~mask] = seq[~mask]
    return full_pred

# -------------------------- 滑动窗口评测函数（LR专用） --------------------------
def sliding_evaluate_windows(seq, time_seq, obs_steps, pred_steps,
                             filter_cls, std_pos, stride=1):
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
        kf = LinearPredictor(std_pos)
        kf.init_state(pos=seq[start])
        pred_win[0] = seq[start]

        # 观测段
        for idx_win in range(1, obs_steps):
            g = start + idx_win
            t_now = time_seq[g]
            kf.add_observation(t_now, seq[g])
            pred_pos = kf.predict_at_time(t_now)
            pred_win[idx_win] = pred_pos

        # 预测段外推
        for idx_win in range(obs_steps, win_total):
            g = start + idx_win
            t_future = time_seq[g]
            pred_pos = kf.predict_at_time(t_future)
            pred_win[idx_win] = pred_pos

        metrics = compute_pred_only_metrics(gt_win, pred_win, obs_steps)
        metric_records.append(metrics)
        start += stride

    return np.array(metric_records)

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 超参配置
    obs_steps = 8
    pred_steps = 5
    dim=3
    stride = 1
    std_pos = 0.05

    # 创建输出目录
    os.makedirs("results/tables", exist_ok=True)
    fig_save_path = r"C:\Users\86134\Desktop\drone\results\figures\Linear_3d.png"
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

    traj_lr = sliding_predict_overlap(
        seq=gt_all,
        time_seq=t_all,
        obs_steps=obs_steps,
        pred_steps=pred_steps,
        filter_cls=LinearPredictor,
        std_pos=std_pos,
        stride=stride
    )

    metric_lr = sliding_evaluate_windows(
        seq=gt_all,
        time_seq=t_all,
        obs_steps=obs_steps,
        pred_steps=pred_steps,
        filter_cls=LinearPredictor,
        std_pos=std_pos,
        stride=stride
    )

    if metric_lr.shape[0] > 0:
        rmse, ade, fde, rx, ry, rz = metric_lr.mean(axis=0)
        print("==== Linear Regression 评测结果 ====")
        print(f"窗口数量: {metric_lr.shape[0]}")
        print(f"RMSE = {rmse:.4f}")
        print(f"ADE  = {ade:.4f}")
        print(f"FDE  = {fde:.4f}")
        summary_data = np.array([
            ["窗口数量", metric_lr.shape[0]],
            ["RMSE", round(rmse, 4)],
            ["ADE", round(ade, 4)],
            ["FDE", round(fde, 4)],
            ["RMSE_X", round(rx,4)],
            ["RMSE_Y", round(ry,4)],
            ["RMSE_Z", round(rz,4)]
        ])
        np.savetxt("results/tables/linear_summary.csv", summary_data, delimiter=",", fmt="%s", encoding="utf-8")

    # 3D绘图
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gt_all[:,0], gt_all[:,1], gt_all[:,2], c="#ff3333", s=8, alpha=0.7, label="Truth")
    ax.plot(traj_lr[:,0], traj_lr[:,1], traj_lr[:,2], c="#0066ff", lw=1.0, label="Linear Regression")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Linear Regression Trajectory Prediction")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    plt.show()