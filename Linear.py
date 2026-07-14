import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================== 滑动窗口参数 ======================
obs_steps = 8    # 观测帧数
pred_steps = 5   # 预测帧数
dims = 3
stride = 1    # 重叠滑动步长
target_win_id = 0
save_window_metrics_csv = True



# ====================== 线性回归轨迹预测 ======================
def sliding_linear_predict_overlap(seq, time_seq, obs_steps, pred_steps, **kwargs):
    """
    最小二乘线性拟合预测
    obs窗口内拟合 t~x/y/z 一次直线，向外外推pred_steps帧
    """
    N = len(seq)
    full_pred = np.zeros_like(seq)
    stride = kwargs.get("stride", 1)
    start = 0
    while True:
        end_obs = start + obs_steps
        end_pred = start + obs_steps + pred_steps

        if end_obs >= N:
            break

        # 观测窗口数据
        obs_t = time_seq[start:end_obs]
        obs_pos = seq[start:end_obs, :]
        full_pred[start:end_obs] = obs_pos.copy()

        # 分别对 X Y Z 拟合线性 y = k*t + b
        def fit_1d(t_arr, val_arr):
            A = np.vstack([t_arr, np.ones(len(t_arr))]).T
            k, b = np.linalg.lstsq(A, val_arr, rcond=None)[0]
            return k, b

        kx, bx = fit_1d(obs_t, obs_pos[:, 0])
        ky, by = fit_1d(obs_t, obs_pos[:, 1])
        kz, bz = fit_1d(obs_t, obs_pos[:, 2])

        # 预测区间外推
        for j in range(end_obs, min(end_pred, N)):
            t_j = time_seq[j]
            x_hat = kx * t_j + bx
            y_hat = ky * t_j + by
            z_hat = kz * t_j + bz
            full_pred[j] = np.array([x_hat, y_hat, z_hat])

        start += stride
        print(f"[Linear] start={start}")
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

if __name__ == "__main__":
    # =========路径修复（解决uv运行相对路径报错）=========
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "./data/mmaud_mavic3_gt_relative.csv")
    df = pd.read_csv(csv_path)
    timestamps = df["timestamp"].values
    gt_all = df[["x", "y", "z"]].values

    # ==========运行线性回归滑动预测==========
    lin_traj = sliding_linear_predict_overlap(gt_all, timestamps, obs_steps, pred_steps)

    # 评估指标（复用你现成函数）
    rmse, ade, fde, rmse_x, rmse_y, rmse_z = compute_pred_only_metrics(gt_all, lin_traj, obs_steps)

    print("=" * 42)
    print(f"线性回归 | obs={obs_steps}, pred={pred_steps}")
    print("=" * 42)
    print(f"整体RMSE : {rmse:.6f} m")
    print(f"ADE      : {ade:.6f} m")
    print(f"FDE      : {fde:.6f} m")
    print("-" * 42)
    print(f"RMSE_X   : {rmse_x:.6f} m")
    print(f"RMSE_Y   : {rmse_y:.6f} m")
    print(f"RMSE_Z   : {rmse_z:.6f} m")
    print("=" * 42)

    # 保存指标表格
    metrics_df = pd.DataFrame({
        "metric": ["RMSE_3D", "ADE", "FDE", "RMSE_X", "RMSE_Y", "RMSE_Z"],
        "value": [rmse, ade, fde, rmse_x, rmse_y, rmse_z],
        "unit": ["m", "m", "m", "m", "m", "m"]
    })
    out_csv = r"C:\Users\86134\Desktop\drone\results\tables\linear_summary.csv"
    metrics_df.to_csv(out_csv, index=False)
    print("线性模型指标已保存")

    # =========绘图=========
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(gt_all[:, 0], gt_all[:, 1], gt_all[:, 2], c="#ff3333", s=4, alpha=0.7, label="真值")
    ax1.plot(lin_traj[:, 0], lin_traj[:, 1], lin_traj[:, 2], c="#0088ff", lw=1.9, label="Linear线性预测")
    ax1.set_xlabel("X (m)", labelpad=8)
    ax1.set_ylabel("Y (m)", labelpad=8)
    ax1.set_zlabel("Z (m)", labelpad=8)
    ax1.set_title("3D轨迹 Linear线性回归", fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)
    fig1.savefig(r"C:\Users\86134\Desktop\drone\results\figures\linear_3d.png", dpi=150, bbox_inches="tight")
    plt.show()

