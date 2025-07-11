import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# 新增：设置matplotlib支持中文
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False   # 正常显示负号
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data import aggregate_daily_data
from tqdm import tqdm
import os

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据准备 —— 调用 data.py 中的函数进行按天聚合
train_df = aggregate_daily_data("train.csv", "day_train.csv")
test_df = aggregate_daily_data("test.csv", "day_test.csv")

# 用均值填充 NaN
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

# 确保日期索引正确并排序
train_df.index = pd.to_datetime(train_df.index)
train_df.sort_index(inplace=True)
test_df.index = pd.to_datetime(test_df.index)
test_df.sort_index(inplace=True)

# ----------------------------------------------
# 选择特征列（包含新增 Sub_metering_remainder）
# ----------------------------------------------
feature_columns = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_remainder",
    "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
]

a_missing = [c for c in feature_columns if c not in train_df.columns]
b_missing = [c for c in feature_columns if c not in test_df.columns]
missing_cols = list(set(a_missing + b_missing))
if missing_cols:
    print(f"[Warn] The following columns are missing and will be dropped: {missing_cols}")
    feature_columns = [c for c in feature_columns if c not in missing_cols]

# 提取特征矩阵
train_data = train_df[feature_columns].values
test_data = test_df[feature_columns].values

# 归一化到 [0,1]
scalers = {col: MinMaxScaler() for col in feature_columns}
train_scaled = np.zeros_like(train_data, dtype=np.float32)
test_scaled = np.zeros_like(test_data, dtype=np.float32)

for idx, col in enumerate(feature_columns):
    train_scaled[:, idx] = scalers[col].fit_transform(train_data[:, idx].reshape(-1, 1)).flatten()
    test_scaled[:, idx] = scalers[col].transform(test_data[:, idx].reshape(-1, 1)).flatten()

# ----------------------------------------------
# 构建 PyTorch Dataset
# ----------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, target_idx: int, seq_len: int, horizon: int):
        self.data = data
        self.target_idx = target_idx
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.horizon, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 序列与预测窗口参数
SEQ_LEN = 90      # 用前90天
HORIZON = 90     # 预测后90天
BATCH_SIZE = 64

# 目标列为 Global_active_power
TARGET_IDX = feature_columns.index("Global_active_power")

train_ds = TimeSeriesDataset(train_scaled, TARGET_IDX, SEQ_LEN, HORIZON)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------------------------
# LSTM 网络
# ----------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, HORIZON)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后时间步
        return out

# 训练函数
def train_model(model, loader, epochs: int = 50, lr: float = 1e-3):
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep + 1}/{epochs}  Loss: {loss.item():.6f}")

# 推理 / 评估函数 —— 用 test 集最后 SEQ_LEN 天预测未来 HORIZON 天
@torch.no_grad()
def forecast(model, data_scaled: np.ndarray, seq_len: int, horizon: int, start_idx: int = 0):
    model.eval()
    input_block = data_scaled[start_idx : start_idx + seq_len]
    inp = torch.tensor(input_block, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    pred_scaled = model(inp).cpu().numpy().flatten()[:horizon]
    # 反归一化到真实值
    scaler_g = scalers["Global_active_power"]
    pred = scaler_g.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    # 真实值
    true_scaled = data_scaled[start_idx + seq_len : start_idx + seq_len + horizon, TARGET_IDX]
    true = scaler_g.inverse_transform(true_scaled.reshape(-1, 1)).flatten()
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    return pred, true, mse, mae

# ----------------------------------------------
# 主流程
# ----------------------------------------------
if __name__ == "__main__":
    INPUT_SIZE = len(feature_columns)
    EPOCHS = 500
    N_EXPERIMENTS = 5
    mse_list = []
    mae_list = []

    # 创建图片输出文件夹
    os.makedirs("image", exist_ok=True)

    for exp in range(N_EXPERIMENTS):
        print(f"\n===== 实验 {exp+1} / {N_EXPERIMENTS} =====")
        torch.manual_seed(42 + exp)
        np.random.seed(42 + exp)
        model = LSTMModel(INPUT_SIZE).to(DEVICE)
        train_model(model, train_loader, epochs=EPOCHS)
        pred_90, truth_90, mse_90, mae_90 = forecast(model, test_scaled, seq_len=SEQ_LEN, horizon=HORIZON, start_idx=0)
        print(f"[90天] MSE: {mse_90:.3f}, MAE: {mae_90:.3f}")
        mse_list.append(mse_90)
        mae_list.append(mae_90)

        # 绘制每次实验的预测曲线与真实曲线
        dates_90 = pd.date_range(start=test_df.index[SEQ_LEN], periods=HORIZON, freq="D")
        plt.figure(figsize=(12, 6))
        plt.plot(dates_90, truth_90, label="真实值", color="blue")
        plt.plot(dates_90, pred_90, "--", label="预测值", color="red")
        plt.xlabel("Date")
        plt.ylabel("Global Active Power (kW)")
        plt.title(f"LSTM 预测未来90天 Global Active Power - 实验{exp+1}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"image/LSTM-90-{exp+1}.png")
        plt.close()

    # 计算均值和标准差
    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)

    print("\n===== 5次实验结果统计 =====")
    for i in range(N_EXPERIMENTS):
        print(f"实验{i+1}: MSE={mse_list[i]:.3f}, MAE={mae_list[i]:.3f}")
    print(f"MSE均值: {mse_mean:.3f}, 标准差: {mse_std:.3f}")
    print(f"MAE均值: {mae_mean:.3f}, 标准差: {mae_std:.3f}")

    # 绘制柱状图（只绘制MAE和MSE，不绘制均值和标准差errorbar）
    # x = np.arange(1, N_EXPERIMENTS+1)
    # width = 0.35
    # plt.figure(figsize=(10,6))
    # plt.bar(x - width/2, mse_list, width, label='MSE', color='skyblue')
    # plt.bar(x + width/2, mae_list, width, label='MAE', color='salmon')
    # plt.xlabel('实验轮次')
    # plt.ylabel('误差')
    # plt.title('5次LSTM预测实验的MSE与MAE')
    # plt.xticks(x, [str(i) for i in x])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('image/LSTM-90-bar.png')
    # plt.close()

    # print("Finished! -> image/LSTM-90-bar.png")
