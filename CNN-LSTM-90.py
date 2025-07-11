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

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 1. 数据准备
# --------------------------------------------------
train_df = aggregate_daily_data("train.csv", "day_train.csv")
test_df = aggregate_daily_data("test.csv", "day_test.csv")

# 用均值填充 NaN
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

train_df.index = pd.to_datetime(train_df.index)
train_df.sort_index(inplace=True)

test_df.index = pd.to_datetime(test_df.index)
test_df.sort_index(inplace=True)

feature_columns = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_remainder",
    "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
]

missing = [c for c in feature_columns if c not in train_df.columns or c not in test_df.columns]
if missing:
    print(f"[Warn] Columns not found and will be dropped: {missing}")
    feature_columns = [c for c in feature_columns if c not in missing]

train_data = train_df[feature_columns].values
test_data = test_df[feature_columns].values

# 归一化
scalers = {c: MinMaxScaler() for c in feature_columns}
train_scaled = np.zeros_like(train_data, dtype=np.float32)
test_scaled = np.zeros_like(test_data, dtype=np.float32)

for i, col in enumerate(feature_columns):
    train_scaled[:, i] = scalers[col].fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
    test_scaled[:, i] = scalers[col].transform(test_data[:, i].reshape(-1, 1)).flatten()

# --------------------------------------------------
# 2. Dataset & Dataloader
# --------------------------------------------------
SEQ_LEN = 90   # 过去 90 天作为输入
HORIZON = 90   # 预测 90 天
BATCH_SIZE = 64
TARGET_IDX = feature_columns.index("Global_active_power")

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, target_idx: int, seq_len: int, horizon: int):
        self.data = data
        self.target_idx = target_idx
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]                        # (seq_len, features)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.horizon, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_ds = TimeSeriesDataset(train_scaled, TARGET_IDX, SEQ_LEN, HORIZON)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# --------------------------------------------------
# 3. 模型：CNN + LSTM
# --------------------------------------------------
class CNNLSTM(nn.Module):
    def __init__(self, input_size: int, conv_channels: int = 128, lstm_hidden: int = 256, lstm_layers: int = 2):
        super().__init__()
        # Conv1d expects shape (batch, channels, seq_len), 所以把特征数当作输入通道
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)  # 降一半序列长度
        # LSTM 接收 (batch, seq_len', channels)
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, HORIZON)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)               # -> (batch, features, seq_len)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)                     # -> (batch, conv_channels, seq_len/2)
        x = x.permute(0, 2, 1)               # -> (batch, seq_len/2, conv_channels)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])    # 取 LSTM 最后时刻
        return out

# --------------------------------------------------
# 4. 训练 & 评估
# --------------------------------------------------
EPOCHS = 2000
LR = 1e-4
criterion = nn.MSELoss()

def train(model: nn.Module, loader: DataLoader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        if ep % 10 == 0:
            print(f"Epoch {ep}/{epochs}  MSE: {epoch_loss / len(loader.dataset):.6f}")

@torch.no_grad()
def forecast(model: nn.Module, data_scaled: np.ndarray):
    model.eval()
    input_block = data_scaled[-SEQ_LEN:]
    inp = torch.tensor(input_block, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    pred_scaled = model(inp).cpu().numpy().flatten()
    scaler_g = scalers["Global_active_power"]
    pred = scaler_g.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    true_scaled = data_scaled[SEQ_LEN:SEQ_LEN+HORIZON, TARGET_IDX]
    truth = scaler_g.inverse_transform(true_scaled.reshape(-1, 1)).flatten()
    mse = mean_squared_error(truth, pred)
    mae = mean_absolute_error(truth, pred)
    return pred, truth, mse, mae

# --------------------------------------------------
# 5. 多轮实验主流程
# --------------------------------------------------
if __name__ == "__main__":
    INPUT_SIZE = len(feature_columns)
    N_EXPERIMENTS = 1
    mse_list = []
    mae_list = []

    # 创建图片输出文件夹
    os.makedirs("image", exist_ok=True)

    for exp in range(N_EXPERIMENTS):
        print(f"\n===== 实验 {exp+1} / {N_EXPERIMENTS} =====")
        torch.manual_seed(RANDOM_SEED + exp)
        np.random.seed(RANDOM_SEED + exp)
        model = CNNLSTM(INPUT_SIZE).to(DEVICE)
        train(model, train_loader, EPOCHS)
        pred_90, truth_90, mse_90, mae_90 = forecast(model, test_scaled)
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
        plt.title(f"CNN+LSTM 预测未来90天 Global Active Power - 实验{exp+1}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"image/CNN-LSTM-90-{exp+1}.png")
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
    # plt.title('5次CNN+LSTM预测实验的MSE与MAE')
    # plt.xticks(x, [str(i) for i in x])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('image/CNN-LSTM-90-bar.png')
    # plt.close()

    # print("Finished! -> image/CNN-LSTM-90-bar.png") 