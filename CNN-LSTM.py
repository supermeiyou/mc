import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data import aggregate_daily_data
from tqdm import tqdm

"""
CNN_LSTM.py  ——  基于 data.py 日聚合结果的 **CNN+LSTM 混合模型**
--------------------------------------------------------------
• 流水线：Conv1d(抽取局部时间窗特征) → LSTM(建模长期依赖) → 全连接预测未来若干步
• 支持 data.py 派生的 "Sub_metering_remainder" 等新列
• 使用与 LSTM.py 相同的数据准备流程，方便横向对比
"""

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
SEQ_LEN = 90   # 过去 60 天作为输入
HORIZON = 90   # 预测 30 天
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
    def __init__(self, input_size: int, conv_channels: int = 64, lstm_hidden: int = 128, lstm_layers: int = 2):
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
EPOCHS = 1000
LR = 1e-3
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
        print(f"Epoch {ep}/{epochs}  MSE: {epoch_loss / len(loader.dataset):.6f}")

@torch.no_grad()
def forecast(model: nn.Module, data_scaled: np.ndarray):
    model.eval()
    input_block = data_scaled[-SEQ_LEN:]
    inp = torch.tensor(input_block, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    pred_scaled = model(inp).cpu().numpy().flatten()
    scaler_g = scalers["Global_active_power"]
    pred = scaler_g.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    true_scaled = data_scaled[:HORIZON, TARGET_IDX]
    truth = scaler_g.inverse_transform(true_scaled.reshape(-1, 1)).flatten()
    mse = mean_squared_error(truth, pred)
    mae = mean_absolute_error(truth, pred)
    return pred, truth, mse, mae

# --------------------------------------------------
# 5. 主流程
# --------------------------------------------------
if __name__ == "__main__":
    INPUT_SIZE = len(feature_columns)
    model = CNNLSTM(INPUT_SIZE).to(DEVICE)

    print("Training CNN+LSTM model …")
    train(model, train_loader, EPOCHS)

    print("Evaluating …")
    pred, truth, mse, mae = forecast(model, test_scaled)
    print(f"MSE: {mse:.3f}, MAE: {mae:.3f}")

    # 结果保存
    dates = pd.date_range(start=test_df.index[0], periods=HORIZON, freq="D")
    # pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Predicted_GAP": pred}).to_csv("cnn_lstm_predictions.csv", index=False)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(dates, truth, label="True")
    plt.plot(dates, pred, "--", label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Global Active Power (kW)")
    plt.title("CNN+LSTM Forecast 30 Days")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cnn_lstm_power_forecast.png")
    plt.close()

    print("Finished! -> cnn_lstm_power_forecast.png")
