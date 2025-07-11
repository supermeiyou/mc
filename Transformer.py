import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 数据准备
# 读取数据
train_df = pd.read_csv('day_train.csv')
test_df = pd.read_csv('day_test.csv')

# 确保日期格式正确并排序
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])
train_df = train_df.sort_values('Date')
test_df = test_df.sort_values('Date')

# 定义所有数值特征列
feature_columns = [
    'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
    'Sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
]

# 检查列是否存在
missing_cols = [col for col in feature_columns if col not in train_df.columns or col not in test_df.columns]
if missing_cols:
    print(f"警告: 以下列缺失: {missing_cols}")
    feature_columns = [col for col in feature_columns if col in train_df.columns and col in test_df.columns]

# 提取特征和目标
train_data = train_df[feature_columns].values
test_data = test_df[feature_columns].values
target_col = 'Global_active_power'  # 预测目标

# 数据归一化
scalers = {col: MinMaxScaler() for col in feature_columns}
train_data_scaled = np.zeros_like(train_data)
test_data_scaled = np.zeros_like(test_data)

for i, col in enumerate(feature_columns):
    train_data_scaled[:, i] = scalers[col].fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
    test_data_scaled[:, i] = scalers[col].transform(test_data[:, i].reshape(-1, 1)).flatten()

# 创建时间序列数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_idx, sequence_length, prediction_horizon):
        self.data = data
        self.target_idx = target_idx  # Global_active_power的列索引
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length, :]  # 所有特征
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_horizon, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 参数
sequence_length = 90  # 过去90天
prediction_horizon = 90  # 预测未来90天
batch_size = 64
input_size = len(feature_columns)  # 特征数量

# 创建数据集
target_idx = feature_columns.index('Global_active_power')
train_dataset = TimeSeriesDataset(train_data_scaled, target_idx, sequence_length, prediction_horizon)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 2. 定义Transformer模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2, dropout=0.1, prediction_horizon=365):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.input_fc = nn.Linear(input_size, d_model)  # 映射输入特征到d_model
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, prediction_horizon)  # 输出未来365天的预测
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_fc.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src: (batch_size, seq_len, input_size)
        src = self.input_fc(src) * np.sqrt(self.d_model)  # 映射到d_model并缩放
        src = self.pos_encoder(src)  # 添加位置编码
        output = self.transformer_encoder(src)  # Transformer编码器
        output = self.dropout(output[:, -1, :])  # 取最后一个时间步
        output = self.fc(output)  # 输出预测
        return output

# 3. 训练和评估函数
def train_model(model, train_loader, num_epochs=100, lr=0.0001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        '''if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')'''

def evaluate_model(model, test_data, sequence_length, prediction_horizon, target_idx, scaler):
    model.eval()
    with torch.no_grad():
        # 使用最后sequence_length天的数据预测未来prediction_horizon天
        input_data = test_data[-sequence_length:, :]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        pred_scaled = model(input_tensor).cpu().numpy().flatten()
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        true = test_data[:prediction_horizon, target_idx]
        true = scaler.inverse_transform(true.reshape(-1, 1)).flatten()
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
    return pred, mse, mae

# 4. 多轮实验
num_runs = 50
mses, maes = [], []
predictions = []

# 初始化Transformer模型
model = TransformerModel(input_size=input_size, d_model=64, nhead=8, num_layers=2, prediction_horizon=prediction_horizon).to(device)

for run in tqdm(range(num_runs)):
    train_model(model, train_loader)
    pred, mse, mae = evaluate_model(model, test_data_scaled, sequence_length, prediction_horizon, target_idx, scalers['Global_active_power'])
    mses.append(mse)
    maes.append(mae)
    predictions.append(pred)

# 计算平均值和标准差
avg_mse = np.mean(mses)
std_mse = np.std(mses)
avg_mae = np.mean(maes)
std_mae = np.std(maes)

print(f'\nResults over {num_runs} runs:')
print(f'Average MSE: {avg_mse:.6f}, Std MSE: {std_mse:.6f}')
print(f'Average MAE: {avg_mae:.6f}, Std MAE: {std_mae:.6f}')

# 取平均预测结果
avg_pred = np.mean(predictions, axis=0)
true_values = scalers['Global_active_power'].inverse_transform(
    test_data_scaled[:prediction_horizon, target_idx].reshape(-1, 1)).flatten()

# 5. 保存预测结果到CSV
dates = pd.date_range(start=test_df['Date'].iloc[0], periods=prediction_horizon, freq='D')
pred_df = pd.DataFrame({
    'Date': dates.strftime('%Y-%m-%d'),
    'Global_active_power': avg_pred
})
pred_df.to_csv('predictions_transformer_901.csv', index=False)
print("Predictions saved as 'predictions_transformer_901.csv'")

# 6. 绘制预测与真实值对比图
plt.figure(figsize=(12, 6))
plt.plot(dates, true_values, label='Ground Truth', color='blue')
plt.plot(dates, avg_pred, label='Predicted', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kW)')
plt.title('Predicted vs Ground Truth Global Active Power (Transformer)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('power_prediction_plot_transformer_901.png')
plt.close()
print("Prediction plot saved as 'power_prediction_plot_transformer_901.png'")