import pandas as pd


def aggregate_daily_data(input_file, output_file, expected_records_per_day=1440):
    """
    按天聚合电力数据，自动处理缺失值，丢弃数据不足一天的日期，并返回聚合后的DataFrame。

    参数:
    - input_file: 输入CSV文件路径
    - output_file: 输出CSV文件路径
    - expected_records_per_day: 每天应有的记录数，默认1440（每分钟一条）

    返回:
    - 聚合后的 pandas DataFrame
    """
    # 读取数据
    df = pd.read_csv(input_file)

    # 转换 DateTime 列为日期时间类型
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # 按天分组，统计每天的记录数
    counts = df.groupby(df['DateTime'].dt.date).size()

    # 仅保留数据完整的日期
    valid_dates = counts[counts >= expected_records_per_day].index
    df = df[df['DateTime'].dt.date.isin(valid_dates)]

    # 定义安全聚合函数（忽略缺失）
    def safe_first(series):
        non_missing = series.dropna()
        return non_missing.iloc[0] if not non_missing.empty else None

    # 按天聚合数据
    daily_agg = df.groupby(df['DateTime'].dt.date).agg({
        'Global_active_power': lambda x: pd.to_numeric(x, errors='coerce').sum(),
        'Global_reactive_power': lambda x: pd.to_numeric(x, errors='coerce').sum(),
        'Sub_metering_1': lambda x: pd.to_numeric(x, errors='coerce').sum(),
        'Sub_metering_2': lambda x: pd.to_numeric(x, errors='coerce').sum(),
        'Sub_metering_3': lambda x: pd.to_numeric(x, errors='coerce').sum(),
        'Voltage': lambda x: pd.to_numeric(x, errors='coerce').mean(),
        'Global_intensity': lambda x: pd.to_numeric(x, errors='coerce').mean(),
        'RR': safe_first,
        'NBJRR1': safe_first,
        'NBJRR5': safe_first,
        'NBJRR10': safe_first,
        'NBJBROU': safe_first
    })

    # ➊ 计算每天的剩余分路能耗（Wh）
    daily_agg['Sub_metering_remainder'] = (
        daily_agg['Global_active_power'] * 1000 / 60
        - (daily_agg['Sub_metering_1'] + daily_agg['Sub_metering_2'] + daily_agg['Sub_metering_3'])
    ).round(3)

    # ➋ RR 单位换算：原数据是“毫米的十分之一”，需 ÷10 才是毫米
    if 'RR' in daily_agg.columns:
        daily_agg['RR'] = daily_agg['RR'] / 10.0

    # 保存结果到CSV
    # daily_agg.to_csv(output_file)
    # print(f"已完成数据按天聚合（已丢弃不满一天的日期），结果已保存为 '{output_file}'")

    return daily_agg


if __name__ == "__main__":
    result_df = aggregate_daily_data('train.csv', 'daily_aggregated_dataset.csv')
    print(result_df.head())
