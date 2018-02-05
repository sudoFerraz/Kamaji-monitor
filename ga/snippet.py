def target_numerical_to_binary(y):
    return y['Values'].apply(lambda x: 1 if x > 0.0 else 0)


def create_numerical_direction(df):
    x = np.zeros(len(df), dtype=np.float64)

    if len(df['open']) == len(df['close']):
        for _ in range(len(df['open'])):
            k = (df['high'][_] - df['open'][_]) - (df['close'][_] - df['low'][_])
            v = math.sqrt((df['close'][_] - df['boll_lb'][_]) * (df['close'][_] - df['boll_lb'][_]))
            u = math.sqrt((df['open'][_] - df['boll_ub'][_]) * (df['open'][_] - df['boll_ub'][_]))
            r = (df['rsi_6'][_] + df['rsi_12'][_]) / 200
            x[_] = (r * (df['middle'][_] * k) + (v - u) * df['macd'][_]) / (r + df['macd'][_])

    return pd.DataFrame(data=x, index=range(len(x)), columns=['Values'])
