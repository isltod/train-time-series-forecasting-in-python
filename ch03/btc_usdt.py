import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def draw_line_chart(x, y, title, xlabel, ylabel, xticks=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])
    plt.tight_layout()
    plt.show()


def mean_over_time(process: np.array) -> np.array:
    """
    시간에 따른 평균 변화 계산
    """
    mean_values = np.zeros(len(process))

    for t in range(len(process)):
        mean_values[t] = np.mean(process[: t + 1])

    return mean_values


def variance_over_time(process: np.array) -> np.array:
    """
    시간에 따른 분산 변화 계산
    """
    variance_values = np.zeros(len(process))

    for t in range(len(process)):
        variance_values[t] = np.var(process[: t + 1])

    return variance_values


def draw_auto_corr(process: np.array, lags: int = 100):
    plot_acf(process, lags=lags)
    plt.tight_layout()
    plt.show()


def test_ADF(y):
    ADF_result = adfuller(y)
    print(f"ADF Statistic: {ADF_result[0]}")
    print(f"p-value: {ADF_result[1]}")
    print(f"Critical Values:")
    for key, value in ADF_result[4].items():
        print(f"   {key}: {value}")


def test_sationary(x, y, title="Data", xlabel="X", ylabel="Y", xticks=None):
    # 1. 원본 데이터 훑어보기

    # 1.1. 기본 차트
    draw_line_chart(x, y, title, xlabel, ylabel, xticks)

    # 1.2. 시간에 따른 평균 변화
    raw_mean = mean_over_time(y)
    draw_line_chart(x, raw_mean, "Raw Mean of " + title, xlabel, "Mean", xticks)

    # 1.3. 시간에 따른 분산 변화
    raw_variance = variance_over_time(y)
    draw_line_chart(x, raw_variance, "Raw Var of " + title, xlabel, "Variance", xticks)

    # 1.4. ADF 테스트
    test_ADF(y)

    # 1.5. 자기상관 차트
    draw_auto_corr(y)

    # 2. 1차 차분 데이터
    diff_df = np.diff(y, n=1)
    diff_dt = x[1:]

    # 2.1. 1차 차분 데이터 차트
    draw_line_chart(diff_dt, diff_df, "1st Diff of " + title, xlabel, "Price", xticks)

    # 2.2 1차 차분 시간 평균 변화
    diff_mean = mean_over_time(diff_df)
    draw_line_chart(
        diff_dt, diff_mean, "1st Diff Mean of " + title, xlabel, "Mean", xticks
    )

    # 2.3 1차 차분 시간 분산 변화
    diff_variance = variance_over_time(diff_df)
    draw_line_chart(
        diff_dt, diff_variance, "1st Diff Var of " + title, xlabel, "Variance", xticks
    )

    # 2.4 ADF 테스트
    test_ADF(diff_df)

    # 2.5 자기상관 차트
    draw_auto_corr(diff_df)


if __name__ == "__main__":
    # 1. 일단 종가부터 테스트...가격 변동 변수는 1st diff로 테스트 된다...
    df = pd.read_csv("./data/20220819_BTC-USDT.csv")
    print(df.head())
    xticks = (
        [0, 119, 239, 359, 479, 599, 719, 839, 959, 1079, 1199, 1319, 1439],
        ["00", "02", "04", "06", "08", "10", "12", "14", "16", "18", "20", "22", "24"],
    )
    # test_sationary(df.dt, df.close, "BTC/USDT", "Time", "Price", xticks)

    # 2. 가격 변동 폭을 테스트 해보자...
    df["var_close"] = df.high - df.low
    test_sationary(df.dt, df.var_close, "BTC/USDT Var", "Time", "Price", xticks)

    # 결론:
    # 종가로 하든, 가격 차/가격 변동폭으로 하든, 1차 차분에서는 stationary하고 자기상관성이 없다...
    # 결국 BTC/USDT 2022-8-19 1분 데이터는 랜덤워크이다.
    # 그럼 아무런 예측이 안된다는 것인가?
    # 중간에 심하게 널뛰는 부분 때문에 시간 평균과 분산이 좀 움직이는데, 레비워크인가?
