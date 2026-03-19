import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append("./")
from ch03.btc_usdt import test_sationary
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# 데이터 프레임, 훈련 집합 크기, 예측 기간, 자기상관 차수, 예측 방법 문자열
def rolling_forecast(
    diff: np.ndarray, train_len: int, horizon: int, window: int, method: str
) -> list:
    total_len = train_len + horizon

    # 평균값 예측 방법
    if method == "mean":
        pred_mean = []

        # train 끝/test 시작에서 시작, 총 갯수에서 끝, window 크기(2)만큼 증가
        for t in range(train_len, total_len, window):
            mean = np.mean(diff[:t])
            # 요 코드가 더 효율적일 거 같다...
            pred_mean.extend([mean] * window)
            # 근데 평균값 또는 마지막 값을 그 뒤로 쫙 붙이는 거 아닌가?
            # 이번에는 2개씩 가면서 업데이트 된 값으로 바꿔가나?
            # pred_mean.extend(mean for _ in range(window))

        return pred_mean

    # 마지막 값으로 예측
    elif method == "last":
        pred_last = []

        for t in range(train_len, total_len, window):
            # diff[t - 1]로 해도 되지만 좀 더 명확하게 처음부터 t-1개 까지의 마지막 원소
            last_value = diff[:t][-1]
            # 2개씩 넘어가고, 2칸씩 채우고...
            pred_last.extend([last_value] * window)

        return pred_last

    # MA(q) - SARIMAX 모델 예측
    elif method == "MA":
        pred_MA = []

        for t in range(train_len, total_len, window):
            # order는 p, d, q인데, 자기회귀 차수 p도 0, 차분 d도 0? 이동 평균 차수만 넣는다?
            # 그럼 p, d는 아예 원 데이터를 넣고 돌릴 때 지정하는 건가?
            model = SARIMAX(diff[:t], order=(0, 0, 2))
            res = model.fit(disp=False)
            # 학습 후 예측인데, 0에서 시작, 마지막 원소 (t - 1) 이후로 window 크기 2만큼 예측
            pred = res.get_prediction(0, (t - 1) + window)
            # 예측 값은 predicted_mean 리스트에 들어있고, 그 마지막 2개가 예측값
            oos_pred = pred.predicted_mean[-window:]
            pred_MA.extend(oos_pred)

        return pred_MA


if __name__ == "__main__":
    df = pd.read_csv("./data/widget_sales.csv")
    print(df.head())
    x = np.arange(len(df))
    y = df["widget_sales"]
    xticks = (
        [
            0,
            30,
            57,
            87,
            116,
            145,
            175,
            204,
            234,
            264,
            293,
            323,
            352,
            382,
            409,
            439,
            468,
            498,
        ],
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            "2020",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
        ],
    )

    title = "Widget Sales XYZ"
    xlabel = "Time"
    ylabel = "Widget Sales (k$)"

    # test_sationary(x, y, title, xlabel, ylabel, xticks)
    # 1. raw data 상태에서는 평균, 분산이 다 변하고, ADF p-value가 0.6,
    # 자기상관은 약 45정도까지 유의미하며 선형적으로 감소한다. - stationary하지 않다.
    # 2. 1차 차분에서는 평균은 9월정도부터 안정되고, 분산은 3월 정도부터 안정되고,
    # ADF p-value가 엄청 작아 stationary 하지만, 자기상관은 앞의 2~3개가 유의미하다 - MA(2) 모델

    # 원래 자료와 1차 차분 자료를 비교
    diff = np.diff(y, n=1)

    train = diff[: int(len(diff) * 0.9)]
    test = diff[int(len(diff) * 0.9) :]

    print(len(train), len(test))

    fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), nrows=2, ncols=1, sharex=True)

    ax1.plot(x, y)
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.axvspan(int(len(diff) * 0.9), len(diff), color="lightgray", alpha=0.5)

    ax2.plot(x[1:], diff)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.axvspan(int(len(diff) * 0.9), len(diff), color="lightgray", alpha=0.5)

    plt.xticks(xticks[0], xticks[1])
    plt.tight_layout()
    # plt.show()

    # 평균, 마지막 값, MA(2) 예측 비교
    TRAIN_LEN = len(train)
    HORIZON = len(test)
    WINDOW = 2
    pred_mean = rolling_forecast(diff, TRAIN_LEN, HORIZON, WINDOW, "mean")
    pred_last = rolling_forecast(diff, TRAIN_LEN, HORIZON, WINDOW, "last")
    pred_MA = rolling_forecast(diff, TRAIN_LEN, HORIZON, WINDOW, "MA")

    # 책하고 달라서 약간 꼬이는데...아무튼 다시 df에서 테스트 분량을 받아서 처리해보자...
    pred_df = df[int(len(diff) * 0.9) : -1].copy()
    pred_df["widget_sales_diff"] = test
    pred_df["pred_mean"] = pred_mean
    pred_df["pred_last"] = pred_last
    pred_df["pred_MA"] = pred_MA
    print(pred_df.head())

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(diff)
    ax.plot(pred_df["widget_sales_diff"], "b-", label="Actual")
    ax.plot(pred_df["pred_mean"], "g:", label="Mean")
    ax.plot(pred_df["pred_last"], "r-.", label="Last Value")
    ax.plot(pred_df["pred_MA"], "k--", label="MA(2)")
    ax.legend(loc=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.axvspan(int(len(diff) * 0.9), len(diff), color="lightgray", alpha=0.5)
    ax.set_xlim(430, 500)
    plt.xticks([439, 468, 498], ["Apr", "May", "Jun"])
    # plt.xticks(xticks[0], xticks[1])
    plt.tight_layout()
    plt.show()

    # 평균, 마지막 값, MA 세 가지 MSE 비교
    mse_mean = mean_squared_error(pred_df["widget_sales_diff"], pred_df["pred_mean"])
    mse_last = mean_squared_error(pred_df["widget_sales_diff"], pred_df["pred_last"])
    mse_MA = mean_squared_error(pred_df["widget_sales_diff"], pred_df["pred_MA"])

    # 차트에서는 last value가 오히려 그럴듯 해보였는데, MSE는 MA가 확실히 낫다...
    # 마지막 값은 오히려 꼴찌...이게 LSTM으로 뒤따라가는 예측을 하면 오히려 망하는 것과 뭔가 관련이 있을까?
    print(f"MSE Mean: {mse_mean}")
    print(f"MSE Last: {mse_last}")
    print(f"MSE MA: {mse_MA}")

    fig, ax = plt.subplots(figsize=(12, 6))

    x = ["Mean", "Last Value", "MA(2)"]
    y = [mse_mean, mse_last, mse_MA]

    ax.bar(x, y, width=0.4)
    ax.set_title("Mean Squared Error")
    ax.set_xlabel("Model")
    ax.set_ylabel("MSE")

    plt.tight_layout()
    plt.show()

    df["pred_widget_sales"] = pd.Series()
    pred_y0 = df["widget_sales"].iloc[450]
    cumsum_yt = pred_df["pred_MA"].cumsum()
    # 책과는 다르게, .loc을 사용해야 했고, 뭐가 다른지 모르겠지만
    # 원래 버전 df['pred_widget_sales'][450:]는 copy 어쩌구 오류 뜨면서 업데이트가 안된다고...
    df.loc[450:, "pred_widget_sales"] = pred_y0 + cumsum_yt.values

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["widget_sales"], "b-", label="Actual")
    ax.plot(df["pred_widget_sales"], "r--", label="Forecast")
    ax.legend(loc=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.axvspan(int(len(diff) * 0.9), len(diff), color="lightgray", alpha=0.5)
    ax.set_xlim(400, 500)
    plt.xticks([409, 439, 468, 498], ["Mar", "Apr", "May", "Jun"])

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    mae_MA_undiff = mean_absolute_error(
        df["widget_sales"].iloc[450:], df["pred_widget_sales"].iloc[450:]
    )
    print(f"MAE MA: {mae_MA_undiff}")
    # MAE는 절대값이지만 원래 스케일에 맞춘 오차로, 2.324 정도...
    # 원래 값이 70 정도에서 노는데, 2.32 정도면 3% 정도 오차라는 건데...잘 맞는건가?
    # 그리고 2개 앞만 예측하는 방식이다...
