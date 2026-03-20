import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tqdm import tqdm
from typing import Union


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


def draw_pacf(process: np.array, lags: int = 20):
    plot_pacf(process, lags=lags)
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


def rolling_forecast(
    ts: np.ndarray, train_len: int, horizon: int, window: int, method: str, order=None
) -> list:
    """
    ts: 전체 시계열,
    train_len: 훈련 집합 크기,
    horizon: 예측 기간,
    window: 자기상관 차수,
    method: 예측 방법 문자열
    order: SARIMAX order 튜플
    return: 예측값 리스트
    """
    total_len = train_len + horizon

    # 평균값 예측 방법
    if method == "mean":
        pred_mean = []

        # train 끝/test 시작에서 시작, 총 갯수에서 끝, window 크기(2)만큼 증가
        for t in range(train_len, total_len, window):
            mean = np.mean(ts[:t])
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
            last_value = ts[:t][-1]
            # 2개씩 넘어가고, 2칸씩 채우고...
            pred_last.extend([last_value] * window)

        return pred_last

    # MA(q) - SARIMAX MA 모델 예측
    elif method == "MA":
        pred_MA = []

        for t in range(train_len, total_len, window):
            # order는 p, d, q인데, 자기회귀 차수 p도 0, 차분 d도 0? 이동 평균 차수만 넣는다?
            # 그럼 p, d는 아예 원 데이터를 넣고 돌릴 때 지정하는 건가?
            model = SARIMAX(ts[:t], order=order)
            res = model.fit(disp=False)
            # 학습 후 예측인데, 0에서 시작, 마지막 원소 (t - 1) 이후로 window 크기 2만큼 예측
            pred = res.get_prediction(0, (t - 1) + window)
            # 예측 값은 predicted_mean 리스트에 들어있고, 그 마지막 2개가 예측값
            oos_pred = pred.predicted_mean[-window:]
            pred_MA.extend(oos_pred)

        return pred_MA

    # AR(p) - SARIMAX AR 모델 예측
    elif method == "AR":
        pred_AR = []

        for t in range(train_len, total_len, window):
            # order는 p, d, q인데, 이동평균 차수 q도 0, 차분 d도 0
            model = SARIMAX(ts[:t], order=order)
            res = model.fit(disp=False)
            # 학습 후 예측, 마지막 원소인 (t-1) 이후로 window 2만큼 예측
            pred = res.get_prediction(0, (t - 1) + window)
            # 예측 값은 predicted_mean 리스트에 들어있고, 그 마지막 2개가 예측값
            oos_pred = pred.predicted_mean[-window:]
            pred_AR.extend(oos_pred)

        return pred_AR


# endog 매개변수는 pd.Series와 list를 다 받는다는 얘기겠지...
def optimize_ARMA(endog: Union[pd.Series, list], order_list: list) -> pd.DataFrame:
    results = []
    # 주피터 노트북에서는 tqdm_notebook
    for order in tqdm(order_list):
        try:
            # 하던대로 order는 p, d, q이므로... simple_differencing은 차분 되지 않도록 False
            model = SARIMAX(
                endog,
                order=(order[0], 0, order[1]),
                simple_differencing=False,
            )
            # 상태 메시지 표시 안한다...뭐 별 차이가 없는데?
            result = model.fit(disp=False)
            # AIC 값은 aic 속성에...
            aic = result.aic
            results.append([order, aic])
        except:
            continue

    # 결과 리스트로 데이터프레임 만들고, 정렬해서 반환
    result_df = pd.DataFrame(results)
    result_df.columns = ["(p, q)", "AIC"]
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)

    return result_df


def draw_train_test(x, y, vs, dy=None, ttl="Data", xlbl="X", ylbl="Y", xticks=None):
    # 1차 차분 데이터 있으면 차트 크기 키우고 축도 하나 더...
    if dy is None:
        fig, ax1 = plt.subplots(figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 12))

    # 기본 데이터 차트 그리고
    ax1.plot(x, y)
    ax1.set_title(ttl)
    # 기본 데이터 x 라벨은 차분 데이터 없는 경우에...
    if dy is None:
        ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    ax1.axvspan(len(y) - vs, len(y), color="#808080", alpha=0.2)

    # 차분 데이터 있으면 그것도 그리기
    if dy is not None:
        dx = x[1:]
        ax2.plot(dx, dy)
        ax2.set_xlabel(xlbl)
        ax2.set_ylabel(ylbl)
        ax2.axvspan(len(dy) - vs, len(dy), color="#808080", alpha=0.2)

    # 틱 데이터 있으면...
    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])

    plt.tight_layout()
    plt.show()
