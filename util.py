import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
from typing import Union

import torch
import tensorflow as tf
from keras import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell


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


def draw_2line_chart(x, y1, y2, title1, title2, xticks=None):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 12))

    ax1.plot(x, y1)
    ax1.set_title(title1)
    ax1.spines["top"].set_visible(False)

    ax2.plot(x, y2)
    ax2.set_title(title2)
    ax2.spines["top"].set_visible(False)

    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def draw_seasonality(x, y, title, xlabel, ylabel, xticks=None, marks=None, vlines=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    if marks is None:
        ax.plot(x, y)
    else:
        ax.plot(x, y, markevery=marks, marker="o", markerfacecolor="r")

    if vlines is not None:
        for line in vlines:
            ax.axvline(x=line, linestyle="--", color="black", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])
    plt.tight_layout()
    plt.show()


def draw_seasonal_decompose(x, y, period, xticks=None):
    decomposition = STL(y, period=period).fit()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12, 12))

    ax1.plot(x, decomposition.observed)
    ax1.set_ylabel("Observed")
    ax2.plot(x, decomposition.trend)
    ax2.set_ylabel("Trend")
    ax3.plot(x, decomposition.seasonal)
    ax3.set_ylabel("Seasonal")
    # 근데 이게 필요할까?
    if (decomposition.seasonal.max() < 1) and (decomposition.seasonal.min() > -1):
        ax3.set_ylim(-1, 1)
    ax4.plot(x, decomposition.resid)
    ax4.set_ylabel("Residual")
    if (decomposition.resid.max() < 1) and (decomposition.resid.min() > -1):
        ax4.set_ylim(-1, 1)

    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])

    fig.autofmt_xdate()
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


def roll_fore_mat(
    ts: pd.DataFrame, trn_len: int, hrzn: int, wndw: int, mthd: str, order=None
) -> list:
    total_len = trn_len + hrzn
    end_idx = trn_len

    cols = ts.columns

    if mthd == "last":
        pred_last = {}
        for col in cols:
            pred_last[col] = []

        for t in range(trn_len, total_len, wndw):
            for col in cols:
                last_value = ts[:t].iloc[-1][col]
                pred_last[col].extend([last_value] * wndw)

        return pred_last

    elif mthd == "VAR":
        pred_VAR = {}
        for col in cols:
            pred_VAR[col] = []

        for t in range(trn_len, total_len, wndw):
            res = model_VARMAX(ts[:t], order)
            preds = res.get_prediction(0, (t - 1) + wndw)
            for col in cols:
                oos_pred = preds.predicted_mean.iloc[-wndw:][col]
                pred_VAR[col].extend(oos_pred)

        return pred_VAR


def roll_fore_vec(
    ts: np.ndarray,
    trn_len: int,
    hrzn: int,
    wndw: int,
    mthd: str,
    ordr=None,
    ssnl_ordr=None,
    exog=None,
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
    total_len = trn_len + hrzn

    # 평균값 예측 방법
    if mthd == "mean":
        pred_mean = []

        # train 끝/test 시작에서 시작, 총 갯수에서 끝, window 크기(2)만큼 증가
        for t in range(trn_len, total_len, wndw):
            mean = np.mean(ts[:t])
            # 요 코드가 더 효율적일 거 같다...
            pred_mean.extend([mean] * wndw)
            # 근데 평균값 또는 마지막 값을 그 뒤로 쫙 붙이는 거 아닌가?
            # 이번에는 2개씩 가면서 업데이트 된 값으로 바꿔가나?
            # pred_mean.extend(mean for _ in range(window))

        return pred_mean

    # 마지막 값으로 예측
    elif mthd == "last":
        pred_last = []

        for t in range(trn_len, total_len, wndw):
            # diff[t - 1]로 해도 되지만 좀 더 명확하게 처음부터 t-1개 까지의 마지막 원소
            last_value = ts[:t][-1]
            # 2개씩 넘어가고, 2칸씩 채우고...
            pred_last.extend([last_value] * wndw)

        return pred_last

    # 마지막 계절성 반복
    elif mthd == "last_season":
        pred_last_season = []

        for t in range(trn_len, total_len, wndw):
            last_season = ts[t - wndw : t]
            pred_last_season.extend(last_season)

        return pred_last_season

    # SARIMAX MA, AR, ARMA 모델 예측
    elif mthd in ["MA", "AR", "ARMA", "ARIMA", "SARIMA", "SARIMAX"]:
        pred_SARIMAX = []

        for t in tqdm(range(trn_len, total_len, wndw)):
            # order는 p, d, q인데, 자기회귀 차수 p도 0, 차분 d도 0? 이동 평균 차수만 넣는다?
            # 그럼 p, d는 아예 원 데이터를 넣고 돌릴 때 지정하는 건가?
            if exog is None:
                res = SARIMAX(
                    ts[:t],
                    order=ordr,
                    seasonal_order=ssnl_ordr,
                    simple_differencing=False,
                ).fit(disp=False)
                # 학습 후 예측인데, 0에서 시작, 마지막 원소 (t - 1) 이후로 window 크기 2만큼 예측
                pred = res.get_prediction(0, (t - 1) + wndw)
            else:
                res = SARIMAX(
                    ts[:t],
                    exog=exog[:t],
                    order=ordr,
                    seasonal_order=ssnl_ordr,
                    simple_differencing=False,
                ).fit(disp=False)
                # 외생변수 있을 때는 아예 모양이 달라지네...
                pred = res.get_prediction(exog=exog)
            # 예측 값은 predicted_mean 리스트에 들어있고, 그 마지막 2개가 예측값
            oos_pred = pred.predicted_mean[-wndw:]
            pred_SARIMAX.extend(oos_pred)

        return pred_SARIMAX


# endog 매개변수는 pd.Series와 list를 다 받는다는 얘기겠지...
def optimize_SARIMA(
    endog: Union[pd.Series, list],
    order_list: list,
    d: int = 0,
    D: int = 0,
    s: int = 0,
    exog: Union[pd.Series, list] = None,
) -> pd.DataFrame:
    results = []
    num_orders = len(order_list[0])
    # 주피터 노트북에서는 tqdm_notebook
    for order in tqdm(order_list):
        try:
            # ARIMA는 order_list가 2, SARIMA는 4
            if num_orders == 2:
                # 하던대로 order는 p, d, q이므로... simple_differencing은 차분 되지 않도록 False
                model = SARIMAX(
                    endog,
                    order=(order[0], d, order[1]),
                    simple_differencing=False,
                )
            elif num_orders == 4:
                model = SARIMAX(
                    endog,
                    exog=exog,
                    order=(order[0], d, order[1]),
                    seasonal_order=(order[2], D, order[3], s),
                    simple_differencing=False,
                )
            # disp는 상태 메시지 표시 안한다...뭐 별 차이가 없는데?
            result = model.fit(disp=False)
            # AIC 값은 aic 속성에...
            aic = result.aic
            results.append([order, aic])
        except:
            continue

    # 결과 리스트로 데이터프레임 만들고, 정렬해서 반환
    result_df = pd.DataFrame(results)
    if num_orders == 2:
        result_df.columns = ["(p, q)", "AIC"]
    elif num_orders == 4:
        result_df.columns = ["(p, q, P, Q)", "AIC"]
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)

    return result_df


def model_VARMAX(endog, order):
    return VARMAX(endog, order=(order, 0)).fit(disp=False)


def optimize_VARMAX(endog: Union[pd.Series, list], max_order: int) -> pd.DataFrame:
    results = []
    for order in tqdm(range(max_order)):
        try:
            model = model_VARMAX(endog, order)
            aic = model.aic
            results.append([order, aic])
        except:
            continue

    result_df = pd.DataFrame(results)
    result_df.columns = ["p", "AIC"]
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)

    return result_df


def resid_VARMAX(endog, order):
    result = model_VARMAX(endog, order)
    print(result.summary())

    cols = result.resid.columns
    for i in range(len(cols)):
        result.plot_diagnostics(figsize=(12, 8), variable=i)
        plt.tight_layout()
        plt.show()

    for col in cols:
        print("Ljung-Box Test for " + col + " ----------------------------")
        lb = acorr_ljungbox(result.resid[col], 10)
        print(lb["lb_pvalue"])

    return result


def model_SARIMAX(ts, p, d, q, P=0, D=0, Q=0, s=0, exog=None):
    # 모델 적용 결과
    model = SARIMAX(
        ts,
        order=(p, d, q),
        exog=exog,
        seasonal_order=(P, D, Q, s),
        simple_differencing=False,
    )
    result = model.fit()
    return result


def resid_SARIMAX(ts, p, d, q, P=0, D=0, Q=0, s=0, exog=None):
    # 모델 적용 결과
    result = model_SARIMAX(ts, p, d, q, P, D, Q, s, exog)
    print(result.summary())
    # 잔차 분석
    result.plot_diagnostics(figsize=(12, 8))
    plt.tight_layout()
    plt.show()
    # 융박스 테스트
    lb = acorr_ljungbox(result.resid, 10)
    print(lb["lb_pvalue"])
    return result


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
        diff = len(y) - len(dy)
        dx = x[diff:]
        ax2.plot(dx, dy)
        ax2.set_xlabel(xlbl)
        ax2.set_ylabel(ylbl)
        ax2.axvspan(len(y) - vs, len(y), color="#808080", alpha=0.2)

    # 틱 데이터 있으면...
    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])

    plt.tight_layout()
    plt.show()


def draw_pred_vec(x, y, vs, preds, ttl="Data", xlbl="X", ylbl="Y", xticks=None):
    symb = ["b-", "g:", "r-.", "k--", "y-"]
    fig, ax = plt.subplots(figsize=(12, 6))

    # 차트 기본형 그리고
    ax.plot(x, y, symb[0], label="Actual")

    # 예측 사전들 돌면서 예측들 그리기
    cnt = 1
    test_start = len(y) - vs
    for key, pred in preds.items():
        ax.plot(x[test_start:], pred, symb[cnt], label=key)
        cnt += 1
        if cnt == len(symb):
            cnt = 1

    # 범례, 제목, 라벨 등
    ax.legend(loc=2)
    ax.set_title(ttl)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)

    # 예측 부분 반전, 범위 한정 등
    ax.axvspan(test_start, len(y), color="#808080", alpha=0.2)

    # 틱 데이터 있으면...
    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])

    # x축 범위를 예측 주변으로...밑에
    test_start = len(y) - int(vs * 1.2)
    ax.set_xlim(test_start, len(y))

    plt.tight_layout()
    plt.show()


def draw_pred_mat(x, ys, vs, preds, ttl="Data", xlbl="X", ylbl="Y", xticks=None):
    symb = ["b-", "g:", "r-.", "k--", "y-"]
    fig, axs = plt.subplots(figsize=(12, 6 * len(ys)), ncols=1, nrows=len(ys))

    # 여기 preds는 [{}, {}] 꼴로 ys와 같은 순서로 사전이 들어있다고 가정...
    # 예측 사전들 돌면서 예측들 그리기
    for i in range(len(ys)):
        # 차트 기본형 그리고
        axs[i].plot(x, ys[i], symb[0], label="Actual")

        cnt = 1
        test_start = len(ys[i]) - vs
        pred_dict = preds[i]
        for key, pred in pred_dict.items():
            axs[i].plot(x[test_start:], pred, symb[cnt], label=key)
            cnt += 1
            if cnt == len(symb):
                cnt = 1

        # 범례, 제목, 라벨 등
        axs[i].legend(loc=2)
        axs[i].set_title(ttl[i])
        axs[i].set_xlabel(xlbl)
        axs[i].set_ylabel(ylbl[i])

        # 예측 부분 반전, 범위 한정 등
        if xticks is not None:
            axs[i].set_xticks(xticks[0])
            axs[i].set_xticklabels(xticks[1])
        axs[i].axvspan(test_start, len(ys[i]), color="#808080", alpha=0.2)
        test_start = len(ys[i]) - int(vs * 1.2)
        plt.xlim(test_start, len(ys[i]))

    plt.tight_layout()
    plt.show()


def _bar_error(dict, title, xlabel, ylabel):
    x = list(dict.keys())
    y = list(dict.values())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, y, width=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(y) * 1.5)

    for i, v in enumerate(y):
        plt.text(x=i, y=v * 1.1, s=str(round(v, 2)), ha="center")

    plt.tight_layout()
    plt.show()


def compare_MSE(test, preds):
    mse = {}

    for key, arr in preds.items():
        mse[key] = mean_squared_error(test, arr)
        print(f"{key} MSE: {mse[key]}")

    _bar_error(mse, "MSE Comparison", "Method", "MSE")


def compare_MAE(test, preds):
    mae = {}

    for key, arr in preds.items():
        mae[key] = mean_absolute_error(test, arr)
        print(f"{key} MAE: {mae[key]}")

    _bar_error(mae, "MAE Comparison", "Method", "MAE")


def compare_MAPE(test, preds):
    mape = {}

    for key, arr in preds.items():
        # mape[key] = np.mean(np.abs((test - arr) / test)) * 100
        mape[key] = mean_absolute_percentage_error(test, arr) * 100
        print(f"{key} MAPE: {mape[key]}")

    _bar_error(mape, "MAPE Comparison", "Method", "MAPE")


def compre_Real_Scale(
    x, y, mdl, cut, pred, ttl="Data", xlbl="X", ylbl="Y", xticks=None
):
    start = len(y) - cut
    undiff_pred = y[start] + pred.cumsum()

    pred_dict = {mdl: undiff_pred}
    draw_pred_vec(x, y, cut, pred_dict, ttl, xlbl, ylbl, xticks)

    mae_undiff = mean_absolute_error(y[start:], undiff_pred)
    print(f"{mdl} MAE: {mae_undiff}")


class DataWindow:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df,
        val_df,
        test_df,
        label_columns=None,
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # 이게 예측하고자 하는 열의 이름이라고...
        self.label_columns = label_columns
        # 번호: 이름을 이름: 번호로...
        if label_columns is not None:
            # 이건 도식화에 사용? 그래프 그린다는 말인가?
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        # 대상 변수에서 특징을 분리할 때 사용? 뭔 소린지...
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        # 0~input_width 슬라이스 만들어서 밑에 줄에서 사용
        self.input_slice = slice(0, input_width)
        # 0~(input_width + shift)까지 리스트 만들고, 다시 0~input_width까지 잘라?
        # 그냥 애초에 np.arange(input_width)하면 될걸 왜 이렇게 하지?
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        # (total_window_size+shif) 만큼의 리스트에서 앞쪽 0~input_width는 input_indices로,
        # shift~마지막까지는 label_indices로...
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    # 일단 입력과 라벨로 나눈다는 얘기 같은데...
    # 짐작하기에 0~24 데이터 주면 0~23은 입력, 1~24는 정답지로 나눠서,
    # 0-1, 1-2, ... 뭐 이렇게 돌리겠다는 거 같긴 한데...디버깅해도 여길 안들어오니 알 수가 있나...
    def split_to_inputs_labels(self, features):
        # 배치 32 x 시간 24 x 컬럼 3?에서 앞은 입력 뒤는 정답지로...
        # input_slice = (처음, input_width)
        inputs = features[:, self.input_slice, :]
        # labels_slice = (label_start, 끝)
        labels = features[:, self.labels_slice, :]
        # 예측 목표가 리스트로 전달됐다면...
        if self.label_columns is not None:
            # 그걸 마지막 차원을 기준으로 이어붙이는데...
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )
        # 그리고 그걸 다시 input_width와 label_width를 중심으로 차원을 조정해?
        # 그럼 배치 32 x 컬럼 3 x 시간 24 이렇게 되나?...
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col="traffic_volume", max_subplots=3):
        # 맨 밑에 @property로 정의한 속성인데...밑에보면 그냥 train에서 값을 하나 읽어오는 건데...
        # 왜 여기는 (inputs, labels) 이렇게 들어있다는 거지?
        inputs, labels = self.sample_batch

        plt.figure(figsize=(12, 8))
        # plot_col로 그리려는 컬럼을 인덱스로 만들고...
        plot_col_index = self.column_indices[plot_col]

        # 부분 차트의 최대 갯수는 입력 데이터 갯수와 max_subplots 지정 숫자 중에 적은 것으로...반복해서
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            # 일단, 3행 1열 구조는 안 변하고, 위에서 아래로 추가해가는 모양...
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f"{plot_col} [scaled]")
            # 이건 입력 데이터 그리기...
            plt.plot(
                self.input_indices,
                # 뭐지? 이것도 batch, time, column 구존가? 일단 마지막은 맞는데...
                # 그렇다면 특정 배치, 특정 속성을 시간 그래프로 그리기...
                # 그럼 왜 배치를 3개까지만 그리지?
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            # 라벨 컬럼 이름 리스트가 있다는 건, 예측할 필드가 2 이상이라는 얘기?
            if self.label_columns:
                # 2 이상이라면 그 중 뭔지 라벨 컬럼 딕셔너리에서 찾고
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                # 아니고 하나라면 그냥 그 인덱스 사용한다...
                label_col_index = plot_col_index

            # 근데 인덱스가 없어? 문제인데 그냥 지나가 버린다...
            if label_col_index is None:
                continue
            # 라벨은 scatter로 뿌린다..이건 정답인가 아닌가...
            # 일단 위에 입력 데이터와 같은 모양으로, 특정 배치, 특정 속성을 시간 그래프로 그리기...
            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                marker="s",
                label="Labels",
                c="#2ca02c",
                s=64,
            )

            # 모델이 지정되어 있다면 그 예측 값을 또한 그리는 모양인데...
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            # 처음 반복에서만 레전드 넣고...마지막이 아니어도 되나?
            if n == 0:
                plt.legend()

        # x축 이름은 마지막에? 먼저 하면 안되나?
        plt.xlabel("Time [h]")
        plt.tight_layout()
        plt.show()

    def make_dataset(self, data):
        # 원래는 df로 받는 모양이네...여기서 np로 바꾸고 그걸 텐서로...
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            # train/val/test 다 들어있다고...
            data=data,
            # 근데 이건 위에 split_to_input_labels에서 처리하니 None이라고?
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )
        # 이게 split_to_inputs_labels에서 targets를 처리한다는 얘기 같은데...여기서 데이터 형식을 봐야...
        # ds는 tf.data.Dataset 인스턴스고, map은 transformation 메서드라고...
        ds = ds.map(self.split_to_inputs_labels)
        return ds

    # 클래스 메서드를 인스턴스 변수처럼 읽게 - 게터로 사용...세터는 @<name>.setter 형태로...
    # train/val/test에 make_dataset 함수를 적용하기 위한 속성이라고...
    # 명시적으로 make_dataset 함수 호출 안했는데도 들어간다면 이걸 거친거라고 보면 된다...
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        # getattr는 객체 self 에서 _sample_batch 속성(필드나 메서드)을 불러오는 함수...
        result = getattr(self, "_sample_batch", None)
        # 즉 _sample_batch란 속성이 지정되어 있지 않다면...
        if result is None:
            # train df를 Iterable로 만들고, 다음 값을 하나 next로 받아와서
            result = next(iter(self.train))
            # 그걸로 _sample_batch 속성을 만든다...
            self._sample_batch = result

        # 즉, train에서 값을 하나 읽어서 _sample_batch 속성에 저장해놓고 반환하고,
        # 다음부터는 _sample_batch 저장된 그 값을 반환하는데,
        # 어떤 이유건 _sample_batch 속성이 없거나 사라졌다면, train에서 이전에 읽었던 값 다음 값을 반환
        return result


class Baseline(Model):
    def __init__(self, label_index=None):
        super().__init__()
        # 이게 예측할 컬럼 인덱스를 정해놓는거라고...
        self.label_index = label_index

    # inputs는 train, val, test로 배치x시간x컬럼(32x1x5)로 오고
    def call(self, inputs):
        # 일단 세 갈래인데...우선 목표인 label_index 없으면 그냥 inputs 반환
        if self.label_index is None:
            return inputs
        # 아니고 목표가 여러개라서 label_index가 list라면...
        if isinstance(self.label_index, list):
            tensors = []
            # 각 컬럼마다 돌면서
            for index in self.label_index:
                # 모든 배치, 시간의 목표 컬럼 값들을 뽑는데, 이러면 컬럼 차원이 사라진다...
                result = inputs[:, :, index]
                # 그래서 마지막 숫자를 차원으로 만들어서 원래 32x1x5 모양을 복원하고...
                result = result[:, :, tf.newaxis]
                tensors.append(result)
            # 그걸 마지막 차원을 기준으로 붙이면 다시 배치 x 시간 x 컬럼 차원의 텐서가 된다...
            return tf.concat(tensors, axis=-1)

        # 그것도 아니면 목표가 하나라서 label_index는 숫자...
        # 모든 배치, 시간의 목표 컬럼 값을 뽑는데...
        # 이러면 마지막 차원이 컬럼별 숫자 -> 숫자 하나로 바뀌어 컬럼 차원이 사라지게 된다...
        result = inputs[:, :, self.label_index]
        # 마지막에 차원을 추가해서 처음 32x1x5 차원으로 만들어 반환
        return result[:, :, tf.newaxis]


class MultiStepLastBaseline(Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            # inputs의 모든 배치, 모든 컬럼의 마지막 시간을 시간 축으로 24번 반복...
            aa = inputs[:, -1:, :]
            bb = tf.tile(aa, [1, 24, 1])
            return tf.tile(inputs[:, -1:, :], [1, 24, 1])
        # 24시간씩 32(배치)개에 traffic_volume 이후의 컬럼들(3개)의 값이 있는데... - 32x24x3
        # 거기서 24시간 묶음의 마지막 값들만 뽑으면 32개 배치 x 마지막 시간 1 x 컬럼 3
        aa = inputs[:, -1:, self.label_index :]
        # 그걸 시간축으로 다시 24번 반복 - 32x24x3
        bb = tf.tile(inputs[:, -1:, self.label_index :], [1, 24, 1])
        # 즉 3개 컬럼 별로, 24시간은 같은 값을 가지는 32개 묶음... 32x24x2 반환
        return tf.tile(inputs[:, -1:, self.label_index :], [1, 24, 1])


class RepeatBaseline(Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        return inputs[:, :, self.label_index :]


# DataWindows를 매개변수로 받고, patience는 주어진 에포크동안 val 손실이 개선되지 않을 때 중단
def compile_and_fit(model, window, patience=3, max_epochs=50):
    # 이게 patience를 val 손실과 연결시키는 부분...
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(),
        metrics=[MeanAbsoluteError()],
        # 요건 필요할 때 텐서 값 보려고 내가 넣은 코드...
        run_eagerly=True,
    )
    # 모델 학습 관련 데이터들이 history에 저장되는 모양...
    history = model.fit(
        # train으로 학습
        window.train,
        epochs=max_epochs,
        # val로 검증
        validation_data=window.val,
        # early_stopping 콜백으로 적용
        callbacks=[early_stopping],
    )
    return history


def update_pf_stats(key, valValue, testValue):
    # 귀찮으니 파일 이름은 고정하고...
    file_name = "pf_stats.pkl"
    # 이전에 저장해놓은 성능 지표값이 있으면 읽어서...
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            val_dict, test_dict = pickle.load(f)
            # 받은 키 값이 없으면 새로 넣고, 있으면 업데이트...
            val_dict[key] = valValue
            test_dict[key] = testValue
    else:
        # 파일이 없다는 건 처음 지표를 저장한다는 거니까, 새로 딕셔너리 만들고
        val_dict = {key: valValue}
        test_dict = {key: testValue}

    # 어쨌거나 업데이트됐으니 딕셔너리를 튜플로 저장하고
    with open(file_name, "wb") as f:
        pickle.dump((val_dict, test_dict), f)

    # 과거 지표까지 포함된 사전을 반환
    return val_dict, test_dict


def compare_pf_stats(val_pf: dict, test_pf: dict):
    if isinstance(list(val_pf.values())[0], (float, int)):
        mae_val = val_pf.values()
        mae_test = test_pf.values()
    else:
        mae_val = [v[1] for v in val_pf.values()]
        mae_test = [v[1] for v in test_pf.values()]

    x = np.arange(len(test_pf))

    fig, ax = plt.subplots()
    ax.bar(
        x - 0.15,
        mae_val,
        width=0.25,
        color="black",
        edgecolor="black",
        label="Validation",
    )
    ax.bar(
        x + 0.15,
        mae_test,
        width=0.25,
        color="white",
        edgecolor="black",
        hatch="/",
        label="Test",
    )
    ax.set_ylabel("MAE")
    ax.set_xlabel("Models")
    for index, value in enumerate(mae_val):
        plt.text(x=index - 0.15, y=value + 0.0025, s=str(round(value, 3)), ha="center")
    for index, value in enumerate(mae_test):
        plt.text(x=index + 0.15, y=value + 0.0025, s=str(round(value, 3)), ha="center")
    plt.ylim(0, 0.4)
    ax.set_xticks(ticks=x, labels=test_pf.keys())
    # ax.set_xticklabels(list(test_pf.keys()), rotation=45)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


class AutoRegressive(Model):
    # LSTMCell, RNN, Dense 층을 이용해서 네트워크를 만드는 모양...
    def __init__(self, units, out_steps):
        super().__init__()
        # 예측을 몇 시간 할지, 뉴런은 몇 개인지...인가? 근데 뉴런 수와 출력 수는 별개냐?
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = LSTMCell(units)
        # LSTMCell의 출력을 래핑하여 상태를 추적하는 RNN 레이어
        self.lstm_rnn = RNN(self.lstm_cell, return_state=True)
        # train_df.shape[1]라는데...뭔 이런...절대로 이해가 되질 않는다...
        self.dense = Dense(5)

    # 자기회귀...라는 방법으로 예측을 다시 넣어서 예측을 하는데, 그 중 첫 번째 예측을 뽑는 과정이라고...
    def warmup(self, inputs):
        # 입력을 RNN에 넣어서 결과(출력과 상태)를 받고
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, units)
        # state는 가변인자...
        x, *state = self.lstm_rnn(inputs)

        # 그걸 Dense 층에 넣어 예측을 받기...
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    # 루프를 돌면서 24개 예측을 만든다고...케라스에 의해 암시적으로 호출된다고...이름을 무조건 call로..
    def call(self, inputs, training=None):
        # 예측값을 저장할 리스트
        predictions = []
        # 초기 예측값과 상태를 얻음 - 케라스가 알아서 call 부르고 그게 또 warmup 불러서 작동하나?
        prediction, state = self.warmup(inputs)

        # 첫 번째 예측값 저장
        predictions.append(prediction)

        # 나머지 (out_steps - 1) 만큼 돌면서 예측...
        for _ in range(1, self.out_steps):
            # 이전 예측값을 입력으로 사용하고,
            x = prediction
            # 나머지는 warmup 예측처럼 RNN -> Dense 통과해 예측
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)

        # predictions.shape => (시간 out_steps, batch, 컬럼 1) 모양이 된다고...
        predictions = tf.stack(predictions)
        # 그걸 (batch, out_steps, 1) 모양으로 변경
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
