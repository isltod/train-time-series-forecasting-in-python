import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    # SARIMAX MA, AR, ARMA 모델 예측
    elif mthd in ["MA", "AR", "ARMA", "ARIMA", "SARIMA", "SARIMAX"]:
        pred_SARIMAX = []

        for t in tqdm(range(trn_len, total_len, wndw)):
            # order는 p, d, q인데, 자기회귀 차수 p도 0, 차분 d도 0? 이동 평균 차수만 넣는다?
            # 그럼 p, d는 아예 원 데이터를 넣고 돌릴 때 지정하는 건가?
            model = SARIMAX(
                ts[:t],
                exog=exog[:t],
                order=ordr,
                seasonal_order=ssnl_ordr,
                simple_differencing=False,
            )
            res = model.fit(disp=False)
            if exog is None:
                # 학습 후 예측인데, 0에서 시작, 마지막 원소 (t - 1) 이후로 window 크기 2만큼 예측
                pred = res.get_prediction(0, (t - 1) + wndw)
            else:
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
