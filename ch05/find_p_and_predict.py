import sys

sys.path.append("./")
from util import *
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_pacf

matplotlib.use("TkAgg")

# 1. 데이터 훑어보기 - 가상 데이터로, 2000~2020년 주간 통행량? 데이터...
df = pd.read_csv("./data/foot_traffic.csv")
print(df.head())

xticks = (np.arange(0, 1000, 104), np.arange(2000, 2020, 2))

draw_line_chart(
    np.arange(len(df)), df["foot_traffic"], "Foot Traffic", "Time", "Count", xticks
)

# 2. 원본 데이터의 ADF 테스트와 ACF - 당연히 통과 안되고, 자기상관도 유의미하고 선형적으로 감소
test_ADF(df["foot_traffic"])
draw_auto_corr(df["foot_traffic"])

# 3. 1차 차분 데이터 만들고 ACF까지
diff = np.diff(df["foot_traffic"], n=1)
# 장기추세는 제거됐지만 그냥 소음은 아니고 뭔가 오르락 내리락...
draw_line_chart(
    np.arange(len(diff)), diff, "Foot Traffic Diff", "Time", "Count", xticks
)
# p값이 10^-6 오더로 정상성은 통과인데...
test_ADF(diff)
# 자기상관이 0 단계 이후에도 유의미하고(랜덤워크 아니고), 기하급수적(선형으로도 보이는데...)으로 감소(AR 과정)
draw_auto_corr(diff)

# PACF 차트로 p 정하기 - 3차
plot_pacf(diff, lags=20)
plt.tight_layout()
plt.show()

# 마지막 1년을 예측 시기로 구분
df_diff = pd.DataFrame({"foot_traffic_diff": diff})
train = df_diff[:-52]
test = df_diff[-52:]
print(len(train), len(test))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

ax1.plot(df["foot_traffic"])
ax1.set_ylabel("Foot Traffic Weekly")
ax1.axvspan(948, 1000, color="#808080", alpha=0.2)

ax2.plot(df_diff["foot_traffic_diff"])
ax2.set_xlabel("Year")
ax2.set_ylabel("Foot Traffic Diff")
ax2.axvspan(947, 999, color="#808080", alpha=0.2)

plt.tight_layout()
plt.show()

# SARIMAX AR로 예측 실행
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(diff, TRAIN_LEN, HORIZON, WINDOW, "mean")
pred_last = rolling_forecast(diff, TRAIN_LEN, HORIZON, WINDOW, "last")
pred_AR = rolling_forecast(diff, TRAIN_LEN, HORIZON, WINDOW, "AR")

test["pred_mean"] = pred_mean
test["pred_last"] = pred_last
test["pred_AR"] = pred_AR

print(test.head())

# 예측 결과 그려보고...
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df_diff["foot_traffic_diff"])
ax.plot(test["foot_traffic_diff"], "b-", label="Actual")
ax.plot(test["pred_mean"], "g:", label="Mean")
ax.plot(test["pred_last"], "r-.", label="Last")
ax.plot(test["pred_AR"], "k--", label="AR(3)")

ax.set_xlabel("Year")
ax.set_ylabel("Foot Traffic Diff")
ax.set_xlim(920, 999)
plt.xticks([936, 988], [2018, 2019])

ax.axvspan(947, 999, color="#808080", alpha=0.2)
ax.legend(loc=2)

plt.tight_layout()
plt.show()

# MSE 결과
mse_mean = mean_squared_error(test["foot_traffic_diff"], test["pred_mean"])
mse_last = mean_squared_error(test["foot_traffic_diff"], test["pred_last"])
mse_AR = mean_squared_error(test["foot_traffic_diff"], test["pred_AR"])

# AR 모델 결과는 0.92, 대충 오차가 1이라는 얘긴가?
print(f"Mean MSE: {mse_mean}")
print(f"Last MSE: {mse_last}")
print(f"AR MSE: {mse_AR}")

fig, ax = plt.subplots(figsize=(12, 6))

x = ["Mean", "Last", "AR(3)"]
y = [mse_mean, mse_last, mse_AR]

ax.bar(x, y, width=0.4)
ax.set_xlabel("Method")
ax.set_ylabel("MSE")

for i, v in enumerate(y):
    plt.text(x=i, y=v + 0.25, s=str(round(v, 2)), ha="center")

plt.tight_layout()
plt.show()

# 원래 스케일로 복원해서 실제값과 AR 결과 비교
df["pred_foot_traffic"] = pd.Series()
df.loc[948:, "pred_foot_traffic"] = (
    df["foot_traffic"].iloc[948] + test["pred_AR"].cumsum()
)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df["foot_traffic"], "b-", label="Actual")
ax.plot(df["pred_foot_traffic"], "r--", label="AR(3)")

ax.set_xlabel("Year")
ax.set_ylabel("Foot Traffic")

ax.axvspan(948, 1000, color="#808080", alpha=0.2)
ax.legend(loc=2)

ax.set_xlim(920, 1000)
ax.set_ylim(650, 770)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# 원래 스케일의 MAE - 실제 스케일에서 3.4 정도...
mae_AR_undiff = mean_absolute_error(
    df["foot_traffic"][948:999], df["pred_foot_traffic"][948:999]
)
print(f"AR(3) MAE: {mae_AR_undiff}")
