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

# 1. 데이터 훑어보기
df = pd.read_csv("./data/20220819_BTC-USDT.csv")
xticks = (
    [0, 119, 239, 359, 479, 599, 719, 839, 959, 1079, 1199, 1319, 1439],
    ["00", "02", "04", "06", "08", "10", "12", "14", "16", "18", "20", "22", "24"],
)

btc = np.array(df["close"])
# 3. 1차 차분 데이터 만들고 ACF까지
# btc = np.diff(btc["close"], n=1)

# draw_line_chart(np.arange(len(btc)), btc, "BTC/USDT", "Time", "Price", xticks)

# 3. ADF 테스트와 ACF
test_ADF(btc)
draw_auto_corr(btc)

# 4. PACF 차트로 p 정하기
draw_pacf(btc, lags=20)

# 5. 훈련/테스트 분리 - 나중에 차트 위해서 df 사용
df = pd.DataFrame({"value": btc})
cut = int(len(df) * 0.8)
train = df.iloc[:cut]
test = df.iloc[cut:]
print(len(train), len(test))

# SARIMAX AR로 예측 실행
TRAIN_LEN = len(train)
HORIZON = len(test)
# 윈도우 크기는 어떻게 정하는거야?
WINDOW = 3

pred_mean = roll_fore_vec(btc, TRAIN_LEN, HORIZON, WINDOW, "mean")
pred_last = roll_fore_vec(btc, TRAIN_LEN, HORIZON, WINDOW, "last")
# 그냥 마지막 값 예측과 거의 같네...조건을 만족하질 못해서 그런가?
pred_AR = roll_fore_vec(btc, TRAIN_LEN, HORIZON, WINDOW, "AR", (3, 0, 0))

test["pred_mean"] = pred_mean
test["pred_last"] = pred_last
test["pred_AR"] = pred_AR

print(test.head())

# 예측 결과 그려보고...
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df["value"])
ax.plot(test["value"], "b-", label="Actual")
ax.plot(test["pred_mean"], "g:", label="Mean")
ax.plot(test["pred_last"], "r-.", label="Last Value")
ax.plot(test["pred_AR"], "k--", label="AR(1)")
# 범례, 제목 등
ax.legend(loc=2)
ax.set_title("AR Process")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
# 모델 예측 구간에 회색 표시
ax.axvspan(TRAIN_LEN, TRAIN_LEN + HORIZON, color="lightgray", alpha=0.5)
# 꽉 채워 그리기...
plt.tight_layout()
plt.show()

# MSE 결과 - 이것도 960정도...안되는구나...
mse_mean = mean_squared_error(test["value"], test["pred_mean"])
mse_last = mean_squared_error(test["value"], test["pred_last"])
mse_AR = mean_squared_error(test["value"], test["pred_AR"])

print(f"Mean MSE: {mse_mean}")
print(f"Last MSE: {mse_last}")
print(f"AR MSE: {mse_AR}")

fig, ax = plt.subplots(figsize=(12, 6))

x = ["Mean", "Last", "AR(3)"]
y = [mse_mean, mse_last, mse_AR]

ax.bar(x, y, width=0.4)
ax.set_title("Mean Squared Error")
ax.set_xlabel("Method")
ax.set_ylabel("MSE")

for i, v in enumerate(y):
    plt.text(x=i, y=v + 0.25, s=str(round(v, 2)), ha="center")

plt.tight_layout()
plt.show()

# # 원래 스케일로 복원해서 실제값과 AR 결과 비교
# btc["pred_foot_traffic"] = pd.Series()
# btc.loc[948:, "pred_foot_traffic"] = (
#     btc["foot_traffic"].iloc[948] + test["pred_AR"].cumsum()
# )

# fig, ax = plt.subplots(figsize=(12, 6))

# ax.plot(btc["foot_traffic"], "b-", label="Actual")
# ax.plot(btc["pred_foot_traffic"], "r--", label="AR(3)")

# ax.set_xlabel("Year")
# ax.set_ylabel("Foot Traffic")

# ax.axvspan(948, 1000, color="#808080", alpha=0.2)
# ax.legend(loc=2)

# ax.set_xlim(920, 1000)
# ax.set_ylim(650, 770)

# fig.autofmt_xdate()
# plt.tight_layout()
# plt.show()

# # 원래 스케일의 MAE - 실제 스케일에서 3.4 정도...
# mae_AR_undiff = mean_absolute_error(
#     btc["foot_traffic"][948:999], btc["pred_foot_traffic"][948:999]
# )
# print(f"AR(3) MAE: {mae_AR_undiff}")
