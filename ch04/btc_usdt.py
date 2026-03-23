import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append("./")
from util import *
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_process import ArmaProcess


# 1. 데이터
df = pd.read_csv("./data/20220819_BTC-USDT.csv")
print(df.head())
xticks = (
    [0, 119, 239, 359, 479, 599, 719, 839, 959, 1079, 1199, 1319, 1439],
    ["00", "02", "04", "06", "08", "10", "12", "14", "16", "18", "20", "22", "24"],
)

# 원래 데이터로 하면 MSE가 억 단위가 나오고 완전 엉뚱한 값을 가리킨다...
btc = np.array(df["close"])
# 1차 차분으로 하면 MSE가 500 정도 나오고 대충은 따라다닌다...
# btc = np.diff(df["close"])[:-1]
# 즉 MA 모델은 데이터가 어떻게든 정상성을 가지도록 해야 한다는 것이고,
# 자기상관이 없으면(백색 소음이면) 잘 안된다는 것...


# 2. 차트 그리고
draw_line_chart(np.arange(len(btc)), btc, "MA(2) Process", "Time", "Value")

# 3. ADF 테스트
test_ADF(btc)

# 4. ACF 차트
draw_auto_corr(btc)

# 5. 훈련/테스트 분리 - 나중에 차트 위해서 df 사용
df = pd.DataFrame({"value": btc})
cut = int(len(df) * 0.8)
train = df.iloc[:cut]
test = df.iloc[cut:]

# 6. 평균, 마지막 값, MA(2) 모델 예측
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2
pred_mean = roll_fore_vec(btc, TRAIN_LEN, HORIZON, WINDOW, "mean")
pred_last = roll_fore_vec(btc, TRAIN_LEN, HORIZON, WINDOW, "last")
pred_MA = roll_fore_vec(btc, TRAIN_LEN, HORIZON, WINDOW, "MA", (0, 0, 2))

# 7. 모델 결과 비교 차트...아직은 모듈화 좀 어렵네...일단 한 번 더 만들어보자...
# 그 중 테스트에 모델 결과 붙이고...
test.loc[:, "pred_mean"] = pred_mean
test.loc[:, "pred_last"] = pred_last
test.loc[:, "pred_MA"] = pred_MA
# 차트 그림 만들고
fig, ax = plt.subplots(figsize=(12, 6))
# 전체, 테스트, 모델 결과 순으로 덧그리고
ax.plot(df["value"])
ax.plot(test["value"], "b-", label="Actual")
ax.plot(test["pred_mean"], "g:", label="Mean")
ax.plot(test["pred_last"], "r-.", label="Last Value")
ax.plot(test["pred_MA"], "k--", label="MA(2)")
# 범례, 제목 등
ax.legend(loc=2)
ax.set_title("MA(2) Process")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
# 모델 예측 구간에 회색 표시
ax.axvspan(TRAIN_LEN, TRAIN_LEN + HORIZON, color="lightgray", alpha=0.5)
# 꽉 채워 그리기...
plt.tight_layout()
plt.show()

# 8. MSE 비교 - 요것도 아직 모듈화 하기가...
mse_mean = mean_squared_error(test["value"], test["pred_mean"])
mse_last = mean_squared_error(test["value"], test["pred_last"])
mse_MA = mean_squared_error(test["value"], test["pred_MA"])

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
