import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


np.random.seed(42)

steps = np.random.standard_normal(size=1000)
# 단순화 위해서 63쪽 수식 3.1에서 C = 0, y0 = 0으로...
steps[0] = 0

# 한 발 걸음을 누적합해서 확률보행 결과 만들기...
random_walk = np.cumsum(steps)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(random_walk)
ax.set_title("Random Walk")
ax.set_xlabel("Time")
ax.set_ylabel("Position")

plt.tight_layout()
# plt.show()

# 훈련, 테스트 데이터 분리
df = pd.DataFrame({"value": random_walk})
train = df.iloc[:800]
test = df.iloc[800:]

# 1. 평균으로 예측하기
mean = np.mean(train["value"])
test.loc[:, "pred_mean"] = mean
print(test.head())

# 2. 마지막 값으로 예측하기
last_value = train["value"].iloc[-1]
test.loc[:, "pred_last"] = last_value
print(test.head())

# 표류(drift?) 예측? 선형 외삽 정도...
deltaX = train.size - 1
deltaY = last_value - 0
drift = deltaY / deltaX

x_vals = np.arange(801, 1001, 1)
test.loc[:, "pred_drift"] = drift * x_vals
print(test.head())

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(train.value, "b-")
ax.plot(test.value, "b-")
ax.plot(test.pred_mean, "r-.", label="Mean")
ax.plot(test.pred_last, "g--", label="Last Value")
ax.plot(test.pred_drift, "k:", label="Drift")

ax.axvspan(800, 1000, color="lightgray", alpha=0.5)
ax.legend(loc=2)

ax.set_title("Random Walk")
ax.set_xlabel("Time")
ax.set_ylabel("Position")

plt.tight_layout()
# plt.show()

# MSE 계산 - 자체 값이 0이 있을 수 있어서 MAPE = p-a/a에서 a=0 문제가 있음...
mse_mean = mean_squared_error(test["value"], test["pred_mean"])
mse_last = mean_squared_error(test["value"], test["pred_last"])
mse_drift = mean_squared_error(test["value"], test["pred_drift"])

# 값이 300 이상이 나오는데...실제(test) 값이 30이 안되는데...오류가 70% 이상...
print(f"Max Test Value: {test['value'].max()}")
print(f"MSE Mean: {mse_mean}")
print(f"MSE Last: {mse_last}")
print(f"MSE Drift: {mse_drift}")

fig, ax = plt.subplots(figsize=(12, 6))

x = ["Mean", "Last Value", "Drift"]
y = [mse_mean, mse_last, mse_drift]

ax.bar(x, y, width=0.4)
ax.set_title("Mean Squared Error")
ax.set_xlabel("Model")
ax.set_ylabel("MSE")

plt.tight_layout()
# plt.show()

# 단순 무식하게 이전값이 다음 값이라고 예측하면...
df_shift = df.shift(1)
print(df_shift.head())

# 엄청 잘 된 것처럼 나오는데...
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df, "b-", label="Actual")
ax.plot(df_shift, "r-", label="Forecasted")
ax.legend(loc=2)

ax.set_title("Random Walk")
ax.set_xlabel("Time")
ax.set_ylabel("Position")

plt.tight_layout()
# plt.show()

# 더 확대해서 보면...
fig, ax = plt.subplots(figsize=(12, 6))


ax.plot(df, "b-", label="Actual")
ax.plot(df_shift, "r-.", label="Forecasted")
ax.legend(loc=2)

ax.set_title("Random Walk")
ax.set_xlabel("Time")
ax.set_ylabel("Position")

# 당연히 그저 앞의 값을 따라하기...시각적 착각일 뿐...근데 LSTM 기본 예측이 대충 이런식이었지 않나?
ax.set_xlim(900, 1000)
ax.set_ylim(15, 28)

plt.tight_layout()
plt.show()
