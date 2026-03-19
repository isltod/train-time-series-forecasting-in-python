import sys

sys.path.append("./")
from util import *
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("TkAgg")

# 1. AR2 시뮬레이션 데이터 1000개 만들기
np.random.seed(42)

# 130쪽 수식 5.4
ma2 = np.array([1, 0, 0])
ar2 = np.array([1, -0.33, -0.50])

AR2_process = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)

# 2. 그려보자 - 유동인구 데이터의 1차 차분 결과와 유사한 그래프...
draw_line_chart(
    np.arange(len(AR2_process)), AR2_process, "AR(2) Process", "Time", "Value"
)

# 3. ADF 테스트
test_ADF(AR2_process)

# 4. 자기상관 차트
draw_auto_corr(AR2_process, 20)

# 5. PACF 차트
draw_pacf(AR2_process)

# 6. 훈련/테스트 분할
df = pd.DataFrame({"AR2_process": AR2_process})
train = df[:800]
test = df[800:]
print(len(train), len(test))

# 7. AR(2)로 테스트 예측
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2

pred_AR2 = rolling_forecast(AR2_process, TRAIN_LEN, HORIZON, WINDOW, "AR", (2, 0, 0))

test["pred_AR2"] = pred_AR2

print(test.head())

# 8. 예측 결과 그려보기
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df["AR2_process"])
ax.plot(test["AR2_process"], "b-", label="Actual")
ax.plot(test["pred_AR2"], "r--", label="AR(2)")

ax.set_xlabel("Time")
ax.set_ylabel("Value")

ax.axvspan(800, 1000, color="#808080", alpha=0.2)
ax.legend(loc=2)

plt.tight_layout()
plt.show()

# 9. MSE 측정
mse_AR2 = mean_squared_error(test["AR2_process"], test["pred_AR2"])
print(f"AR(2) MSE: {mse_AR2}")

fig, ax = plt.subplots(figsize=(12, 6))

x = ["AR(2)"]
y = [mse_AR2]

ax.bar(x, y, width=0.4)
ax.set_xlabel("Method")
ax.set_ylabel("MSE")

for i, v in enumerate(y):
    plt.text(x=i, y=v + 0.25, s=str(round(v, 2)), ha="center")

plt.tight_layout()
plt.show()
