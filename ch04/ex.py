import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append("./")
from util import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess


# 1. 1000개 샘플 생성
np.random.seed(42)

ma2 = np.array([1, 0.9, 0.3])
ar2 = np.array([1, 0, 0])

MA2_process = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)

# 2. 차트 그리고
draw_line_chart(
    np.arange(len(MA2_process)), MA2_process, "MA(2) Process", "Time", "Value"
)

# 3. ADF 테스트
test_ADF(MA2_process)

# 4. ACF 차트
draw_auto_corr(MA2_process)

# 5. 훈련/테스트 분리 - 나중에 차트 위해서 df 사용
df = pd.DataFrame({"value": MA2_process})
train = df.iloc[:800]
test = df.iloc[800:]

# 6. 평균, 마지막 값, MA(2) 모델 예측
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2
pred_mean = rolling_forecast(MA2_process, TRAIN_LEN, HORIZON, WINDOW, "mean")
pred_last = rolling_forecast(MA2_process, TRAIN_LEN, HORIZON, WINDOW, "last")
pred_MA = rolling_forecast(MA2_process, TRAIN_LEN, HORIZON, WINDOW, "MA")

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
