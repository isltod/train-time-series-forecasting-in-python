from itertools import product
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import sys

sys.path.append("./")
from util import *

# 1. 10,000개 ARMA(2,2) 생성
np.random.seed(42)

ar2 = np.array([1, -0.33, -0.5])
ma2 = np.array([1, 0.9, 0.3])

ARMA_2_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)

# 2. 차트
# draw_line_chart(np.arange(len(ARMA_2_2)), ARMA_2_2, "ARMA(2,2)", "Time", "Value")

# 3. ADF 테스트
test_ADF(ARMA_2_2)

# 4. 훈련/테스트 분리
cut = 200
train = ARMA_2_2[:-cut]
test = ARMA_2_2[-cut:]
print(len(train), len(test))

# 5. p, q 조합 생성
ps = range(0, 4)
qs = range(0, 4)

order_list = list(product(ps, qs))
print(order_list)

# # 6. AIC 비교
# result_df = optimize_ARMA(ARMA_2_2, order_list)
# print(result_df)

# # 7. 모델 적용 후 잔차
# model = SARIMAX(train, order=(2, 0, 2), simple_differencing=False)
# result = model.fit()
# print(result.summary())
# residuals = result.resid

# # 8. 정성적 잔차 분석
# result.plot_diagnostics(figsize=(12, 8))
# plt.tight_layout()
# plt.show()

# # 9. 정량적 잔차 분석
# lb = acorr_ljungbox(residuals, 10)
# print(lb["lb_pvalue"])

# 10. 평균, 마지막 값, ARMA 예측
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2

pred_mean = rolling_forecast(ARMA_2_2, TRAIN_LEN, HORIZON, WINDOW, "mean")
pred_last = rolling_forecast(ARMA_2_2, TRAIN_LEN, HORIZON, WINDOW, "last")
pred_AR = rolling_forecast(ARMA_2_2, TRAIN_LEN, HORIZON, WINDOW, "ARMA", (2, 0, 2))

df = pd.DataFrame({"value": test})
df["pred_mean"] = pred_mean
df["pred_last"] = pred_last
df["pred_AR"] = pred_AR

print(df.head())

# 11. 예측 결과 그리기
preds = df.iloc[:, 1:]
pred_dict = preds.to_dict(orient="list")
draw_predicts(np.arange(len(ARMA_2_2)), ARMA_2_2, len(test), pred_dict)

# 12. MSE 비교
compare_MSE(df["value"], pred_dict)
