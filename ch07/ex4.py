from itertools import product
from statsmodels.stats.diagnostic import acorr_ljungbox
import sys

sys.path.append("./")
from util import *

# 1. 일단 AIC 비교는 해야하니까, order list 만들기
ps = range(0, 4)
qs = range(0, 4)
order_list = list(product(ps, qs))
print(order_list)

# 2. 5장에 310 적용
# 2.1 데이터 읽고 훈련/테스트 분리 - 어차피 ARIMA라서 차분 없이 사용
#  ADF 테스트 - 실패, 1차 차분 필요
df7 = pd.read_csv("./data/20220819_BTC-USDT.csv")
print(df7.head())
col_name = "close"
cut = 288
train = df7[col_name][:-cut]
test = pd.DataFrame(df7[col_name][-cut:])


# 2.2 정상성 테스트 - 여기도 1차 차분 성공, d=1
test_ADF(df7[col_name])
df4_diff = np.diff(df7[col_name], n=1)
test_ADF(df4_diff)
d = 1

# 2.3 AIC 비교 - 012 아니고 313이 가장 낮은 AIC
result_df = optimize_SARIMA(train, order_list, d)
print(result_df)

# 2.4 최소 AIC로 모델 적용하고 잔차 분석 - 적절하면 베이스 모델과 비교
sel = 0
p, q = result_df.iloc[sel, 0][0], result_df.iloc[sel, 0][1]
print("p:", p, "q:", q)
model = SARIMAX(train, order=(p, d, q), simple_differencing=False)
result = model.fit()
print(result.summary())
result.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()
lb = acorr_ljungbox(result.resid, 10)
print(lb["lb_pvalue"])

# 2.5 베이스 모델과 비교
TRAIN_LEN = len(train)
HORIZON = len(test)
# 여긴 1인데, 근데 이 윈도우는 어떻게 정하는 거냐...
WINDOW = 3
ts = df7[col_name].to_numpy()

pred_mean = rolling_forecast(ts, TRAIN_LEN, HORIZON, WINDOW, "mean")
pred_last = rolling_forecast(ts, TRAIN_LEN, HORIZON, WINDOW, "last")
pred_ARIMA = rolling_forecast(ts, TRAIN_LEN, HORIZON, WINDOW, "ARIMA", (p, d, q))

test["pred_mean"] = pred_mean
test["pred_last"] = pred_last
test["pred_ARIMA"] = pred_ARIMA

print(test.head())

# 2.6 예측 결과 그리기
preds = test.iloc[:, 1:]
pred_dict = preds.to_dict(orient="list")
draw_predicts(np.arange(len(df7)), df7[col_name], len(test), pred_dict)

# 2.7 MSE 비교
compare_MAE(test[col_name], pred_dict)
