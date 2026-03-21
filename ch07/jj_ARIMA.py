from itertools import product
from statsmodels.stats.diagnostic import acorr_ljungbox
import sys

sys.path.append("./")
from util import *

# Johnson & Jhonson 데이터
df = pd.read_csv("./data/jj.csv")
print(df.head())

title = "J&J Earning Per Share"
xlabel = "Year"
ylabel = "Earning per Share ($)"
xtics = (
    np.arange(0, 81, 8),
    [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980],
)

# draw_line_chart(np.arange(len(df)), df["data"], title, xlabel, ylabel, xtics)

# ADF 테스트
test_ADF(df["data"])

# 1차 차분
diff = np.diff(df["data"], n=1)
test_ADF(diff)

# 2차 차분 - 차수 d는 2
diff2 = np.diff(df["data"], n=2)
test_ADF(diff2)

# draw_line_chart(np.arange(len(diff2)), diff2, title, xlabel, ylabel, xtics)

# 훈련/테스트 분리
cut = 4
y = df["data"]
train = y[:-cut]
test = df[-cut:]
print(len(train), len(test))

# draw_train_test(np.arange(len(df)), df["data"], cut, None, title, xlabel, ylabel, xtics)

# AIC 비교
ps = range(0, 4)
qs = range(0, 4)
d = 2
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train, order_list, d)
print(result_df)

# 잔차분석
p, q = result_df.iloc[0, 0][0], result_df.iloc[0, 0][1]
model = SARIMAX(train, order=(p, d, q), simple_differencing=False)
result = model.fit()
print(result.summary())
# result.plot_diagnostics(figsize=(12, 8))
# plt.tight_layout()
# plt.show()

residuals = result.resid
lb = acorr_ljungbox(residuals, 10)
print(lb["lb_pvalue"])

# 예측
# 베이스라인은 직전 4분기를 그대로...
test["naive_seasonal"] = df["data"].iloc[76:80].values
# ARIMA예측은 롤링 안하고 위에 fitting 결과에서 바로 찾아쓴다...왜 이번엔?
ARIMA_pred = result.get_prediction(80, 83).predicted_mean
test["ARIMA"] = ARIMA_pred

predicts = test.iloc[:, 2:]
pred_dict = predicts.to_dict(orient="list")
# draw_predicts(
#     np.arange(len(df)), df["data"], cut, pred_dict, title, xlabel, ylabel, xtics
# )

# MSE 비교
compare_MAPE(test["data"], pred_dict)
