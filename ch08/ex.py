import sys

sys.path.append("./")
from util import *

from itertools import product


# 1. 데이터 훑어보기
df = pd.read_csv("./data/jj.csv")
print(df.head())
col_name = "data"
x = np.arange(len(df))
y = df[col_name].to_numpy()
title = "J&J Earning Per Share"
xlabel = "Quater"
ylabel = "Per Share"
xticks = (
    np.arange(0, 81, 8),
    [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980],
)
# draw_line_chart(x, y, title, xlabel, ylabel, xticks)

# 2. 계절성 차트표시
# marks = np.arange(6, 145, 12)
# vlines = np.arange(0, 145, 12)
# draw_seasonality(x, y, title, xlabel, ylabel, xticks, marks, vlines)

# 3. 추세와 계절성 분리
period = 4
# draw_seasonal_decompose(x, y, period, xticks)

# 4. d, D 결정위해 ADF 테스트 - d는 1, D도 1?
print("ADF Test for Original Data--------------------------------------")
test_ADF(y)

print("ADF Test for 1st Diff Data--------------------------------------")
diff = np.diff(y, n=1)
test_ADF(diff)

print("ADF Test for 2nd Diff Data--------------------------------------")
diff2 = np.diff(y, n=2)
test_ADF(diff2)
# ARIMA로 돌릴려면 이걸 이용하고...

# 계절성 적분 차수 D는 d와 쌍으로 결정...d=1은 정상성이 없지만, 거기에 D=1을 더하면 stationary...
print("ADF Test for Seasonal Diff of 1st Diff Data---------------------")
season_diff = np.diff(diff, n=period)
test_ADF(season_diff)
# SARIMA로 돌릴려면 d=1, D=1 이걸 이용한다...

# 5. 훈련/테스트 분리
cut = 4
train = df[col_name][:-cut]
test = pd.DataFrame(df[col_name][-cut:])
# draw_train_test(x, y, cut, diff2, title, xlabel, ylabel, xticks)

# 6. SARIMA AIC 비교를 하는데...계절성을 0으로 놓고, p,q를 12개까지?
ps = range(0, 13)
qs = range(0, 13)
Ps = [0]
Qs = [0]
d = 2
D = 0
s = period
ARIMA_order_list = list(product(ps, qs, Ps, Qs))
# result_df = optimize_SARIMA(train, ARIMA_order_list, d, D, s)
# print(result_df)

# 7. ARIMA(11,2,3)의 잔차 분석
p = 11
d = 2
q = 3
# ARIMA_result = analysis_residual(train, p, d, q)

# 8. SARIMA AIC 비교해서 잔차 분석
ps = range(0, 4)
qs = range(0, 4)
Ps = range(0, 4)
Qs = range(0, 4)
SARIMA_order_list = list(product(ps, qs, Ps, Qs))
d = 1
D = 1
s = period
SARIMA_result_df = optimize_SARIMA(train, SARIMA_order_list, d, D, s)
print(SARIMA_result_df)

sel_idx = 0
sel = SARIMA_result_df.iloc[sel_idx, 0]
p, q, P, Q = sel

# p, q, P, Q = 2, 1, 1, 2
SARIMA_result = resid_SARIMAX(train, p, d, q, P, D, Q, s)

# 9. 예측 비교
# test["naive_seasonal"] = y[-(cut + s) : -cut]
# p, d, q = 11, 2, 3
# ARIMA_pred = (
#     modelling(train, p, d, q).get_prediction(len(y) - cut, len(y) - 1).predicted_mean
# )
# test["ARIMA"] = ARIMA_pred
# p, d, q = 2, 1, 1
SARIMA_pred = (
    model_SARIMAX(train, p, d, q, P, D, Q, s)
    .get_prediction(len(y) - cut, len(y) - 1)
    .predicted_mean
)
test["SARIMA"] = SARIMA_pred

# # 수치가 뭔가 비슷하면서 약간씩 틀리는데...이거 왜 이렇지?
print(test)

pred_dict = test.iloc[:, 1:].to_dict(orient="list")
draw_pred_vec(x, y, cut, pred_dict)
compare_MAPE(test[col_name], pred_dict)
