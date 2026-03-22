import sys

sys.path.append("./")
from util import *

from itertools import product


# 1. 데이터 훑어보기
df = pd.read_csv("./data/air-passengers.csv")
print(df.head())
col_name = "Passengers"
x = np.arange(len(df))
y = df[col_name].to_numpy()
title = "Air Passengers"
xlabel = "Month"
ylabel = "Passengers"
xticks = (np.arange(0, 145, 12), np.arange(1949, 1962))
# draw_line_chart(x, y, title, xlabel, ylabel, xticks)

# 2. 계절성 차트표시
marks = np.arange(6, 145, 12)
vlines = np.arange(0, 145, 12)
# draw_seasonality(x, y, title, xlabel, ylabel, xticks, marks, vlines)

# 3. 추세와 계절성 분리
period = 12
# draw_seasonal_decompose(x, y, period, xticks)
# 계절성 없으면 Seasonal 차트가 평평하게 나온다고...
# lienar_ts = np.arange(0, 144)
# draw_seasonal_decompose(x, lienar_ts, period, xticks)

# 4. d, D 결정위해 ADF 테스트 - d는 1, D도 1?
print("ADF Test for Original Data--------------------------")
test_ADF(y)

print("ADF Test for 1st Diff Data--------------------------")
diff = np.diff(y, n=1)
test_ADF(diff)

print("ADF Test for 2nd Diff Data--------------------------")
diff2 = np.diff(y, n=2)
test_ADF(diff2)
# ARIMA로 돌릴려면 이걸 이용하고...

print("ADF Test for Seasonal Diff Data---------------------")
season_diff = np.diff(y, n=period)
test_ADF(season_diff)
# SARIMA로 돌릴려면 d=1, D=1 이걸 이용한다...

# 5. 훈련/테스트 분리
cut = 12
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
# analysis_residual(train, p, d, q)

# 일단...예측 비교를 준비하자...
# test["naive_seasonal"] = y[-(cut + s) : -cut]
# print(test)
# ARIMA_pred = (
#     modelling(train, p, d, q).get_prediction(len(y) - cut, len(y) - 1).predicted_mean
# )
# test["ARIMA"] = ARIMA_pred
# # 수치가 뭔가 비슷하면서 약간씩 틀리는데...이거 왜 이렇지?
# print(test)
