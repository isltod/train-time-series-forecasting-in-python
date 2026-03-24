import sys

sys.path.append("./")
from util import *
from itertools import product
import warnings

warnings.filterwarnings("ignore")


# 1. 데이터 읽기
df = pd.read_csv("./data/AusAntidiabeticDrug.csv")
print(df.head())
print(df.shape)

x = np.arange(len(df))
y = df.y.to_numpy()
period = 12
title = "Capstone Project"
xtics = (np.arange(6, 203, period), np.arange(1992, 2009))

# 2. 계절성 확인 - SARIMA
# draw_seasonal_decompose(x, y, period, xtics)

# 3. ADF 테스트로 d, D 결정 - SARIMA(p,1,q)(P,1,Q)12
# print("ADF Test for Original Data------------------------------")
# test_ADF(y)
# print("1st Diff Test for Original Data-------------------------")
# diff = np.diff(y, n=1)
# test_ADF(diff)
# print("1st Seasonal Diff for 1st Diff Data---------------------")
# sdiff = np.diff(diff, n=period)
# test_ADF(sdiff)

d = 1
D = 1

# 4. 훈련/시험 데이터 분리 - 36개월 예측
cut = 36
train = y[:-cut]
test = y[-cut:]
print(len(train), len(test))

# 5. AIC 비교로 p,q,P,Q 선택 -
# 내 코드, 노트북을 옮긴 코드, 노트북 모두 다른 결과가 나온다...어떻게 이럴 수가 있나?
ps = range(0, 4)
qs = range(0, 4)
Ps = range(0, 4)
Qs = range(0, 4)
order_list = list(product(ps, qs, Ps, Qs))
# SARIMA_result_df = optimize_SARIMA(train, order_list, d, D, s=period)
# SARIMA_result_df = optimize_SARIMA(train, [(2, 3, 1, 3)], d, D, s=period)
# print(SARIMA_result_df)
# 일단 책대로 2,3,1,3 쓴다...

p, q, P, Q = 2, 3, 1, 3

# 6. 잔차 분석 - 여기 AIC 결과가 276나온다...즉 책은 일단 틀렸다는 얘기...
# SARIMA_result = resid_SARIMAX(train, p, d, q, P, D, Q, s=period)

# 7. 롤링 예측
TRAIN_LEN = len(train)
HRZN = len(test)
WINDOW = 12

pred_dict = {}

pred_dict["last_season"] = roll_fore_vec(y, TRAIN_LEN, HRZN, WINDOW, "last_season")
pred_dict["SARIMA"] = roll_fore_vec(
    y, TRAIN_LEN, HRZN, WINDOW, "SARIMA", (p, d, q), (P, D, Q, period)
)

pred_df = pd.DataFrame(pred_dict)
print(pred_df)

# 8. 시각화
draw_pred_vec(x, y, cut, pred_dict, ttl=title, xlbl="X", ylbl="Y", xticks=xtics)
compare_MAPE(test, pred_dict)
