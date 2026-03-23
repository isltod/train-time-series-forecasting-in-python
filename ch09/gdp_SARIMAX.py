import sys

sys.path.append("./")
import statsmodels.api as sm
from itertools import product
from util import *

# 1. 데이터 불러와서 목표 변수와 외생 변수 구분
macro_econ_data = sm.datasets.macrodata.load_pandas().data
print(macro_econ_data.head())

target = macro_econ_data.realgdp.to_numpy()
exog = macro_econ_data[["realcons", "realinv", "realgovt", "realdpi", "cpi"]]
print(exog.head())

# 2. 목표 변수의 ADF 테스트 - 실패, 1차 차분 성공, d=1
test_ADF(target)

diff = np.diff(target, n=1)
test_ADF(diff)

d = 1
D = 0

# 3. AIC 비교로 p, q, P, Q 결정
ps = range(0, 4)
qs = range(0, 4)
Ps = range(0, 4)
Qs = range(0, 4)
s = 4

order_list = list(product(ps, qs, Ps, Qs))

# 근데 이건 train이 AIC 비교할 때랑 나중에 예측할 때 달라진다?
target_train = target[:200]
exog_train = exog[:200]

# 아무튼 비교...
# result_df = optimize_SARIMA(target_train, order_list, d, D, s, exog_train)
# print(result_df)

sel_idx = 0
# sel = result_df.iloc[sel_idx, 0]
# p, q, P, Q = sel
p, q, P, Q = 3, 3, 0, 0
print("p:", p, "q:", q, "P:", P, "Q:", Q)

# 4. 잔차 분석
result = resid_SARIMAX(target_train, p, d, q, P, D, Q, s, exog_train)

# 5. 예측 - 여긴 rolling forecast
cut = 196
target_train = target[:cut]
target_test = target[cut:]

TRAIN_LEN = len(target_train)
HORIZON = len(target_test)
WINDOW = 1

pred_last = roll_fore_vec(target, TRAIN_LEN, HORIZON, WINDOW, "last")
pred_SARIMAX = roll_fore_vec(
    target, TRAIN_LEN, HORIZON, WINDOW, "SARIMAX", (p, d, q), (P, D, Q, s), exog
)
pred_df = pd.DataFrame({"actual": target_test})
pred_df["last"] = pred_last
pred_df["SARIMAX"] = pred_SARIMAX
print(pred_df.head())

# 6. 비교
predicts = pred_df.iloc[:, 1:]
print(predicts)
pred_dict = predicts.to_dict(orient="list")
compare_MAPE(target_test, pred_dict)
vs = len(target) - cut
draw_pred_vec(np.arange(len(target)), target, vs, pred_dict)
