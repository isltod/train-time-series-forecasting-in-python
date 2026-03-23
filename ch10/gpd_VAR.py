import statsmodels.api as sm
import sys

sys.path.append("./")
from util import *
from statsmodels.tsa.stattools import grangercausalitytests


# 1. 데이터 훑어보기
macro_econ_data = sm.datasets.macrodata.load_pandas().data
print(macro_econ_data.head())

x = macro_econ_data.index
y1 = macro_econ_data.realdpi
y2 = macro_econ_data.realcons
title1 = "Real DPI"
title2 = "Real Consumption"
xtics = (np.arange(0, 208, 16), np.arange(1959, 2010, 4))

# draw_2line_chart(x, y1, y2, title1, title2, xtics)

# 2. ADF 테스트 - 둘 다 실패, 1차 차분은 둘 다 성공
print("ADF Test for Real DPI------------------------------")
test_ADF(y1)
print("ADF Test for Real Consumption---------------------")
test_ADF(y2)

diff1 = np.diff(y1, n=1)
diff2 = np.diff(y2, n=1)
print("1st Diff Test for Real DPI------------------------")
test_ADF(diff1)
print("1st Diff Test for Real Consumption----------------")
test_ADF(diff2)

d = 1

# 3. AIC 비교, 최대 15차 이내
max_p = 15
endog = macro_econ_data[["realdpi", "realcons"]]
endog_diff = macro_econ_data[["realdpi", "realcons"]].diff()[1:]

# VAR(p)는 포함된 모든 시계열이 stationary 해야 한다...
cut = 162
train = endog_diff[:cut]
test = endog_diff[cut:]

# result_df = optimize_VARMAX(train, max_p)
# print(result_df)

# 4. Granger Casuality Test
print()
print("realcons Granger-causes realdpi?------------------")
# 두 번째가 첫 번째를 유발하는가 테스트...
granger1 = grangercausalitytests(
    macro_econ_data[["realdpi", "realcons"]].diff()[1:], [3]
)
print(granger1)
print("\n")
print("realcons Granger-causes realdpi?------------------")
granger2 = grangercausalitytests(
    macro_econ_data[["realcons", "realdpi"]].diff()[1:], [3]
)
print(granger2)
print()

# 5. 잔차 분석
# result = resid_VARMAX(train, 3)

# 6. 예측
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 4

# 컬럼명의 딕셔너리로 받아서
preds_last = roll_fore_mat(endog, TRAIN_LEN, HORIZON, WINDOW, "last")
preds_VAR = roll_fore_mat(endog_diff, TRAIN_LEN, HORIZON, WINDOW, "VAR", 3)

# 뭔가 엉망진창 복잡한데...위에서 만들었던 test를 다르게 다시 만드네...
cut = 163
test = endog[163:]
for key, pred in preds_last.items():
    test[key + "_last"] = pred
for key, pred in preds_VAR.items():
    test[key + "_VAR"] = np.cumsum(pred) + endog.iloc[162][key]

print(test)

# 일단 두 가지의 key가 같다는 가정하에...
preds = []
ys = []
titles = []
ylables = []
for key in preds_VAR.keys():
    ys.append(macro_econ_data[key])
    titles.append(key)
    ylables.append(key)
    key1 = key + "_last"
    key2 = key + "_VAR"
    pred_dict = {key1: test[key1], key2: test[key2]}
    preds.append(pred_dict)
vs = len(endog) - cut
# draw_pred_mat(x, ys, vs, preds, titles, "Year", ylables, xtics)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mape_realdpi_var = mape(test["realdpi"], test["realdpi_VAR"])
mape_realcons_var = mape(test["realcons"], test["realcons_VAR"])
mape_realdpi_last = mape(test["realdpi"], test["realdpi_last"])
mape_realcons_last = mape(test["realcons"], test["realcons_last"])
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12, 12))
x = ["last", "VAR"]
y1 = [mape_realdpi_last, mape_realdpi_var]
y2 = [mape_realcons_last, mape_realcons_var]
ax1.bar(x, y1, width=0.4)
ax1.set_title("Real DPI")
ax1.set_xlabel("Method")
ax1.set_ylabel("MAPE")
ax1.set_ylim(0, 3.5)
ax2.bar(x, y2, width=0.4)
ax2.set_title("Real Consumption")
ax2.set_xlabel("Method")
ax2.set_ylabel("MAPE")
ax2.set_ylim(0, 3.5)
for i, v in enumerate(y1):
    ax1.text(x=i, y=v * 1.1, s=str(round(v, 2)), ha="center")
for i, v in enumerate(y2):
    ax2.text(x=i, y=v * 1.1, s=str(round(v, 2)), ha="center")
plt.tight_layout()
plt.show()
