from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import sys

sys.path.append("./")
from util import *

np.random.seed(42)

ar1 = np.array([1, -0.33])
ma1 = np.array([1, 0.9])

ARMA_1_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)

cut = int(len(ARMA_1_1) * 0.8)
train = ARMA_1_1[:cut]
test = ARMA_1_1[cut:]
print(len(train), len(test))

pred_mean = rolling_forecast(ARMA_1_1, cut, len(test), 1, "mean")
pred_last = rolling_forecast(ARMA_1_1, cut, len(test), 1, "last")
pred_AR = rolling_forecast(ARMA_1_1, cut, len(test), 1, "ARMA", (1, 0, 1))

df = pd.DataFrame({"value": test})
df["pred_mean"] = pred_mean
df["pred_last"] = pred_last
df["pred_AR"] = pred_AR

print(df.head())

preds = df.iloc[:, 1:]
pred_dict = preds.to_dict(orient="list")
draw_predicts(np.arange(len(ARMA_1_1)), ARMA_1_1, len(test), pred_dict)

compare_MSE(df["value"], pred_dict)
