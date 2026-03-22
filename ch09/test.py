from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
from itertools import product
from typing import Union

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

macro_econ_data = sm.datasets.macrodata.load_pandas().data
target = macro_econ_data["realgdp"]
exog = macro_econ_data[["realcons", "realinv", "realgovt", "realdpi", "cpi"]]


def optimize_SARIMAX(
    endog: Union[pd.Series, list],
    exog: Union[pd.Series, list],
    order_list: list,
    d: int,
    D: int,
    s: int,
) -> pd.DataFrame:

    results = []

    for order in tqdm(order_list):
        try:
            model = SARIMAX(
                endog,
                exog,
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False,
            ).fit(disp=False)
        except:
            continue

        aic = model.aic
        results.append([order, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ["(p,q,P,Q)", "AIC"]

    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)

    return result_df


p = range(0, 4, 1)
d = 1
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 0
Q = range(0, 4, 1)
s = 4

parameters = product(p, q, P, Q)
parameters_list = list(parameters)

target_train = target[:200]
exog_train = exog[:200]

# result_df = optimize_SARIMAX(target_train, exog_train, parameters_list, d, D, s)
# print(result_df)

best_model = SARIMAX(
    target_train,
    exog_train,
    order=(3, 1, 3),
    seasonal_order=(0, 0, 0, 4),
    simple_differencing=False,
)
best_model_fit = best_model.fit(disp=False)

print(best_model_fit.summary())

best_model_fit.plot_diagnostics(figsize=(10, 8))
plt.show()
