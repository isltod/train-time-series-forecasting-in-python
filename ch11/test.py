from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm_notebook
from itertools import product
from typing import Union

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("./data/AusAntidiabeticDrug.csv")
print(df.head(10))
print(df.tail(10))
print(df.shape)

train = df.y[:168]
test = df.y[168:]

print(len(test))

from typing import Union
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX


def optimize_SARIMAX(
    endog: Union[pd.Series, list],
    exog: Union[pd.Series, list],
    order_list: list,
    d: int,
    D: int,
    s: int,
) -> pd.DataFrame:

    results = []

    for order in order_list:
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
        results.append([order, model.aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ["(p,q,P,Q)", "AIC"]

    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)

    return result_df


d = 1
D = 1
s = 12

SARIMA_result_df = optimize_SARIMAX(train, None, [(1, 0, 2, 3)], d, D, s)
print(SARIMA_result_df)
