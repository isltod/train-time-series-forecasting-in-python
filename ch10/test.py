from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.varmax import VARMAX
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


def optimize_VAR(endog: Union[pd.Series, list]) -> pd.DataFrame:

    results = []

    for i in tqdm(range(15)):
        try:
            model = VARMAX(endog, order=(i, 0)).fit(dips=False)
        except:
            continue

        aic = model.aic
        results.append([i, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ["p", "AIC"]

    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)

    return result_df


endog = macro_econ_data[["realdpi", "realcons"]]

endog_diff = macro_econ_data[["realdpi", "realcons"]].diff()[1:]

train = endog_diff[:162]
test = endog_diff[162:]

# result_df = optimize_VAR(train)
# print(result_df)

print("realcons Granger-causes realdpi?\n")
print("------------------")
granger_1 = grangercausalitytests(
    macro_econ_data[["realdpi", "realcons"]].diff()[1:], [3]
)
print(granger_1)

print("\nrealdpi Granger-causes realcons?\n")
print("------------------")
granger_2 = grangercausalitytests(
    macro_econ_data[["realcons", "realdpi"]].diff()[1:], [3]
)
print(granger_2)
