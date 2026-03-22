import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

macro_econ_data = sm.datasets.macrodata.load_pandas().data
print(macro_econ_data)

fig, ax = plt.subplots()
ax.plot(macro_econ_data.realgdp)
ax.set_xlabel("Year")
ax.set_ylabel("Real GDP (k$)")
plt.xticks = (np.arange(0, 208, 16), np.arange(1959, 2010, 4))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()[:6]):
    data = macro_econ_data.iloc[:, i + 2]
    ax.plot(data, color="black", linewidth=1)
    ax.set_title(macro_econ_data.columns[i + 2])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    # 그래프의 데이터영역 경계선, top...
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()
