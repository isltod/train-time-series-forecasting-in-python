import sys

sys.path.append("./")
from util import *
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use("TkAgg")

np.random.seed(42)

# 130쪽 수식 5.4
ma2 = np.array([1, 0, 0])
ar2 = np.array([1, -0.33, -0.50])

AR2_process = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)

# 그려보자 - 유동인구 데이터의 1차 차분 결과와 유사한 그래프...
draw_line_chart(
    np.arange(len(AR2_process)), AR2_process, "AR(2) Process", "Time", "Value"
)
plot_pacf(AR2_process, lags=20)
plt.tight_layout()
plt.show()
