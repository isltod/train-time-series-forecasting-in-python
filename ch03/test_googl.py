import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 실습 루트에서 데이터와 소스 폴더 따로 만들어 실행시킨다고 가정하고...
df = pd.read_csv("./data/GOOGL.csv")
print(df.head())

# 우상향하는 성질이 잘 보이는 종가 그래프...
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df["Date"], df["Close"])
ax.set_title("Google Stock Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")

plt.xticks(
    [4, 24, 46, 68, 89, 110, 132, 152, 174, 193, 212, 235],
    [
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "Jan 2021",
        "Feb",
        "Mar",
        "Apr",
    ],
)
plt.tight_layout()
plt.show()

# 그대로 ADF 테스트에 넣으면 당연히 비정상성...
ADF_result = adfuller(df["Close"])

print(f"ADF Statistic: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")
print(f"Critical Values:")
for key, value in ADF_result[4].items():
    print(f"   {key}: {value}")

# 자기상관관계도 있겠지...20 즈음해서 음영 안으로 들어간다...덜 랜덤하다는 얘기겠지?
plot_acf(df["Close"], lags=40)
plt.tight_layout()
plt.show()

# 1차 차분을 하면 ADF 테스트 통과하고...
diff_df = np.diff(df["Close"], n=1)

ADF_result = adfuller(diff_df)

print(f"ADF Statistic: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")
print(f"Critical Values:")
for key, value in ADF_result[4].items():
    print(f"   {key}: {value}")

# 자기상관도 없어지긴 하는데...5와 18에서 살짝 음영을 벗어나지만, 일관성/지속성이 없으므로 없는 것으로 본다...
plot_acf(diff_df, lags=20)
plt.tight_layout()
plt.show()
