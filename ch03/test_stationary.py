# 먼저 WSLg 사용을 위한 GUI 툴킷 설치
# sudo apt-get update
# sudo apt-get install python3-tk -y
# 그리고 아래 코드로 Agg 대신 TkAgg 사용 선언해야 matplotlib 그래프를 그림으로 띄운다...
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


np.random.seed(42)

steps = np.random.standard_normal(size=1000)
# 단순화 위해서 63쪽 수식 3.1에서 C = 0, y0 = 0으로...
steps[0] = 0

# 한 발 걸음을 누적합해서 확률보행 결과 만들기...
random_walk = np.cumsum(steps)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(random_walk)
ax.set_title("Random Walk")
ax.set_xlabel("Time")
ax.set_ylabel("Position")

plt.tight_layout()
plt.show()

# 기본 랜덤워크도 추세라는 게 있어서 그냥하면 ADF 테스트를 통과못한다...
ADF_result = adfuller(random_walk)

# p-value가 0.05보다 크므로 유의하지 않고, 정상적이지 않다...
print(f"ADF Statistic: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")
print(f"Critical Values:")
for key, value in ADF_result[4].items():
    print(f"   {key}: {value}")

# 자기상관관계가 선형적으로 줄고 있어서 추세가 있고, 음영 밖에 있어서 의미가 있다?
# 100개로 늘려보면 대략 60 정도에서 음영 안으로 들어간다? 코인 가격이라면 대략 60개 정도만 보면 된다는 말인가?
plot_acf(random_walk, lags=100)
plt.tight_layout()
plt.show()

# 그래서 차분 랜덤워크를 보면 정상성이 성립된다는 얘기...계절성이 있으면 고차 차분을 해야 한다고...
diff_random_walk = np.diff(random_walk, n=1)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(diff_random_walk)
plt.title("Differenced Random Walk")
plt.xlabel("Time")
plt.ylabel("Position")

plt.tight_layout()
plt.show()

# 겨우 1차 차분을 했는데, p-value가 0이 나와버리네? 정상적이다...
ADF_result = adfuller(diff_random_walk)

print(f"ADF Statistic: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")
print(f"Critical Values:")
for key, value in ADF_result[4].items():
    print(f"   {key}: {value}")

# 첫 번째를 제외하면 성냥개비들이 몽땅 음영 안으로 들어가서 왔다갔다 한다...
# 자기상관관계가 무의미하다...
plot_acf(diff_random_walk, lags=20)
plt.tight_layout()
plt.show()

# 정리하자면,
# 1. 랜덤워크에도 추세라는 것이 있다?
# 2. 그걸 1차 차분하면 추세가 사라지고 백색소음이 된다?
# 3. 내가 본 차트에도 그 유령같은 추세가 보여서 뭔가 할 수 있다고 착각하고 있다?
