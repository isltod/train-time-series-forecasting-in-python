# 먼저 WSLg 사용을 위한 GUI 툴킷 설치
# sudo apt-get update
# sudo apt-get install python3-tk -y
# 그리고 아래 코드로 Agg 대신 TkAgg 사용 선언해야 matplotlib 그래프를 그림으로 띄운다...
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

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
