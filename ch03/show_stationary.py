# wsl2에서 차트 그림으로 띄우기 설정
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np


# 1. random walk process 만들기
def simulate_process(is_stationary: bool) -> np.array:
    """
    71쪽 수식 3.9와 3.10 구현
    stationary 조건에 따라 정상성 있거나 없는 random walk 만들기
    """
    np.random.seed(42)
    process = np.empty(400)

    if is_stationary:
        alpha = 0.5
        process[0] = 0
    else:
        alpha = 1
        process[0] = 10

    for t in range(1, 400):
        process[t] = alpha * process[t - 1] + np.random.standard_normal()

    return process


stationary_process = simulate_process(is_stationary=True)
non_stationary_process = simulate_process(is_stationary=False)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(stationary_process, label="Stationary Process", linestyle="-")
ax.plot(non_stationary_process, label="Non-Stationary Process", linestyle="--")
ax.set_title("Random Walk")
ax.set_xlabel("Time")
ax.set_ylabel("Position")
ax.legend(loc=2)

plt.tight_layout()
plt.show()


# 2. 평균 변화 확인
def mean_over_time(process: np.array) -> np.array:
    """
    시간에 따른 평균 변화 계산
    """
    mean_values = np.zeros(len(process))

    for t in range(len(process)):
        mean_values[t] = np.mean(process[: t + 1])

    return mean_values


stationary_mean = mean_over_time(stationary_process)
non_stationary_mean = mean_over_time(non_stationary_process)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(stationary_mean, label="Stationary Process", linestyle="-")
ax.plot(non_stationary_mean, label="Non-Stationary Process", linestyle="--")
ax.set_title("Mean Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Mean")
ax.legend(loc=1)

plt.tight_layout()
plt.show()


# 3. 분산 변화 확인
def variance_over_time(process: np.array) -> np.array:
    """
    시간에 따른 분산 변화 계산
    """
    variance_values = np.zeros(len(process))

    for t in range(len(process)):
        variance_values[t] = np.var(process[: t + 1])

    return variance_values


stationary_variance = variance_over_time(stationary_process)
non_stationary_variance = variance_over_time(non_stationary_process)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(stationary_variance, label="Stationary Process", linestyle="-")
ax.plot(non_stationary_variance, label="Non-Stationary Process", linestyle="--")
ax.set_title("Variance Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Variance")
ax.legend(loc=2)

plt.tight_layout()
plt.show()
