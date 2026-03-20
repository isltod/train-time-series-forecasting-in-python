from itertools import product
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import pandas as pd
import sys

sys.path.append("./")
from util import *

# 1. 데이터 훑어보기 - 정상성 테스트 실패 -> 1차 차분 필요
df = pd.read_csv("./data/bandwidth.csv")
# print(df.head())

col_name = "hourly_bandwidth"
title = "Datacenter Hourly Bandwidth"
xlabel = "Month"
ylabel = "Hourly Bandwidth (MBps)"
xticks = (
    np.arange(0, 10000, 730),
    [
        "Jan 2019",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "Jan 2020",
        "Feb",
    ],
)
x = np.arange(len(df))
y = df[col_name]
# draw_line_chart(x, y, title, xlabel, ylabel, xticks)
# test_ADF(y)

# 2. 1차 차분 - 정상성 테스트 성공
diff_y = np.diff(y, n=1)
diff_x = x[1:]
# draw_line_chart(diff_x, diff_y, title, xlabel, ylabel, xticks)
# test_ADF(diff_y)

# 3. ACF, PACF 확인 - ACF는 30 넘어까지 유의미하고 왔다갔다..PACF도 6, 7 정도까지....실패 -> ARMA 적용
# draw_auto_corr(diff_y)
# draw_pacf(diff_y)

# 4. 훈련/테스트 분리 - 7일 168시간
df_diff = pd.DataFrame({col_name: diff_y})
train = df_diff[:-168]
test = df_diff[-168:]
print(len(train), len(test))
# 원본 데이터와 1차 차분 데이터 그래프 그리고 테스트 기간 음영 부분을 함수로 뺐고...
# draw_train_test(x, y, 168, diff_y, title, xlabel, ylabel, xticks)

# 5. 여러 차수들에 대한 AIC 비교 - 3등 (2,2)가 매개변수가 적으면서 점수는 1등과 거의 비슷하니 그걸 선택...
# (3,2)에서 최소 AIC는 맞는데, 숫자가 좀 틀린데...1%정도...왜일까?
ps = range(0, 4)
qs = range(0, 4)
order_list = list(product(ps, qs))
# result_df = optimize_ARMA(diff_y, order_list)
# print(result_df)

# 6. ARMA 예측과 잔차 분석
model = SARIMAX(train, order=(2, 0, 2), simple_differencing=False)
result = model.fit()
print(result.summary())
result.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()
residuals = result.resid
# 이게 자꾸 헷갈리는데...
# 0.05보다 크면 귀무 랜덤 잔차고, 그보다 작으면 유의 비랜덤 상관이고, 그래서 모델 못쓴다는 얘기...
lb = acorr_ljungbox(residuals, 10)
print(lb["lb_pvalue"])
