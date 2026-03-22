import sys

sys.path.append("./")
from util import *

from itertools import product


# 1. 데이터 훑어보기
df = pd.read_csv("./data/air-passengers.csv")
print(df.head())
col_name = "Passengers"
x = np.arange(len(df))
y = df[col_name].to_numpy()
title = "Air Passengers"
xlabel = "Month"
ylabel = "Passengers"
xticks = (np.arange(0, 145, 12), np.arange(1949, 1962))
# draw_line_chart(x, y, title, xlabel, ylabel, xticks)

# 2. 계절성 차트표시
marks = np.arange(6, 145, 12)
vlines = np.arange(0, 145, 12)
# draw_seasonality(x, y, title, xlabel, ylabel, xticks, marks, vlines)

# 3. 추세와 계절성 분리
period = 12
draw_seasonal_decompose(x, y, period, xticks)
# 이건 왜 그리지?
lienar_ts = np.arange(0, 144)
draw_seasonal_decompose(x, lienar_ts, period, xticks)

cut = 288
train = df[col_name][:-cut]
test = pd.DataFrame(df[col_name][-cut:])

# 1. 일단 AIC 비교는 해야하니까, order list 만들기
ps = range(0, 4)
qs = range(0, 4)
order_list = list(product(ps, qs))
print(order_list)
