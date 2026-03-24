import sys

sys.path.append("./")
from util import *
from sklearn.preprocessing import MinMaxScaler
import datetime
import warnings

warnings.filterwarnings("ignore")

# 데이터 읽기
df = pd.read_csv("./data/metro_interstate_traffic_volume_preprocessed.csv")
print(df.head())
print(df.tail())
print(df.shape)
xtics_2weeks = (
    np.arange(7, 400, 24),
    [
        "Friday",
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)

# 교통량 2주
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(df.date_time, df.traffic_volume)
# ax.set_title("Metro Interstate Traffic Volume")
# ax.set_xlabel("Date Time")
# ax.set_ylabel("Traffic Volume")
# plt.xticks(
#     xtics_2weeks[0],
#     xtics_2weeks[1],
# )
# plt.xlim(0, 400)
# fig.autofmt_xdate()
# plt.tight_layout()
# plt.show()

# 온도
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(df.date_time, df.temp)
# ax.set_xlabel("Date Time")
# ax.set_ylabel("Temperature")
# plt.xticks([2239, 10999], [2017, 2018])
# plt.tight_layout()
# plt.show()

# 온도 2주
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(df.date_time, df.temp)
# ax.set_xlabel("Date Time")
# ax.set_ylabel("Temperature")
# plt.xticks(
#     xtics_2weeks[0],
#     xtics_2weeks[1],
# )
# plt.xlim(0, 400)
# fig.autofmt_xdate()
# plt.tight_layout()
# plt.show()

#
ddt = df.describe().transpose()
print(ddt)
# 변화가 없는 rain, snow 제거
cols_to_drop = ["rain_1h", "snow_1h"]
df = df.drop(cols_to_drop, axis=1)
print(df.head())

timestamp_s = pd.to_datetime(df.date_time).map(datetime.datetime.timestamp)

# 일단 이게 뭔 문제인지는 모르겠는데, 당연하게도 timestamp로 바꾼 시간은 그냥 선형적으로 증가만 한다...
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(timestamp_s)
# ax.set_xlabel("Date Time")
# ax.set_ylabel("Timestamp")
# plt.tight_layout()
# plt.show()

# 시간을, 그것도 오전 오후 같은 숫자를 가지는 시간을 timestamp에서 만들어낸다...
day = 24 * 60 * 60
df["day_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
df["day_cos"] = np.cos(timestamp_s * (2 * np.pi / day))
df = df.drop(["date_time"], axis=1)
# print(df.head())
# df.sample(50).plot.scatter("day_sin", "day_cos").set_aspect("equal")
# plt.tight_layout()
# plt.show()

# train/validation/test 분할
n = len(df)
print(n)
i7, i9 = int(n * 0.7), int(n * 0.9)
# .loc을 쓰면 뒤 인덱스가 배제되지 않고 포함된다...이거 헛갈리네...
train = df[:i7]
val = df[i7:i9]
test = df[i9:]
print(train.shape, val.shape, test.shape)

tdt = train.describe().transpose()
print(tdt)

scaler = MinMaxScaler()
scaler.fit(train)
train = pd.DataFrame(scaler.transform(train), columns=train.columns)
print(train.head())
val[val.columns] = scaler.transform(val[val.columns])
print(val.head())
test = pd.DataFrame(scaler.transform(test), columns=test.columns)
print(test.head())

train.to_csv("./data/train.csv", index=False)
val.to_csv("./data/val.csv", index=False)
test.to_csv("./data/test.csv", index=False)
