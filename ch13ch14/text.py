import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
import sys

sys.path.append("./")
from util import *


train_df = pd.read_csv("./data/train.csv")
val_df = pd.read_csv("./data/val.csv")
test_df = pd.read_csv("./data/test.csv")

print(train_df.shape)
print(val_df.shape)
print(test_df.shape)

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

# 데이터 윈도우라는 걸 설명한다는데...이건 암만봐도 뭔가 이상하다...
# fig, ax = plt.subplots(figsize=(12, 6))
# for n in range(0, 17, 2):
#     start = 24 * n
#     stop = 24 * (n + 1)
#     ax.plot(
#         train_df.traffic_volume[start:stop], marker="s", color="blue", label="input"
#     )
#     ax.plot(
#         train_df.traffic_volume[stop : 2 * stop], marker="x", color="red", label="label"
#     )
# ax.set_xlabel("Time")
# ax.set_ylabel("Traffic Volume")
# plt.xticks(xtics_2weeks[0], xtics_2weeks[1])
# plt.xlim(0, 400)
# fig.autofmt_xdate()
# plt.tight_layout()
# plt.show()

single_stp_window = DataWindow(
    input_width=1,
    label_width=1,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["traffic_volume"],
)
wide_window = DataWindow(
    input_width=24,
    label_width=24,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["traffic_volume"],
)

column_indices = {name: i for i, name in enumerate(train_df.columns)}
baseline_last = Baseline(label_index=column_indices["traffic_volume"])
baseline_last.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance["Baseline - Last"] = baseline_last.evaluate(single_stp_window.val)
performance["Baseline - Last"] = baseline_last.evaluate(
    single_stp_window.test, verbose=0
)
wide_window.plot(baseline_last)
