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

# 이건 위에 DataWindow 클래스 만들 때 다 있는데 왜 또 만들지? 그냥 wide_window.column_indices 하면 되는데?
column_indices = {name: i for i, name in enumerate(train_df.columns)}
# 베이스라인 클래스에 traffic_volume을 예측 컬럼으로 지정..
baseline_last = Baseline(label_index=column_indices["traffic_volume"])
# 케라스 모델을 컴파일하면 예측을 생성한다고...
# 손실함수로 MSE를, 평가지표로는 MAE를? 뭐가 다르지? 둘 다 차이를 계산해서 역전파 경사도를 만드는 함수 아닌가?
# run_eagerly=True는 심볼릭 텐서라서 값을 보기가 어려워 내가 붙은 코드...
baseline_last.compile(
    loss=MeanSquaredError(), metrics=[MeanAbsoluteError()], run_eagerly=True
)

# val과 test의 MAE
val_performance = {}
performance = {}
# input, label, shift를 1로 주면 이전 단계인 입력을 그대로 예측으로 내놓는 모델이 되는 모양...
# evaluate 메서드는 여기서 만든게 아니라 상속받은 Model 클래스에 있는 건데...
val_performance["Baseline - Last"] = baseline_last.evaluate(single_stp_window.val)
performance["Baseline - Last"] = baseline_last.evaluate(
    single_stp_window.test, verbose=0
)
# 둘 이상 차트를 보려면 wide_window를 써야 한다...
# 근데 모델은 single로 만들고...그 모델을 다른 DataWindow에 넣어서 그리고...
# 그런데도 plot은 DataWindow 클래스에 있어야 한다고...도무지 납득이...
wide_window.plot(baseline_last)
# MAE는 리스트로 들어가나?
print(performance["Baseline - Last"][1])
