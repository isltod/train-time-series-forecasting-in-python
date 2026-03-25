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

# # 이건 위에 DataWindow 클래스 만들 때 다 있는데 왜 또 만들지? 그냥 wide_window.column_indices 하면 되는데?
column_indices = {name: i for i, name in enumerate(train_df.columns)}

# 이건 마지막 값으로 1 시간 단위를 예측했답시고 내놓은 last 모델...
single_step_window = DataWindow(
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

# # 베이스라인 클래스에 traffic_volume을 예측 컬럼으로 지정..
# baseline_last = Baseline(label_index=column_indices["traffic_volume"])
# # 케라스 모델을 컴파일하면 예측을 생성한다고...
# # 손실함수로 MSE를, 평가지표로는 MAE를? 뭐가 다르지? 둘 다 차이를 계산해서 역전파 경사도를 만드는 함수 아닌가?
# # run_eagerly=True는 심볼릭 텐서라서 값을 보기가 어려워 내가 붙은 코드...
# baseline_last.compile(
#     loss=MeanSquaredError(), metrics=[MeanAbsoluteError()], run_eagerly=True
# )

# val과 test의 MAE
val_performance = {}
performance = {}
# # input, label, shift를 1로 주면 이전 단계인 입력을 그대로 예측으로 내놓는 모델이 되는 모양...
# # evaluate 메서드는 여기서 만든게 아니라 상속받은 Model 클래스에 있는 건데...
# val_performance["Baseline - Last"] = baseline_last.evaluate(single_stp_window.val)
# performance["Baseline - Last"] = baseline_last.evaluate(
#     single_stp_window.test, verbose=0
# )
# # 둘 이상 차트를 보려면 wide_window를 써야 한다...
# # 근데 모델은 single로 만들고...그 모델을 다른 DataWindow에 넣어서 그리고...
# # 그런데도 plot은 DataWindow 클래스에 있어야 한다고...도무지 납득이...
# wide_window.plot(baseline_last)
# # MAE는 리스트로 들어가나?
# print(performance["Baseline - Last"][1])

# # 이건 마지막 값으로 24 시간의 예측값이라고 내놓는 last 24 모델
multi_window = DataWindow(
    input_width=24,
    label_width=24,
    shift=24,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["traffic_volume"],
)

ms_baseline_last = MultiStepLastBaseline(label_index=column_indices["traffic_volume"])
ms_baseline_last.compile(
    loss=MeanSquaredError(), metrics=[MeanAbsoluteError()], run_eagerly=True
)
ms_val_performance = {}
ms_performance = {}

ms_val_performance["Multi-Step Last Baseline"] = ms_baseline_last.evaluate(
    multi_window.val
)
ms_performance["Multi-Step Last Baseline"] = ms_baseline_last.evaluate(
    multi_window.test, verbose=0
)
# multi_window.plot(ms_baseline_last)
# print(ms_performance["Multi-Step Last Baseline"][1])

# # 마지막 24시간 값들을 다음 24시간 예측이랍시고 내놓는 모델
ms_baseline_repeat = RepeatBaseline(label_index=column_indices["traffic_volume"])
ms_baseline_repeat.compile(
    loss=MeanSquaredError(), metrics=[MeanAbsoluteError()], run_eagerly=True
)
ms_val_performance["Repeat Baseline"] = ms_baseline_repeat.evaluate(multi_window.val)
ms_performance["Repeat Baseline"] = ms_baseline_repeat.evaluate(
    multi_window.test, verbose=0
)
# multi_window.plot(ms_baseline_repeat)
# print(ms_performance["Repeat Baseline"][1])

# # 이건 위에 마지막 값으로 다음 예측이랍시고 내놓는 모델을 2가지 예측하도록 바꾼 모델...
mo_single_step_window = DataWindow(
    input_width=1,
    label_width=1,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["temp", "traffic_volume"],
)
mo_wide_window = DataWindow(
    input_width=24,
    label_width=24,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["temp", "traffic_volume"],
)
# mo_baseline_last = Baseline(label_index=[0, 2])
# mo_baseline_last.compile(
#     loss=MeanSquaredError(), metrics=[MeanAbsoluteError()], run_eagerly=True
# )

mo_val_performance = {}
mo_performance = {}
# mo_val_performance["Baseline - Last"] = mo_baseline_last.evaluate(
#     mo_single_step_window.val
# )
# mo_performance["Baseline - Last"] = mo_baseline_last.evaluate(
#     mo_single_step_window.test, verbose=0
# )
# mo_wide_window.plot(mo_baseline_last)
# mo_wide_window.plot(model=mo_baseline_last, plot_col="temp")
# print(mo_performance["Baseline - Last"][1])

# 여기부터는 학습하는 모형이긴 한데...우선 선형(히든 없는) 모델들...
# 이건 1시간 읽고 1시간 예측하는 모델이라고...
# 다른 종류의 계층을 순서대로 쌓는 Sequential, 선형 모형이라 Dense, 1변수 예측이므로 1
# liner = Sequential([Dense(1)])
# history = complile_and_fit(liner, single_step_window)
# val_performance["Linear"] = liner.evaluate(single_step_window.val)
# performance["Linear"] = liner.evaluate(single_step_window.test, verbose=0)
# wide_window.plot(liner)

# 이건 24시간 읽고 24시간 예측
# tf.initializers.zeros는 훈련 속도를 빠르게 한다고...
ms_linear = Sequential([Dense(1, kernel_initializer=tf.initializers.zeros)])
history = complile_and_fit(ms_linear, multi_window)
ms_val_performance["Linear"] = ms_linear.evaluate(multi_window.val)
ms_performance["Linear"] = ms_linear.evaluate(multi_window.test, verbose=0)
# multi_window.plot(ms_linear)

# 이건 1시간 읽고 1시간 예측하는데, 온도와 교통량 2가지 예측이라고...
# 결국 Dense 숫자는 예측할 변수 숫자라라는 건데...
# mo_linear = Sequential([Dense(2)])
# history = complile_and_fit(mo_linear, mo_single_step_window)
# mo_val_performance["Linear"] = mo_linear.evaluate(mo_single_step_window.val)
# mo_performance["Linear"] = mo_linear.evaluate(mo_single_step_window.test, verbose=0)
# mo_wide_window.plot(mo_linear)

# 이제부터는 히든 있는 딥러닝 계열로...
# # 이건 한 단계 읽고 한 단계 예측하는 모델...
# dense = Sequential(
#     [
#         Dense(units=64, activation="relu"),
#         Dense(units=64, activation="relu"),
#         Dense(units=1),
#     ]
# )
# history = complile_and_fit(dense, single_step_window)
# val_performance["Dense"] = dense.evaluate(single_step_window.val)
# performance["Dense"] = dense.evaluate(single_step_window.test, verbose=0)
# wide_window.plot(dense)

# 이건 24시간 읽고 24시간 예측하는 모델...
ms_dense = Sequential(
    [
        Dense(units=64, activation="relu"),
        Dense(units=64, activation="relu"),
        Dense(units=1, kernel_initializer=tf.initializers.zeros),
    ]
)
history = complile_and_fit(ms_dense, multi_window)
ms_val_performance["Dense"] = ms_dense.evaluate(multi_window.val)
ms_performance["Dense"] = ms_dense.evaluate(multi_window.test, verbose=0)
multi_window.plot(ms_dense)

ms_mae_val = [v[1] for v in ms_val_performance.values()]
ms_mae_test = [v[1] for v in ms_performance.values()]

x = np.arange(len(ms_performance))

fig, ax = plt.subplots()
ax.bar(
    x - 0.15,
    ms_mae_val,
    width=0.25,
    color="black",
    edgecolor="black",
    label="Validation",
)
ax.bar(
    x + 0.15,
    ms_mae_test,
    width=0.25,
    color="white",
    edgecolor="black",
    hatch="/",
    label="Test",
)
ax.set_ylabel("MAE")
ax.set_xlabel("Models")
for index, value in enumerate(ms_mae_val):
    plt.text(x=index - 0.15, y=value + 0.0025, s=str(round(value, 3)), ha="center")
for index, value in enumerate(ms_mae_test):
    plt.text(x=index + 0.15, y=value + 0.0025, s=str(round(value, 3)), ha="center")
plt.ylim(0, 0.4)
ax.set_xticks(ticks=x, labels=ms_performance.keys())
# ax.set_xticklabels(list(ms_performance.keys()), rotation=45)
ax.legend(loc="best")
plt.tight_layout()
plt.show()
