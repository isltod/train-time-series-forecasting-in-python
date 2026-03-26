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

column_indices = {name: i for i, name in enumerate(train_df.columns)}
print(column_indices)

# 24시간 읽고 1시간 예측하는 모델 - single step model
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
# # h와 c로 과거 정보를 사용하려면 return_sequences=True
# lstm_model = Sequential([LSTM(32, return_sequences=True), Dense(1)])
# history = complile_and_fit(lstm_model, wide_window)
# val_performance = lstm_model.evaluate(wide_window.val)
# test_performance = lstm_model.evaluate(wide_window.test, verbose=2)
# val_dict, test_dict = update_pf_stats("LSTM", val_performance, test_performance)
# wide_window.plot(lstm_model)
# compare_pf_stats(val_dict, test_dict)

# 24시간 읽고 24시간 예측 모델
multi_window = DataWindow(
    input_width=24,
    label_width=24,
    shift=24,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["traffic_volume"],
)
ms_lstm_model = Sequential(
    [
        LSTM(32, return_sequences=True),
        Dense(units=1, kernel_initializer=tf.initializers.zeros),
    ]
)
history = complile_and_fit(ms_lstm_model, multi_window)
ms_val_performance = ms_lstm_model.evaluate(multi_window.val)
ms_test_performance = ms_lstm_model.evaluate(multi_window.test, verbose=0)
val_dict, test_dict = update_pf_stats(
    "MS-LSTM", ms_val_performance, ms_test_performance
)
multi_window.plot(ms_lstm_model)
compare_pf_stats(val_dict, test_dict)
