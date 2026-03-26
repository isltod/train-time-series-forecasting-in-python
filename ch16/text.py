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

col_indices = {name: i for i, name in enumerate(train_df.columns)}
print(col_indices)

# # 24시간 읽고 1시간 예측 CNN
# KERNEL_WIDTH = 3
# LABEL_WIDTH = 24
# INPUT_WIDTH = LABEL_WIDTH + KERNEL_WIDTH - 1

# conv_window = DataWindow(
#     input_width=KERNEL_WIDTH,
#     label_width=1,
#     shift=1,
#     train_df=train_df,
#     val_df=val_df,
#     test_df=test_df,
#     label_columns=["traffic_volume"],
# )
# wide_conv_window = DataWindow(
#     input_width=INPUT_WIDTH,
#     label_width=LABEL_WIDTH,
#     shift=1,
#     train_df=train_df,
#     val_df=val_df,
#     test_df=test_df,
#     label_columns=["traffic_volume"],
# )
# mae_val = [0.083, 0.068, 0.033, 0.03]
# mae_test = [0.081, 0.068, 0.029, 0.026]

# labels = ["Baseline - Last", "Linear", "Dense", "LSTM"]
# val_dict = {labels[i]: mae_val[i] for i in range(len(labels))}
# test_dict = {labels[i]: mae_test[i] for i in range(len(labels))}
# compare_pf_stats(val_dict, test_dict)

# cnn_model = Sequential(
#     [
#         Conv1D(filters=32, kernel_size=(KERNEL_WIDTH,), activation="relu"),
#         Dense(units=32, activation="relu"),
#         Dense(units=1),
#     ]
# )
# history = complile_and_fit(cnn_model, conv_window)
# val_performance = {}
# performance = {}
# val_performance["CNN"] = cnn_model.evaluate(conv_window.val)
# performance["CNN"] = cnn_model.evaluate(conv_window.test, verbose=0)
# wide_conv_window.plot(cnn_model)

# # 24시간 읽고 1시간 예측 CNN + LSTM
# cnn_lstm_model = Sequential(
#     [
#         Conv1D(filters=32, kernel_size=(KERNEL_WIDTH,), activation="relu"),
#         LSTM(32, return_sequences=True),
#         LSTM(units=32, return_sequences=True),
#         Dense(units=1),
#     ]
# )
# history = complile_and_fit(cnn_lstm_model, conv_window)
# val_performance["CNN+LSTM"] = cnn_lstm_model.evaluate(conv_window.val)
# performance["CNN+LSTM"] = cnn_lstm_model.evaluate(conv_window.test, verbose=0)
# wide_conv_window.plot(cnn_lstm_model)

# # 앞 모델들과 mae 비교...
# mae_val.extend(v[1] for v in val_performance.values())
# mae_test.extend(v[1] for v in performance.values())
# labels = ["Baseline - Last", "Linear", "Dense", "LSTM", "CNN", "CNN + LSTM"]
# val_dict = {labels[i]: mae_val[i] for i in range(len(labels))}
# test_dict = {labels[i]: mae_test[i] for i in range(len(labels))}
# compare_pf_stats(val_dict, test_dict)

# 24시간 읽고 24시간 예측
KERNEL_WIDTH = 3
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + KERNEL_WIDTH - 1

multi_window = DataWindow(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=24,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["traffic_volume"],
)

ms_mae_val = [0.352, 0.347, 0.088, 0.078, 0.070]
ms_mae_test = [0.347, 0.341, 0.076, 0.064, 0.058]

# CNN 모델
ms_cnn_model = Sequential(
    [
        Conv1D(filters=32, kernel_size=(KERNEL_WIDTH,), activation="relu"),
        Dense(units=32, activation="relu"),
        Dense(units=1, kernel_initializer=tf.initializers.zeros),
    ]
)
history = complile_and_fit(ms_cnn_model, multi_window)
ms_val_performance = {}
ms_performance = {}
ms_val_performance["CNN"] = ms_cnn_model.evaluate(multi_window.val)
ms_performance["CNN"] = ms_cnn_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(ms_cnn_model)

# CNN + LSTM 모델
ms_cnn_lstm_model = Sequential(
    [
        Conv1D(filters=32, kernel_size=(KERNEL_WIDTH,), activation="relu"),
        LSTM(32, return_sequences=True),
        Dense(1, kernel_initializer=tf.initializers.zeros),
    ]
)
history = complile_and_fit(ms_cnn_lstm_model, multi_window)
ms_val_performance["CNN+LSTM"] = ms_cnn_lstm_model.evaluate(multi_window.val)
ms_performance["CNN+LSTM"] = ms_cnn_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(ms_cnn_lstm_model)

# 앞 모델들과 mae 비교
ms_mae_val.extend(v[1] for v in ms_val_performance.values())
ms_mae_test.extend(v[1] for v in ms_performance.values())
labels = [
    "Baseline - Last",
    "Baseline - Repeat",
    "Linear",
    "Dense",
    "LSTM",
    "CNN",
    "CNN + LSTM",
]
ms_val_dict = {labels[i]: ms_mae_val[i] for i in range(len(labels))}
ms_test_dict = {labels[i]: ms_mae_test[i] for i in range(len(labels))}
compare_pf_stats(ms_val_dict, ms_test_dict)
