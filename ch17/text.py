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

# 24시간 읽고 24시간 예측 ARLSTM
multi_window = DataWindow(
    input_width=24,
    label_width=24,
    shift=24,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["traffic_volume"],
)

ms_mae_val = [0.352, 0.347, 0.088, 0.078, 0.070, 0.078, 0.069]
ms_mae_test = [0.347, 0.341, 0.076, 0.064, 0.058, 0.063, 0.055]

# ARLSTM
AR_LSTM = AutoRegressive(units=32, out_steps=24)
history = compile_and_fit(AR_LSTM, multi_window)
ms_val_performance = {}
ms_performance = {}
ms_val_performance["AR_LSTM"] = AR_LSTM.evaluate(multi_window.val)
ms_performance["AR_LSTM"] = AR_LSTM.evaluate(multi_window.test, verbose=0)
multi_window.plot(AR_LSTM)

ms_mae_val.append(ms_val_performance["AR_LSTM"][1])
ms_mae_test.append(ms_performance["AR_LSTM"][1])
labels = [
    "Baseline - Last",
    "Baseline - Repeat",
    "Linear",
    "Dense",
    "LSTM",
    "CNN",
    "CNN + LSTM",
    "AR - LSTM",
]
val_dict = dict(zip(labels, ms_mae_val))
test_dict = dict(zip(labels, ms_mae_test))
compare_pf_stats(val_dict, test_dict)
