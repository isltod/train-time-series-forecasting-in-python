# 웃기게도 tf를 사용하려면 꼭 먼저 torch를 import 해야만 한다...
import torch
import tensorflow as tf

print("PyTorch 버전:", torch.__version__)
print("GPU 사용 가능 여부:", torch.cuda.is_available())
print("GPU 개수 확인:", torch.cuda.device_count())
print("현재 GPU 기기 번호:", torch.cuda.current_device())
print("GPU 이름 확인:", torch.cuda.get_device_name(0))
print("=" * 100)
print("TensorFlow 버전:", tf.__version__)
print("GPU 목록:", tf.config.list_physical_devices("GPU"))
print("GPU 이름 확인:", tf.test.gpu_device_name())
