# 웃기게도 tf를 사용하려면 꼭 먼저 torch를 import 해야만 한다...
import torch
import tensorflow as tf
import time
import datetime

# 장치 확인
print("PyTorch 버전:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")
print("GPU 개수 확인:", torch.cuda.device_count())
print("GPU 이름 확인:", torch.cuda.get_device_name(0))
print(f"GPU 메모리 할당: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"GPU 메모리 캐시: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
print("현재 GPU 기기 번호:", torch.cuda.current_device())
print("=" * 100)
print("TensorFlow 버전:", tf.__version__)
print("GPU 목록:", tf.config.list_physical_devices("GPU"))
print("GPU 이름 확인:", tf.test.gpu_device_name())


# 현재 시간 포맷팅 함수
def get_formatted_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# 큰 텐서 생성 및 연산 시간 측정
size = 25000
print(f"\n{size}x{size} 크기의 행렬 곱셈 테스트:")

# CPU에서 연산
print(f"\n[CPU 연산 시작: {get_formatted_time()}]")
start_time = time.time()
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)
c_cpu = torch.matmul(a_cpu, b_cpu)
cpu_time = time.time() - start_time
print(f"[CPU 연산 종료: {get_formatted_time()}]")
print(f"CPU 연산 시간: {cpu_time:.4f}초")

# GPU에서 연산
if device.type == "cuda":
    print(f"\n[GPU 연산 시작: {get_formatted_time()}]")
    start_time = time.time()
    a_gpu = torch.randn(size, size, device=device)
    b_gpu = torch.randn(size, size, device=device)
    c_gpu = torch.matmul(a_gpu, b_gpu)
    # GPU 연산이 완료될 때까지 대기
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
print(f"[GPU 연산 종료: {get_formatted_time()}]")
print(f"GPU 연산 시간: {gpu_time:.4f}초")

if cpu_time > 0:
    print(f"속도 향상: {cpu_time / gpu_time:.2f}배 빠름")

# GPU 메모리 사용량 확인
print(f"연산 후 GPU 메모리 할당: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"연산 후 GPU 메모리 캐시: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# 전체 테스트 완료 시간
print(f"\n[테스트 완료: {get_formatted_time()}]")
