import sys

sys.path.append("./")
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from util import *


# 1. ARMA(1,1) 과정을 가상으로 만들고
np.random.seed(42)

ar1 = np.array([1, -0.33])
ma1 = np.array([1, 0.9])

ARMA_1_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)

# 2. ADF 테스트, ACF, PACF 그리기
# p value 1.7e-8으로 정상적인데
test_ADF(ARMA_1_1)
# 자기상관이 있고, 3 이후에 무의미하게 되는데...코사인 곡선?처럼 위아래를 왔다갔다...
# 근데 위에서 ma1의 계수를 1차로 줬는데 2차 자기상관이 나오니 뭔가 잘못됐다는 얘기라는데...
draw_auto_corr(ARMA_1_1)
# 편자기상관이 거의 20까지 사라지질 않고 위아래 왔다갔다...
draw_pacf(ARMA_1_1)

# 그래서 여기까지 결론은?
#   이런 모양은 ARMA이고, 어쨌든 p, q를 찾아야 하는데, 이럴 때는 ACF나 PACF가 쓸모없다...
