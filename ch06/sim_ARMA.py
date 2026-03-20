import sys

sys.path.append("./")
import numpy as np
from itertools import product
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import ArmaProcess

# 밑에 SARIMAX는 아래 util에서 import 되어서 그냥 쓴다..
from util import *


# 1. ARMA(1,1) 과정을 가상으로 만들고
np.random.seed(42)

ar1 = np.array([1, -0.33])
ma1 = np.array([1, 0.9])

ARMA_1_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)

# 2. ADF 테스트, ACF, PACF 그리기
# p value 1.7e-8으로 정상적인데
# test_ADF(ARMA_1_1)
# 자기상관이 있고, 3 이후에 무의미하게 되는데...코사인 곡선?처럼 위아래를 왔다갔다...
# 근데 위에서 ma1의 계수를 1차로 줬는데 2차 자기상관이 나오니 뭔가 잘못됐다는 얘기라는데...
# draw_auto_corr(ARMA_1_1)
# 편자기상관이 거의 20까지 사라지질 않고 위아래 왔다갔다...
# draw_pacf(ARMA_1_1)

# 그래서 여기까지 결론은?
#   이런 모양은 ARMA이고, 어쨌든 p, q를 찾아야 하는데, 이럴 때는 ACF나 PACF가 쓸모없다...

# p, q 0~3 조합으로 AIC 비교
ps = range(0, 4)
qs = range(0, 4)

order_list = list(product(ps, qs))
print(order_list)

# 애초에 차수는 1, 1로 만들었으니 (1, 1)이 가장 AIC 낮게 나온다...
# result_df = optimize_ARMA(ARMA_1_1, order_list)
# print(result_df)

# 잔차 분석
# 감마 분포 등은 다른 곳에 휜다...
gamma = np.random.default_rng().standard_gamma(shape=2, size=1000)
qqplot(gamma, line="45")
# 일단 잔차는 정규분보를 가정하고, qqplot에서 직선이 나와야 한다...
normal = np.random.normal(size=1000)
qqplot(normal, line="45")
plt.tight_layout()
# plt.show()

# 그럼 (1, 1)로 모델링 한 결과에서 잔차를 받아서 qqplot해보면...
model = SARIMAX(ARMA_1_1, order=(1, 0, 1), simple_differencing=False)
result = model.fit()
residuals = result.resid

qqplot(residuals, line="45")
plt.tight_layout()
# plt.show()

result.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
# plt.show()

# 융박스 테스트 - 이건 귀무가 랜덤이므로 0.05보다 작으면 모델을 못쓴단 얘기가 된다...
# 책과는 다르게, 결과는 df로 나오고, range 안하고 그냥 lag=10 주면 됨...
lb = acorr_ljungbox(residuals, 10)
print(lb["lb_pvalue"])
