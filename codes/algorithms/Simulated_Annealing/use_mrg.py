
from mrg32k3a.mrg32k3a import MRG32k3a
rng = MRG32k3a(s_ss_sss_index=[1, 2, 3])

x = rng.normalvariate(mu=2, sigma=5)
print(x)