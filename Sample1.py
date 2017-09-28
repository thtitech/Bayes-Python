
# coding: utf-8

# In[1]:


# pymc3がない人はコメントアウト
import numpy as np
import pymc3 as pm3
import pymc as pm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import optimize


# In[2]:


# データセットの作成

MU = 10.0
SIGMA = 1.0

y0 = np.random.normal(MU, SIGMA, size=10) 
y1 = np.random.normal(MU, SIGMA, size=100)
y2 = np.random.normal(MU, SIGMA, size=1000)
y3 = np.random.normal(MU, SIGMA, size=10000)
y = [y0, y1, y2, y3]


# In[3]:


# 可視化

fig = plt.figure()
for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.hist(y[i], bins=30)
plt.show()


# In[4]:


# GLMでのモデリング（ただの最尤推定）
for i in [1, 2, 3, 4]:
    size = 10**i
    x = [1 for j in range(size)]
    glm = sm.GLM(np.array(y[i - 1]), np.array(x), family=sm.families.Gaussian())
    res = glm.fit()
    print(res.summary())


# In[6]:


# pymc3のコード

with pm3.Model() as model:
    mu = pm3.Uniform("mu", upper= 10**2, lower= -(10**2))
    tau = pm3.Uniform("tau", upper= 10**2, lower= 0)
    Y_obs = pm3.Normal("Y_obs", mu=mu, sd=tau, observed=np.array(y3)) #ここのyを書き換えてください
    
with model:
    start = pm3.find_MAP(fmin=optimize.fmin_powell)
    step = pm3.Metropolis()
    trace = pm3.sample(5000, start=start, step=step)
    pm3.traceplot(trace[2000:])
    plt.show()
    pm3.summary(trace[2000:])


# In[44]:


# pymcのコード

mu = pm.Uniform("mu", upper= 10**2, lower= -(10**2))
tau = pm.Uniform("tau", upper= 10**2, lower= 0)
obs = pm.Normal("obs", mu=mu, tau=tau, value=y3, observed=True)
model = pm.Model([obs, mu, tau])
map_ = pm.MAP(model)
map_.fit()
mcmc = pm.MCMC(model)
mcmc.sample(5000, 2000)


# In[46]:


plt.close()
pm.Matplot.plot(mcmc.trace("mu"), common_scale=False)
pm.Matplot.plot(mcmc.trace("tau"), common_scale=False)
plt.show()


# In[ ]:




