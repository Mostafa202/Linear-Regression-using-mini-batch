import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,-1].values
n=x
from sklearn.preprocessing import *

s=StandardScaler()
x=s.fit_transform(x)

k=0
def mini_batch(theta,x,y):
    
    n_epochs=50
    mini_batch_size=8
    m=len(x)
    lr=0.1
    for epoch in range(n_epochs):
        x_perm=np.random.permutation(m)
        x_s=x[x_perm]
        y_s=y[x_perm]
        for i in range(0,m,mini_batch_size):
            xi=x_s[i:i+mini_batch_size]
            yi=y_s[i:i+mini_batch_size]
            k=yi
            gradients=2/mini_batch_size*xi.T.dot(xi.dot(theta)-yi.reshape(-1,1))
            theta=theta-lr*gradients
            
    return theta
    
    
    

x=np.c_[np.ones((x.shape[0],1)),x]
theta=np.random.randn(x.shape[1],1)
th=mini_batch(theta,x,y)

y_pred=x.dot(th)

from sklearn.metrics import *

err=mean_absolute_error(y,y_pred)

print(err)

plt.scatter(n,y,c='r')
plt.plot(n,y_pred,c='b')













