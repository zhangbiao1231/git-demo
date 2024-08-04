import numpy as np
import pandas as pd
def initialize_params(dims):
    #初始化权重 w
    #初始化偏置 b
    w = np.zeros((dims,1))
    b = 0
    return w,b
##定义符号函数
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
# 利用numpy 对符号函数进行向量化操作
vec_sign = np.vectorize(sign)
def l1(X,y,w,b,alpha):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = np.dot(X,w) + b
    loss = np.sum((y-y_hat)**2)/num_train+np.sum(np.abs(alpha*w))
    dw = np.dot(X.T,(y_hat-y)) / num_train +alpha*vec_sign(w)
    db = np.sum((y_hat-y)) / num_train
    return y_hat,loss,dw,db

def l1_train(X,y,learning_rate=0.01,epochs=1000):
    loss_his = []
    w,b = initialize_params(X.shape[1])
    for i in range(1,epochs):
        y_hat, loss, dw, db = l1(X, y, w, b,0.1)
        w += -learning_rate * dw
        b += -learning_rate * db
        loss_his.append(loss)
        if i % 50 == 0:
            print(f'epoch {i} cost {loss}')
    params = {'w':w,'b':b}
    grads = {'dw':dw,'db':db}
    return loss_his, params, grads
#读取示例数据

data = np.genfromtxt('/Users/zebulonzhang/PycharmProjects/PDSHProject/data/example.dat',delimiter=',')
x = data[:,0:100]
y = data[:,100].reshape(-1,1)

X = np.column_stack((np.ones((x.shape[0],1)),x))
X_train,y_train = X[:70],y[:70]
X_test,y_test = X[70:],y[70:]
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
#执行训练示例数据
loss_his,params,grads = l1_train(X_train,y_train,0.01,300)
# print(params)


from sklearn import linear_model
sk_LASSO = linear_model.Lasso(alpha=0.1)
sk_LASSO.fit(X_train,y_train)
#打印模型相关系数
print(f"sklearn LASSO intercept: {sk_LASSO.intercept_}")
print(f"sklearn LASSO coefficients: \n {sk_LASSO.coef_}")
print(f"\nsklearn LASSO number of iterations: {sk_LASSO.n_iter_}")
## git 账户使用

