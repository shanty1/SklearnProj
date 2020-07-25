import pandas as pd
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import warnings

# filter warnings
warnings.filterwarnings('ignore')
# 正常显示中文
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示符号
from matplotlib import rcParams

rcParams['axes.unicode_minus'] = False

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.base import clone


def loadXY(datafilepath):
    label_flag = 'OUT'
    data = pd.read_table(datafilepath, sep=',')
    x = data.loc[:, data.columns != label_flag]
    y = data.loc[:, label_flag]

    mean_cols = x.mean()
    # x=x.fillna(mean_cols)  #填充缺失值
    # x=pd.get_dummies(x)    #独热编码
    # y = np.log(y)  # 平滑处理Y
    y = np.array(y).reshape(-1, 1)
    # 归一化
    mm_x = MinMaxScaler()
    x = mm_x.fit_transform(x)
    # 标准化
    scale_x = StandardScaler()
    x = scale_x.fit_transform(x)
    scale_y = StandardScaler()
    y = scale_y.fit_transform(y)

    y = y.ravel()  # 转一维
    return x, y


def train(times, k, seed=1, datafilepath='./data/HRB95.txt'):
    random.seed(seed)
    np.random.seed(seed)
    # data = np.loadtxt('./data/HRB95.txt', dtype=float, delimiter=',', skiprows=1)
    # x = data[:,1:data.shape[1]]
    # y = data[:,0]

    models = [
        # LinearRegression(normalize=True),
        # KNeighborsRegressor(),
        # GridSearchCV(SVR(), param_grid={"C": np.logspace(0, 2, 4), "gamma": np.logspace(-2, 2, 7)}),
        # Ridge(alpha=0.0001,max_iter=5000, random_state=seed),
        # MLPRegressor(hidden_layer_sizes=(100,200,50,20),max_iter=1000, random_state=seed),
        RandomForestRegressor(random_state=seed),
        # GradientBoostingRegressor(random_state=seed),
        # BaggingRegressor(random_state=seed),
        # VotingRegressor(estimators=[
        #         ("knn",  KNeighborsRegressor()),
        #         ("gbdt",GradientBoostingRegressor(random_state=seed)),
        #         ("RandomForest",RandomForestRegressor(random_state=seed)),
        #         ("mlp", MLPRegressor(hidden_layer_sizes=(100,200,50,20),max_iter=1000,random_state=seed)),
        #         ("ridge", Ridge(random_state=seed)),
        #     ],n_jobs=-1),
        # StackingRegressor(estimators=[
        #         ("knn",  KNeighborsRegressor()),
        #         ("gbdt",GradientBoostingRegressor(random_state=seed)),
        #         ("RandomForest",RandomForestRegressor(random_state=seed)),
        #         ("mlp", MLPRegressor(hidden_layer_sizes=(100,200,50,20),max_iter=1000,random_state=seed)),
        #         ("ridge", Ridge(alpha=0.01,max_iter=5000, random_state=seed)),
        #     ],  final_estimator=None, cv=2, n_jobs=-1),
        StackingRegressor(estimators=[
                ("knn",  KNeighborsRegressor()),
                ("gbdt",GradientBoostingRegressor(random_state=seed)),
                ("RandomForest",RandomForestRegressor(random_state=seed)),
                ("mlp", MLPRegressor(hidden_layer_sizes=(100,200,50,20),max_iter=1000,random_state=seed)),
                ("ridge", Ridge(alpha=0.01,max_iter=5000, random_state=seed)),
            ],  final_estimator=None, n_jobs=-1, cv=LeaveOneOut()),
    ]
    models_str = [
        # 'LinearRegression'
        # 'KNNRegressor',
        # 'SVR',
        # 'Ridge',
        # 'MLPRegressor',
        'RandomForest',
        # 'GradientBoost',
        # 'Bagging',
        # 'VotingRegressor',
        # 'StackingRegressor10',
        'StackingRegressor5'
    ]

    for time in range(times):
        x, y = loadXY(datafilepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=5, random_state=seed, shuffle=True)
        for name, m in zip(models_str, models):
            y_vals, y_val_p_s = [], []
            model = clone(m)
            # 交叉验证
            if k > 1:
                kf = RepeatedKFold(n_splits=k, n_repeats=20, random_state=seed)
            else:
                kf = LeaveOneOut()
            for t, v in kf.split(x_train):
                model.fit(x_train[t], y_train[t])  # fitting
                y_val_p = model.predict(x_train[v])
                y_vals = np.append(y_vals, y_train[v])
                y_val_p_s = np.append(y_val_p_s, y_val_p)
            y_test_p = model.predict(x_test)
            print(name)
            print("\t验证集MSE：{:.3f} R2：{:.3f}".format(mse(y_vals, y_val_p_s), r2_score(y_vals, y_val_p_s)))
            print("\t测试集MSE：{:.3f} R2：{:.3f}".format(mse(y_test, y_test_p), model.score(x_test, y_test)))
            joblib.dump(model, 'save/%s.model' % name)


def search_best_params(gridcv=None, datafilepath='./data/HRB95.txt'):
    x, y = loadXY(datafilepath)
    # gridcv = GridSearchCV(SVR(),cv=10,n_jobs=-1,
    #                     param_grid={"kernel": ("linear", 'rbf'),"C": np.logspace(0, 4, 10),
    #                                 "gamma": np.logspace(-3, 3, 10)})
    gridcv = GridSearchCV(KNeighborsRegressor(), cv=10, n_jobs=-1,
                          param_grid={"n_neighbors": [nb for nb in range(1, 20)], "p": [p for p in range(1, 10)],
                                      "weights": ['uniform', 'distance'], "leaf_size": [s for s in range(3, 30)]
                                      })
    gridcv.fit(x, y)
    print(gridcv.best_params_, '\n', gridcv.best_score_)


if __name__ == '__main__':
    time = 1 #不是训练次数，而是重新划分数据集重新训练测试次数
    train(times=1, k=1, seed=1, datafilepath='./data/HRB95.txt')
    train(times=1, k=5, seed=1, datafilepath='./data/HRB95.txt')
    train(times=1, k=10,seed=1, datafilepath='./data/HRB95.txt')
    train(times=1, k=1, seed=2, datafilepath='./data/HRB95.txt')
    train(times=1, k=5, seed=2, datafilepath='./data/HRB95.txt')
    train(times=1, k=1, seed=3, datafilepath='./data/HRB95.txt')
    train(times=1, k=5, seed=3, datafilepath='./data/HRB95.txt')
    # search_best_params()

'''
scores=[]
plt.cla()
plt.close()
# plt.figure()
print('{:20s} {:10s} {:10s}'.format("算法", "Score", ""))
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state=1)
for name,model in zip(models_str,models):

    scores.append(cross_val_score(model, x, y, scoring='r2', cv=10))

    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    # y_pred_original = scale_y.inverse_transform(y_pred)
    # y_test_original = scale_y.inverse_transform(y_test)
    loss = mse(y_pred,y_test)
    score = model.score(x_test, y_test)
    # scorea.append(str(score)[:5])

    plt.plot(np.arange(len(y_test)), y_test, "ro-", label="Predict value")
    plt.plot(np.arange(len(y_test)), y_pred, "go-", label="Predict value")
    plt.title(f"{name}---score:{score}")
    plt.legend()
    plt.show()


    print('{:20s} {:10s} {:10s}'.
          format(name, str(score)[:5], str(r2_score(y_pred,y_test))[:5]))

# print(scores)
'''
