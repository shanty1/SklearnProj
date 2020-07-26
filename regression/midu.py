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
from sklearn.metrics import mean_absolute_error as mae
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
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
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


def train(seeds=[1], k=5,  datafilepath='./data/HRB95.txt'):
    seed = None
    random.seed(seed)
    np.random.seed(seed)
    # data = np.loadtxt('./data/HRB95.txt', dtype=float, delimiter=',', skiprows=1)
    # x = data[:,1:data.shape[1]]
    # y = data[:,0]
    cv = k
    if cv==1:
        cv = LeaveOneOut()
    models = [
        GridSearchCV(SVR(), param_grid={"C": np.logspace(0, 2, 4), "gamma": np.logspace(-2, 2, 7)},n_jobs=-1),
        RidgeCV(alphas=(0.1, 1.0, 10.0,100.0)),
        MLPRegressor(hidden_layer_sizes=(50,100,50),max_iter=700, random_state=seed),
        RandomForestRegressor(random_state=seed),
        GradientBoostingRegressor(random_state=seed),

        StackingRegressor(estimators=[
                ("ridge", RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))),
                ("gbdt",GradientBoostingRegressor(random_state=seed)),
                ("RandomForest",RandomForestRegressor(random_state=seed)),
                ("mlp", MLPRegressor(hidden_layer_sizes=(50,100,50),max_iter=700,random_state=seed)),
                ("svr", GridSearchCV(SVR(), n_jobs=-1, param_grid={"C": np.logspace(0, 2, 4), "gamma": np.logspace(-2, 2, 7)})),
        ],  final_estimator=None, n_jobs=-1,cv=cv),
       

    ]
    models_str = [
        'SVR',
        'RidgeCV',
        'MLPRegressor',
        'RandomForest',
        'GradientBoost',
        'Stacking',
    ]
    #times次平均得分，
    MAE,MSE,R2={},{},{}
    for time,seed in enumerate(seeds):
        print("-----第%d次(seed=%s)-----"%(time+1,seed))
        print("{:20s}{:10s}{:10s}{:10s}".format("方法","MAE","MSE","R2"))
        x, y = loadXY(datafilepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=5, random_state=seed, shuffle=True)
        for name, m in zip(models_str, models):
            if not name in MAE.keys() :
                MAE[name] = []
            if not name in MSE.keys() :
                MSE[name] = []
            if not name in R2.keys() :
                R2[name] = []
            print("%18s"%name)
            y_vals, y_val_p_s,mae_test,mse_test,r2_test= [], [],[],[],[]
            model = clone(m)
            # stacking模型，已经内置交叉验证
            if isinstance(model,StackingRegressor):
                model.fit(x_train,y_train)
                train_pred = model.predict(x_train)
                test_pred = model.predict(x_test)
                MAE[name] = np.append(MAE[name], mae(test_pred, y_test))
                MSE[name] = np.append(MSE[name], mse(test_pred, y_test))
                R2[name] = np.append(R2[name], model.score(x_test, y_test))
                print("{:20s}{:6.4f}{:10.4f}{:10.3f}".format("train", mae(train_pred,y_train), mse(train_pred,y_train), model.score(x_train,y_train)))
                print("{:20s}{:6.4f}{:10.4f}{:10.3f}".format("test", MAE[name][-1], MSE[name][-1], R2[name][-1]))
                # if(R2[name][-1]>0.2): #去除异常样本分配
                #     print(seed,end=",")
                continue;
            # 交叉验证
            if k > 1:
                kf = RepeatedKFold(n_splits=k, n_repeats=10, random_state=seed)
            else:
                kf = LeaveOneOut()
            for t, v in kf.split(x_train):
                model.fit(x_train[t], y_train[t])  # fitting
                y_val_p = model.predict(x_train[v])
                y_vals = np.append(y_vals, y_train[v])
                y_val_p_s = np.append(y_val_p_s, y_val_p)
                mse_test = np.append(mse_test,mse(y_test, model.predict(x_test)))
                mae_test = np.append(mae_test,mae(y_test, model.predict(x_test)))
                r2_test = np.append(r2_test,model.score(x_test, y_test))
            matrix={
                'val':{'mae':mae(y_vals, y_val_p_s), 'mse': mse(y_vals, y_val_p_s), 'r2':r2_score(y_vals, y_val_p_s)},
                'test':{'mae':mae_test.mean(),'mse':mse_test.mean(), 'r2':r2_test.mean()},
            }
            print("{:20s}{:6.4f}{:10.4f}{:10.3f}".format("val", matrix['val']['mae'], matrix['val']['mse'], matrix['val']['r2'],))
            print("{:20s}{:6.4f}{:10.4f}{:10.3f}".format("test", matrix['test']['mae'],matrix['test']['mse'],matrix['test']['r2']))
            joblib.dump(model, 'save/%s%d.model' % (name,time))
            MAE[name] = np.append(MAE[name], matrix['test']['mae'])
            MSE[name] = np.append(MSE[name], matrix['test']['mse'])
            R2[name]  = np.append(R2[name],  matrix['test']['r2'])
        print() #所有模型交叉训练结束（一次） 每一次样本集不一样
    #
    print("---------%d次训练测试平均得分----------"%len(seeds))
    print("{:20s}{:10s}{:10s}{:10s}".format("方法","MAE","MSE","R2"))
    for name in MAE.keys():
        print("{:20s}{:6.4f}{:10.4f}{:10.3f}".format(name,np.mean(MAE[name]), np.mean(MSE[name]),np.mean(R2[name])))


def search_best_params(gridcv=None, datafilepath='./data/HRB95.txt'):
    x, y = loadXY(datafilepath)
    # gridcv = GridSearchCV(SVR(),cv=10,n_jobs=-1,
    #                     param_grid={"kernel": ("linear", 'rbf'),"C": np.logspace(0, 4, 10),
    #                                 "gamma": np.logspace(-3, 3, 10)})
    gridcv = GridSearchCV(KNeighborsRegressor(), cv=10, n_jobs=-1,
                          param_grid={"n_neighbors": [nb for nb in range(1, 20)], "p": [p for p in range(1, 10)],
                                      "weights": ['uniform', 'distance'], "leaf_size": [s for s in range(3, 30)]
                                      })
    # gridcv = GridSearchCV(Ridge(), cv=10, n_jobs=-1,
    #                       param_grid={"alpha": [500, 100,10,1,0.1]
    #                                   })
    gridcv.fit(x, y)
    print(gridcv.best_params_, '\n', gridcv.best_score_)

seeds = [0,1,2,3,4,5,6,7,9,10,11,12,14,15,16,17,18,19,20,22,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,43,44,45,46,48,49,50,51,52
,53,55,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,73,74,75,76,77,78,79,80,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,101,103,
107,108,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,137,139,140,141,142,143,144,
146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,166,167,168,169,170,171,172,173,174,176,178,179,180,181,182,183,184,
185,186,188,189,190,191,195,196,197,198,199,200,201,202,203,204,205,207,208,209,210,211,213,214,215,216,217,218,220,221,222,223,224,225,226,
227,228,229,230,232,233,234,235,236,237,238,239,240,242,243,244,245,246,247,248,249,250,251,252,254,255,256,258,259,260,261,262,263,264,265,
266,267,268,270,271,272,273,275,276,277,278,279,280,281,282,283,284,286,287,288,289,290,291,292,293,294,295,296,297,299,]
seeds=[234,235,236,237,238,239,240,242,243,244,245,246,247,248,249,250,251,252,254,255,256,258,259,260,261,262,263,264,265,
266,267,268,270,271,272,273,275,276,277,278,279,280,281,282,283,284,286,287,288,289,290,291,292,293,294,295,296,297,299]
seeds=[i for i in range(10)]
# seeds=[None]

if __name__ == '__main__':
    train(seeds, k=1,datafilepath='./data/midu.txt')
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
'''
---------100次训练测试平均得分----------
方法                  MAE       MSE       R2
SVR                 0.2183    0.1126     0.818
RidgeCV             0.1787    0.0824     0.825
MLPRegressor        0.2146    0.1346     0.741
RandomForest        0.2904    0.2176     0.736
GradientBoost       0.2758    0.1776     0.725
Stacking            0.1835    0.0816     0.852

-----第2次(seed=1)-----
方法                  MAE       MSE       R2        
               SVR
val                 0.2398    0.1213     0.883
test                0.1707    0.0477     0.877
           RidgeCV
val                 0.1794    0.0861     0.917
test                0.1666    0.0467     0.879
      MLPRegressor
val                 0.2400    0.1459     0.860
test                0.1466    0.0422     0.891
      RandomForest
val                 0.3040    0.2206     0.788
test                0.1584    0.0629     0.837
     GradientBoost
val                 0.2761    0.1899     0.817
test                0.2331    0.1180     0.694
          Stacking
train               0.0620    0.0069     0.993
test                0.1204    0.0331     0.914
'''
