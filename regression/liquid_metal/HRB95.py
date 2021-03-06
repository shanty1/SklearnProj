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

font2 = {
'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 22,
}
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 28,
}
markers = ['4','8','p','d','h','s','<','>','1','2','3','^','+','x','^','o']*10000
scale_x = StandardScaler()
scale_y = StandardScaler()

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
    x = scale_x.fit_transform(x)
    y = scale_y.fit_transform(y)

    y = y.ravel()  # 转一维
    return x, y


def train(seeds=[1], k=5,  datafilepath='./data/HRB95.txt', test_size=10):
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    # data = np.loadtxt('./data/HRB95.txt', dtype=float, delimiter=',', skiprows=1)
    # x = data[:,1:data.shape[1]]
    # y = data[:,0]
    cv = k
    if cv==1:
        cv = LeaveOneOut()
    models = [
        GridSearchCV(KNeighborsRegressor(leaf_size=3,  p=2, weights='distance'), n_jobs=-1, cv=5,
                            param_grid={"n_neighbors": [3,5,17]}),
        SVR(C=2.78, gamma=0.1),
        RidgeCV(alphas=(0.1, 1.0, 10.0,100.0)),
        MLPRegressor(hidden_layer_sizes=(50,100,50),max_iter=500, random_state=seed),
        RandomForestRegressor(random_state=seed),
        GradientBoostingRegressor(random_state=seed),

        StackingRegressor(estimators=[
                ("ridge", RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))),
                ("knn",  GridSearchCV(KNeighborsRegressor(leaf_size=3,  p=2, weights='distance'), n_jobs=-1, cv=5,
                            param_grid={"n_neighbors": [3,5,17]})),
                ("gbdt",GradientBoostingRegressor(random_state=seed)),
                ("RandomForest",RandomForestRegressor(random_state=seed)),
                ("mlp", MLPRegressor(hidden_layer_sizes=(50,100,50),max_iter=700,random_state=seed)),
                ("svr", SVR(C=2.78, gamma=0.1)),
        ],  final_estimator=None, n_jobs=-1,cv=cv),

    ]
    models_str = [
        'KNN',
        'SVR',
        'Ridge',
        'MLP',
        'RF',
        'GBDT',
        'Stacking',
    ]

    #times次平均得分，
    MAE,MSE,R2={},{},{}
    for time,seed in enumerate(seeds):
        print("-----第%d次(seed=%s,k=%d)-----"%(time+1,seed,k))
        print("{:20s}{:10s}{:10s}{:10s}".format("方法","MAE","MSE","R2"))
        x, y = loadXY(datafilepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, shuffle=True)
        plt.figure(time,figsize=(10,10))
        plt.tick_params(labelsize=18)
        # plt.xlim(0, 6)
        # plt.ylim(3, 7, 0.3)
        plt.scatter([x for x in range(1, test_size+1)], scale_y.inverse_transform(y_test), marker='*', label='True Label', s=250)

        # plt.plot( scale_y.inverse_transform(y_test), scale_y.inverse_transform(y_test))
        for index,name, m in zip(range(len(models_str)),models_str, models):
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
            else:
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
            marker = 'X' if isinstance(model, StackingRegressor)  else markers.pop()
            plt.plot([x for x in range(1, test_size + 1)], scale_y.inverse_transform(model.predict(x_test)), label=name, marker=marker )
            # plt.scatter(scale_y.inverse_transform(y_test), scale_y.inverse_transform(model.predict(x_test)),marker=marker, label=name)
            plt.legend(edgecolor='black', loc=8, prop=font2, ncol=3)  # 让图例标签展示
            plt.xlabel(u"Test Data",fontdict=font1)  # X轴标签
            plt.ylabel('Density',fontdict=font1)  # Y轴标签
            plt.title('Prediction on NS20',fontdict=font1)  # 标题
        plt.ioff()
        print(end='\n') #所有模型交叉训练结束（一次） 每一次样本集不一样
    #
        plt.show()
    print("---------%d次训练测试平均得分----------"%len(seeds))
    print("{:20s}{:10s}{:10s}{:10s}".format("方法","MAE","MSE","R2"))
    for name in MAE.keys():
        pass
        print("{:20s}{:6.4f}{:10.4f}{:10.3f}".format(name,np.mean(MAE[name]), np.mean(MSE[name]),np.mean(R2[name])))
        

def search_best_params(gridcv=None, datafilepath='./data/HRB95.txt'):
    x, y = loadXY(datafilepath)
    gridcv = GridSearchCV(SVR(),cv=5,n_jobs=-1,
                        param_grid={"kernel": ("linear", 'rbf'),"C": np.logspace(0, 4, 10),
                                    "gamma": np.logspace(-3, 3, 10)})
    # gridcv = GridSearchCV(KNeighborsRegressor(), cv=5, n_jobs=-1,
    #                       param_grid={"n_neighbors": [nb for nb in range(1, 20)], "p": [p for p in range(1, 10)],
    #                                   "weights": ['uniform', 'distance'], "leaf_size": [s for s in range(3, 30)]
    #                                   })
    # gridcv = GridSearchCV(Ridge(), cv=5, n_jobs=-1,
    #                       param_grid={"alpha": [500, 100,10,1,0.1]
    #                                   })
    gridcv.fit(x, y)
    print(gridcv.best_params_, '\n', gridcv.best_score_)

# seeds = [1,2,3,5,23,12,18,34,23,65,47,98,30,60,73,79,83,95,89,55,100]
seeds = [65] #图
# seeds = [None]
if __name__ == '__main__':
    # train(seeds=[1,2,3,4,5,], k=5,datafilepath='./data/HRB95.txt')
    train(seeds=seeds, k=1,datafilepath='./data/HRB95.txt')
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
