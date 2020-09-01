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
from sklearn.multioutput import MultiOutputRegressor

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

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color
#设置图例并且设置图例的字体及大小
font2 = {
'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 22,
}
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 28,
}
markers = ['h','s','<','>','1','2','3','4','8','p','d','+','^','x','o']*1000
colors = ['r','g','c','y','k','m','b']*1000

scale_x = StandardScaler()
scale_y = StandardScaler()

def loadXY(datafilepath, label_flag = '就业增长率'):
    data = pd.read_table(datafilepath, sep=',')
    x = data.loc[:, '地区生产总值':]
    y = data.loc[:, label_flag]

    mean_cols = x.mean()
    # x=x.fillna(mean_cols)  #填充缺失值
    # x=pd.get_dummies(x)    #独热编码
    # y = np.log(y)  # 平滑处理Y
    y = np.array(y).reshape(-1, 1)
    # 归一化
    # mm_x = MinMaxScaler()
    # x = mm_x.fit_transform(x)
    # 标准化
    x = scale_x.fit_transform(x)
    y = scale_y.fit_transform(y)

    y = y.ravel()  # 转一维
    return x, y


def train(seeds=[1], k=5,  datafilepath='./data/HRB95.txt',test_size=5, label_flag = '就业增长率'):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    # data = np.loadtxt('./data/HRB95.txt', dtype=float, delimiter=',', skiprows=1)
    # x = data[:,1:data.shape[1]]
    # y = data[:,0]
    cv = k
    if cv==1:
        cv = LeaveOneOut()
    models = [
        # KNeighborsRegressor(leaf_size=3, n_neighbors= 2, p=1, weights='distance'),
        # GridSearchCV(SVR(), param_grid={"C": np.logspace(0, 2, 4), "gamma": np.logspace(-2, 2, 7)},n_jobs=-1),
        # RidgeCV(alphas=(0.1, 1.0, 10.0,100.0)),
        MLPRegressor(hidden_layer_sizes=(5),random_state=seed),
        # RandomForestRegressor(random_state=seed),
        # GradientBoostingRegressor(random_state=seed),

        # StackingRegressor(estimators=[
                # ( 'KNN', KNeighborsRegressor(leaf_size=3, n_neighbors= 2, p=1, weights='distance')),
                # ("ridge", RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))),
                # ("gbdt",GradientBoostingRegressor(random_state=seed)),
                # ("RandomForest",RandomForestRegressor(random_state=seed)),
                # ("mlp", MLPRegressor(hidden_layer_sizes=(50,100,50),max_iter=700,random_state=seed)),
        #         ("svr", GridSearchCV(SVR(), n_jobs=-1, param_grid={"C": np.logspace(0, 2, 4), "gamma": np.logspace(-2, 2, 7)})),
        # ],  final_estimator=RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)), n_jobs=-1,cv=cv),
       

    ]
    models_str = [
        # 'KNeighborsRegressor',
        # 'SVR',
        # 'RidgeCV',
        'MLP',
        # 'RF',
        # 'GBDT',
        # 'Stacking',
    ]

    #times次平均得分，
    MAE,MSE,R2={},{},{}
    for time,seed in enumerate(seeds):
        print("-----第%d次(seed=%s)-----"%(time+1,seed))
        print("{:20s}{:10s}{:10s}{:10s}".format("方法","MAE","MSE","R2"))
        x, y = loadXY(datafilepath, label_flag)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, shuffle=True)
        x_train, y_train = x, y
        plt.figure(time, figsize=(10, 10))
        plt.tick_params(labelsize=18)
        # plt.xlim(0, 6)
        # plt.ylim(3, 7, 0.3)
        # plt.plot([x for x in range(1, test_size + 1)],scale_y.inverse_transform(y_test),label='True Label')
        plt.scatter([x for x in range(1, test_size + 1)], scale_y.inverse_transform(y_test),
                    marker='*',label='True Label', s=250)
        for i,name, m in zip(range(100),models_str, models):
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
                test_pred = model.predict(x_test)
                mse_test = np.append(mse_test,mse(y_test, test_pred))
                mae_test = np.append(mae_test,mae(y_test, test_pred))
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
            print(model.coefs_)
            print(len(model.coefs_))
            print(len(model.coefs_[0]))

            plt.matshow(model.coefs_[0], cmap='hot')
            plt.colorbar()
            plt.show()

            '''
            plt.plot([x for x in range(1, test_size + 1)], scale_y.inverse_transform(model.predict(x_test)),
                     marker='o', linestyle=':', label=name,c=colors.pop())
            # plt.scatter([x+i*0.2 for x in range(1, test_size + 1)], scale_y.inverse_transform(model.predict(x_test)),
            #             label=name,c=randomcolor())
            plt.legend(edgecolor='black', loc=1, prop=font2,ncol=2)  # 让图例标签展示
            plt.xlabel(u"Test Data",fontdict=font1)  # X轴标签
            plt.ylabel(label_flag,fontdict=font1)  # Y轴标签
            plt.title('Prediction on GI20',fontdict=font1)  # 标题
        plt.ioff()
        print() #所有模型交叉训练结束（一次） 每一次样本集不一样
        plt.show()
        '''
    print("---------%d次训练测试平均得分----------"%len(seeds))
    print("{:20s}{:10s}{:10s}{:10s}".format("方法","MAE","MSE","R2"))
    for name in MAE.keys():
        print("{:20s}{:6.4f}{:10.4f}{:10.3f}".format(name,np.mean(MAE[name]), np.mean(MSE[name]),np.mean(R2[name])))


def search_best_params(gridcv=None, datafilepath='./data/HRB95.txt'):
    x, y = loadXY(datafilepath , label_flag = '就业弹性系数')
    # gridcv = GridSearchCV(SVR(),cv=10,n_jobs=-1,
    #                     param_grid={"kernel": ("linear", 'rbf'),"C": np.logspace(0, 4, 10),
    #                                 "gamma": np.logspace(-3, 3, 10)})
    gridcv = GridSearchCV(KNeighborsRegressor(), cv=LeaveOneOut(), n_jobs=-1,scoring='neg_mean_squared_error',
                          param_grid={"n_neighbors": [nb for nb in range(1, 20)], "p": [p for p in range(1, 10)],
                                      "weights": ['uniform', 'distance'], "leaf_size": [s for s in range(3, 30)]
                                      })
    # gridcv = GridSearchCV(Ridge(), cv=10, n_jobs=-1,
    #                       param_grid={"alpha": [500, 100,10,1,0.1]
    #                                   })
    gridcv.fit(x, y)
    print(gridcv.best_params_, '\n', gridcv.best_score_)

seeds = [0,1,2,3,4,5,6,7,10,11,12,16,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,48,49,50,51,52,53,55,57,58,59,60,61,62,63,64,66,67,68,69,70,71
,235,236,237,238,239,240,242,243,244,245,246,247,248,249,250,281,282,283,284,286,287,288,289,290,291,292]
# seeds=[234,235,236,237,238,239,240,242,243,244,245,246,247,248,249,250,251,252,254,255,256,258,259,260,261,262,263,264,265,
# 266,267,268,270,271,272,273,275,276,277,278,279,280,281,282,283,284,286,287,288,289,290,291,292,293,294,295,296,297,299]
# seeds=[None]
seeds=[i for i in range(1)]
# seeds = [289]
if __name__ == '__main__':
    train(seeds, k=1,datafilepath='./data/data2016',test_size=20, label_flag = '就业增长率')
#     search_best_params(gridcv=None, datafilepath='./data/HRB95.txt')



