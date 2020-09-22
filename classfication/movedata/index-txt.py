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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.base import clone

scale_x = StandardScaler()
scale_y = StandardScaler()

def loadXY(datafilepath, label_flag = '标签名'):
    # data = pd.read_table(datafilepath, sep=',')
    data = pd.read_csv(datafilepath)
    x = data.loc[:, data.columns != label_flag]
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
    # y = scale_y.fit_transform(y)

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
        XGBClassifier(random_state=seed),
        # LogisticRegressionCV()
    ]
    models_str = [
        # 'logist',
        'GBDT',
    ]

    #times次平均得分，
    MAE,MSE,R2={},{},{}
    for time,seed in enumerate(seeds):
        print("-----第%d次(seed=%s)-----"%(time+1,seed))
        print("{:20s}{:10s}{:10s}{:10s}".format("方法","MAE","MSE","R2"))
        x, y = loadXY(datafilepath, label_flag)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, shuffle=True)
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

            print("")
            accuracy = accuracy_score(y_test, test_pred)  # 准确度
            precision = precision_score(y_test, test_pred)  # 精确度
            recall = recall_score(y_test, test_pred)  # 召回率
            macro_f1 = f1_score(y_test, test_pred, average='macro')
            micro_f1 = f1_score(y_test, test_pred, average='micro')
            weighted_f1 = f1_score(y_test, test_pred, average='weighted')
            print('{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}'.format('正例', '准确率', '精确率', '召回率', 'macro-f1', 'micro-f1',
                                                                      'weighted-f1'))
            print('{:10s}{:10.4f}{:10.4f}{:10.4f}{:10.4f}{:10.4f}{:10.4f}'.format('就业', accuracy, precision, recall,
                                                                                  macro_f1, micro_f1, weighted_f1))

            accuracy = accuracy_score(y_test, test_pred)  # 准确度
            precision = precision_score(y_test, test_pred, pos_label=0)  # 精确度
            recall = recall_score(y_test, test_pred, pos_label=0)  # 召回率
            macro_f1 = f1_score(y_test, test_pred, average='macro', pos_label=0)
            micro_f1 = f1_score(y_test, test_pred, average='micro', pos_label=0)
            weighted_f1 = f1_score(y_test, test_pred, average='weighted', pos_label=0)
            print('{:10s}{:10.4f}{:10.4f}{:10.4f}{:10.4f}{:10.4f}{:10.4f}'.format('未就业', accuracy, precision, recall,
                                                                                  macro_f1, micro_f1, weighted_f1))


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
seeds=[i for i in range(100)]
# seeds = [289]
if __name__ == '__main__':
    train(seeds, k=10,datafilepath='./data/cleanData.csv',test_size=0.2, label_flag = 'categorical')
#     search_best_params(gridcv=None, datafilepath='./data/HRB95.txt')



