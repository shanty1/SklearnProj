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
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegressionCV
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


def train(modelfile, seeds=[1],  datafilepath='',test_size=5, label_flag = '就业增长率'):

    for time,seed in enumerate(seeds):
        random.seed(seed)
        np.random.seed(seed)
        x, y = loadXY(datafilepath, label_flag)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, shuffle=True)
        model = joblib.load(filename=modelfile)
        test_pred = model.predict(x_test)
        print("就业作为正类，未就业作为负类：")
        accuracy = accuracy_score(y_test,test_pred)  #准确度
        precision = precision_score(y_test, test_pred) #精确度
        recall = recall_score(y_test, test_pred) #召回率
        macro_f1 = f1_score(y_test,test_pred, average='macro')
        micro_f1 = f1_score(y_test,test_pred, average='micro')
        weighted_f1 = f1_score(y_test,test_pred, average='weighted')
        print('{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}'.format('准确率','精确率','召回率','macro-f1','micro-f1','weighted-f1'))
        print('{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}'.format(accuracy,precision,recall,macro_f1,))

        print("\n未就业作为正类，就业作为负类")
        accuracy = accuracy_score(y_test, test_pred)  # 准确度
        precision = precision_score(y_test, test_pred,pos_label=0)  # 精确度
        recall = recall_score(y_test, test_pred,pos_label=0)  # 召回率
        macro_f1 = f1_score(y_test, test_pred, average='macro',pos_label=0)
        micro_f1 = f1_score(y_test, test_pred, average='micro',pos_label=0)
        weighted_f1 = f1_score(y_test, test_pred, average='weighted',pos_label=0)
        print('{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}'.format('准确率', '精确率', '召回率', 'macro-f1', 'micro-f1', 'weighted-f1'))
        print('{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}'.format(accuracy, precision, recall, macro_f1, ))



# seeds=[None]
seeds=[i for i in range(100)]
if __name__ == '__main__':
    modelfile = './save/save1.model'
    train(modelfile,seeds,datafilepath='./data/cleanData.csv',test_size=0.2, label_flag = 'categorical')
#     search_best_params(gridcv=None, datafilepath='./data/HRB95.txt')



