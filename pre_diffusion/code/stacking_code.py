# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 14:18:05 2025

@author: huanghe
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:38:21 2023
@author: huanghe
"""

from datetime import datetime
start_time = datetime.now()
from lightgbm import LGBMRegressor as lgb
from xgboost import XGBRegressor
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold, train_test_split as TTS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

# 载入数据集D
file_path = r'E:\黄河论文1\数据\ML_D.xlsx' 
data = pd.read_excel(file_path, sheet_name='D')

# 缺失值处理
data = data.dropna(axis=0)

## 划分数据集S
x = data.iloc[:,1:47]
X = x.values.astype(np.float64)
y = data.iloc[:,47]
Y = y.values.astype(np.float64)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.2, random_state=None)

# 特征工程：标准化
transfer = StandardScaler()
X_train = transfer.fit_transform(X_train)
X_test = transfer.transform(X_test)

# 1. 定义基础学习器（LightGBM和XGBoost）使用原有参数
lgb_model = lgb(objective='regression',
                num_leaves=580,
                n_estimators=320,
                learning_rate=0.07,
                max_depth=-1,
                force_col_wise=True,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.5,
                reg_lambda=0.5,
                subsample_for_bin=50000,
                importance_type='split',
                n_jobs=-1)

xgb_model = XGBRegressor(learning_rate=0.12, 
                         n_estimators=300, 
                         max_depth=6,
                         min_child_weight=1,
                         subsample=0.8,
                         gamma=0.0,
                         colsample_bytree=0.8,
                         n_jobs=-1, 
                         reg_alpha=0.8, 
                         reg_lambda=1,
                         importance_type='total_gain',
                         seed=1314)

# 2. 定义元学习器（线性回归）
meta_learner = LinearRegression()

# 3. 创建堆叠模型（5折交叉验证）
stacking_model = StackingRegressor(
    estimators=[
        ('lgb', lgb_model),
        ('xgb', xgb_model)
    ],
    final_estimator=meta_learner,
    cv=5,  # 使用5折交叉验证（符合要求不要太多）
    n_jobs=-1
)

# 4. 训练堆叠模型
stacking_model.fit(X_train, y_train)

# 5. 预测
Y_train_pred = stacking_model.predict(X_train)
Y_test_pred = stacking_model.predict(X_test)

# 6. 模型评估
# 训练集评估
print('训练集评估:')
print('MSE_train:', metrics.mean_squared_error(y_train, Y_train_pred))
print('MAE_train:', metrics.mean_absolute_error(y_train, Y_train_pred))
print('RMSE_train:', np.sqrt(metrics.mean_squared_error(y_train, Y_train_pred)))
print('R2_train:', metrics.r2_score(y_train, Y_train_pred))

# 测试集评估
print('\n测试集评估:')
print('MSE_test:', metrics.mean_squared_error(y_test, Y_test_pred))
print('MAE_test:', metrics.mean_absolute_error(y_test, Y_test_pred))
print('RMSE_test:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred)))
print('R2_test:', metrics.r2_score(y_test, Y_test_pred))

# 7. 输出元学习器权重（各基础模型的重要性）
print('\n元学习器权重:')
print('LightGBM权重:', stacking_model.final_estimator_.coef_[0])
print('XGBoost权重:', stacking_model.final_estimator_.coef_[1])

end_time = datetime.now()
print('\n运行时间:', end_time - start_time)