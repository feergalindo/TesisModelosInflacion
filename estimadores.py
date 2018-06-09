import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
from scipy import stats
import random

class Selector(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                 INPC_transf_rezago_1 =True,
                 INPC_transf_rezago_2 =True,
                 INPC_transf_rezago_3 =True,
                 INPC_transf_rezago_4 =True,
                 INPC_transf_rezago_10=True,
                 INPC_transf_rezago_11=True,
                 INPC_transf_rezago_12=True,
                 INPC_transf_rezago_13=True,
                 INPC_transf_rezago_14=True,
                 INPC_transf_rezago_22=True,
                 INPC_transf_rezago_23=True,
                 INPC_transf_rezago_24=True,
                 INPC_transf_rezago_25=True,
                 INPC_transf_rezago_26=True):
        self.INPC_transf_rezago_1  = INPC_transf_rezago_1    
        self.INPC_transf_rezago_2  = INPC_transf_rezago_2 
        self.INPC_transf_rezago_3  = INPC_transf_rezago_3 
        self.INPC_transf_rezago_4  = INPC_transf_rezago_4 
        self.INPC_transf_rezago_10 = INPC_transf_rezago_10
        self.INPC_transf_rezago_11 = INPC_transf_rezago_11
        self.INPC_transf_rezago_12 = INPC_transf_rezago_12
        self.INPC_transf_rezago_13 = INPC_transf_rezago_13
        self.INPC_transf_rezago_14 = INPC_transf_rezago_14
        self.INPC_transf_rezago_22 = INPC_transf_rezago_22
        self.INPC_transf_rezago_23 = INPC_transf_rezago_23
        self.INPC_transf_rezago_24 = INPC_transf_rezago_24
        self.INPC_transf_rezago_25 = INPC_transf_rezago_25
        self.INPC_transf_rezago_26 = INPC_transf_rezago_26
    
    def fit(self, X, y=None):
        self.cols_idx_ = np.array([
                                  self.INPC_transf_rezago_1,
                                  self.INPC_transf_rezago_2,
                                  self.INPC_transf_rezago_3,
                                  self.INPC_transf_rezago_4,
                                  self.INPC_transf_rezago_10,
                                  self.INPC_transf_rezago_11,
                                  self.INPC_transf_rezago_12,
                                  self.INPC_transf_rezago_13,
                                  self.INPC_transf_rezago_14,
                                  self.INPC_transf_rezago_22,
                                  self.INPC_transf_rezago_23,
                                  self.INPC_transf_rezago_24,
                                  self.INPC_transf_rezago_25,
                                  self.INPC_transf_rezago_26
                                  ])
        return self
    
    def transform(self, X, y=None):
        if np.all(~self.cols_idx_):
            return np.zeros((X.shape[0], 1))
        else:
            return X.iloc[:, self.cols_idx_]

class SelectorMacro(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 INPC_transf_rezago_1     =True,
                 INPC_transf_rezago_2     =True,
                 INPC_transf_rezago_3     =True,
                 INPC_transf_rezago_4     =True,
                 INPC_transf_rezago_10    =True,
                 INPC_transf_rezago_11    =True,
                 INPC_transf_rezago_12    =True,
                 INPC_transf_rezago_13    =True,
                 INPC_transf_rezago_14    =True,
                 INPC_transf_rezago_22    =True,
                 INPC_transf_rezago_23    =True,
                 INPC_transf_rezago_24    =True,
                 INPC_transf_rezago_25    =True,
                 INPC_transf_rezago_26    =True,
                 usd_transf_rezago_1      =True,
                 usd_transf_rezago_2      =True,
                 tiie_transf_rezago_1     =True,
                 tiie_transf_rezago_2     =True,
                 IPC_transf_rezago_1      =True,
                 IPC_transf_rezago_2      =True,
                 WTI_transf_rezago_1      =True,
                 WTI_transf_rezago_2      =True,
                 IGAE_transf_rezago_1     =True,
                 IGAE_transf_rezago_2     =True,
                 yield_usa_transf_rezago_1=True, 
                 yield_usa_transf_rezago_2=True,
                 yield_mex_transf_rezago_1=True, 
                 yield_mex_transf_rezago_2=True):
        self.INPC_transf_rezago_1     = INPC_transf_rezago_1     
        self.INPC_transf_rezago_2     = INPC_transf_rezago_2     
        self.INPC_transf_rezago_3     = INPC_transf_rezago_3     
        self.INPC_transf_rezago_4     = INPC_transf_rezago_4     
        self.INPC_transf_rezago_10    = INPC_transf_rezago_10    
        self.INPC_transf_rezago_11    = INPC_transf_rezago_11    
        self.INPC_transf_rezago_12    = INPC_transf_rezago_12    
        self.INPC_transf_rezago_13    = INPC_transf_rezago_13    
        self.INPC_transf_rezago_14    = INPC_transf_rezago_14    
        self.INPC_transf_rezago_22    = INPC_transf_rezago_22    
        self.INPC_transf_rezago_23    = INPC_transf_rezago_23    
        self.INPC_transf_rezago_24    = INPC_transf_rezago_24    
        self.INPC_transf_rezago_25    = INPC_transf_rezago_25    
        self.INPC_transf_rezago_26    = INPC_transf_rezago_26    
        self.usd_transf_rezago_1      = usd_transf_rezago_1      
        self.usd_transf_rezago_2      = usd_transf_rezago_2      
        self.tiie_transf_rezago_1     = tiie_transf_rezago_1     
        self.tiie_transf_rezago_2     = tiie_transf_rezago_2     
        self.IPC_transf_rezago_1      = IPC_transf_rezago_1      
        self.IPC_transf_rezago_2      = IPC_transf_rezago_2      
        self.WTI_transf_rezago_1      = WTI_transf_rezago_1      
        self.WTI_transf_rezago_2      = WTI_transf_rezago_2      
        self.IGAE_transf_rezago_1     = IGAE_transf_rezago_1     
        self.IGAE_transf_rezago_2     = IGAE_transf_rezago_2     
        self.yield_usa_transf_rezago_1= yield_usa_transf_rezago_1
        self.yield_usa_transf_rezago_2= yield_usa_transf_rezago_2
        self.yield_mex_transf_rezago_1= yield_mex_transf_rezago_1
        self.yield_mex_transf_rezago_2= yield_mex_transf_rezago_2

    def fit(self, X, y=None):
        self.cols_idx_ = np.array([
                                  self.INPC_transf_rezago_1     ,
                                  self.INPC_transf_rezago_2     ,
                                  self.INPC_transf_rezago_3     ,
                                  self.INPC_transf_rezago_4     ,
                                  self.INPC_transf_rezago_10    ,
                                  self.INPC_transf_rezago_11    ,
                                  self.INPC_transf_rezago_12    ,
                                  self.INPC_transf_rezago_13    ,
                                  self.INPC_transf_rezago_14    ,
                                  self.INPC_transf_rezago_22    ,
                                  self.INPC_transf_rezago_23    ,
                                  self.INPC_transf_rezago_24    ,
                                  self.INPC_transf_rezago_25    ,
                                  self.INPC_transf_rezago_26    ,
                                  self.usd_transf_rezago_1      ,
                                  self.usd_transf_rezago_2      ,
                                  self.tiie_transf_rezago_1     ,
                                  self.tiie_transf_rezago_2     ,
                                  self.IPC_transf_rezago_1      ,
                                  self.IPC_transf_rezago_2      ,
                                  self.WTI_transf_rezago_1      ,
                                  self.WTI_transf_rezago_2      ,
                                  self.IGAE_transf_rezago_1     ,
                                  self.IGAE_transf_rezago_2     ,
                                  self.yield_usa_transf_rezago_1,
                                  self.yield_usa_transf_rezago_2,
                                  self.yield_mex_transf_rezago_1,
                                  self.yield_mex_transf_rezago_2
                                  ])
        return self
    
    def transform(self, X, y=None):
        if np.all(~self.cols_idx_):
            return np.zeros((X.shape[0], 1))
        else:
            return X.iloc[:, self.cols_idx_]    
        
class MySplits():
    
    def __init__(self, n_splits=3, test_size=48):
        self.n_splits = n_splits
        self.test_size = test_size
         
    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        test_starts = range(n_samples - self.n_splits * self.test_size, n_samples, self.test_size)
        for test_start in test_starts:
            yield (indices[:test_start], indices[test_start:test_start + self.test_size])
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def get_test_size(self):
        return self.test_size
    
    def get_splits_weights(self, X):
        n_samples = X.shape[0]
        train_sizes = np.arange(n_samples - self.n_splits * self.test_size, n_samples, self.test_size)
        train_weights = train_sizes / n_samples
        return train_weights / train_weights.sum()
    
class StackedRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_models, meta_model, cv=KFold(n_splits=5, shuffle=True, random_state=0)):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv
        
    def fit(self, X, y):
        X_stack = np.zeros((X.shape[0], len(self.base_models)))
        for train_idx, val_idx in self.cv.split(X):
            self.base_models_ = [mod.fit(X.iloc[train_idx], y.iloc[train_idx]) for mod in self.base_models]
            X_stack[val_idx] = np.column_stack([mod.predict(X.iloc[val_idx]) for mod in self.base_models_])
        self.base_models_ = [model.fit(X, y) for model in self.base_models]
        self.meta_model_ = self.meta_model.fit(X_stack, y)
        return self
                           
    def predict(self, X, y=None):
        X_stack = np.column_stack([mod.predict(X) for mod in self.base_models_])
        return self.meta_model_.predict(X_stack)
        