# -*- coding: utf-8 -*-
import numpy as np
#import scipy.stats as stats
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Imputer, FunctionTransformer, LabelBinarizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import TransformerMixin

UNV = pd.read_csv('university_train.csv')
UNV_test = pd.read_csv('university_test.csv')
y = UNV['percent_on_student_loan']

multinomial = ['STABBR', 'ACCREDAGENCY', 'NUMBRANCH', 'PREDDEG', 'HIGHDEG', 
               'CONTROL', 'LOCALE', 'CCUGPROF', 'CCSIZSET', 'RELAFFIL']
binomial = ['MAIN', 'HBCU', 'PBI', 'MENONLY', 'WOMENONLY', 'DISTANCEONLY']
numeric = ['UGDS', 'AGE_ENTRY', 'FEMALE', 'MARRIED', 'DEPENDENT', 'MD_FAMINC']


def Get_Columns(data=None, columns=numeric + binomial):
    #return data[:,1].reshape(-1, 1)
    return data[columns]
columns_extractor = FunctionTransformer(Get_Columns, validate=False)
imputer = Imputer(missing_values=np.nan, strategy='mean', axis=1)
    
class FeatureExtractor(TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.column].values.reshape(-1,1)

LabelBinarizers = [LabelBinarizer() for item in multinomial]
    
def DropFirst(df=None): 
    return pd.DataFrame(df).iloc[:,1:]    
firstcolumn_dropper = FunctionTransformer(DropFirst, validate=False)    

# pipes for (numeric + binomial) columns as first feature union step
pipes = make_pipeline(columns_extractor, imputer) 
feature_union_steps = [('numeric+binomial', pipes)] 
# adding pipes for multinomial columns to feature union steps
for i, column in enumerate(multinomial):
    pipe = make_pipeline(FeatureExtractor(column), 
                         LabelBinarizers[i], 
                         firstcolumn_dropper)
    feature_union_steps.append((column, pipe))
fu = FeatureUnion(feature_union_steps)

sgd_params = {
    'loss':['squared_loss','huber'],
    'penalty':['l1','l2', 'elasticnet'],
    'alpha':np.logspace(-5,1,6),
    'l1_ratio':[i/10.0 for i in range(11)] 
}
sgd_reg = SGDRegressor()
sgd_reg_gs = GridSearchCV(sgd_reg, sgd_params, cv=3, verbose=1, 
                          scoring='neg_mean_squared_error')

fu_pipe = make_pipeline(fu, StandardScaler()) 
X = fu_pipe.fit_transform(UNV)
X_test = fu_pipe.transform(UNV_test)

sgd_reg_gs.fit(X, y)
print(sgd_reg_gs.best_params_)
print(sgd_reg_gs.best_score_)
y_test = sgd_reg_gs.predict(X_test)
#{'alpha': 0.63095734448019303, 'penalty': 'l1', 'loss': 'squared_loss', 'l1_ratio': 0.0} -355.845509062

#final_pipe = make_pipeline(fu, StandardScaler(), sgd_reg_gs)
#print(final_pipe.named_steps)
#final_pipe.fit(UNV, y)
# y_test = final_pipe.predict(UNV_test)

def csv_spitter(y_test=None, test=UNV_test):
    prediction = pd.DataFrame(y_test, columns=['Prediction'])
    Id = test[['id_number']]
    Id.join(prediction).to_csv('reg_predictions.csv', index=False)
    
#csv_spitter(y_test, UNV_test)