import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
import re
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler, Imputer
from sklearn.externals import joblib

d_path = '/Users/jacobhumber/Google Drive/Data Science/Credit Scoring/lending-club-loan-data/'

d = pd.read_csv(d_path+'loan.csv', dtype = {'desc': 'object', 'verification_status_joint': 'object'})

#####Default flag#######
#https://help.lendingclub.com/hc/en-us/articles/216127747

#only going to use Fully Paid vs Default

d.loc[d['loan_status'] == 'Fully Paid', 'default'] = 0
d.loc[(d['loan_status'] == 'Charged Off') | (d['loan_status'] == 'Default'), 'default'] = 1

d['default'].value_counts(dropna = False)

#drop if default is missing

d = d[~np.isnan(d['default'])]

#Removing loan_status, avoid potential target leak

d = d.drop(['loan_status'], axis = 1)

#####Clean Data##########

#TO DO
#Look at string, which can be turn qualitative
#look at correlation and ACE between independent variables and depend as well as within depend
#look for outlier in data
#look for coded data
#change dates from string to date object

#####QUAL DATA###

d.dtypes.value_counts()

qual_vars = d.dtypes[d.dtypes == 'object'].index

pd.options.display.max_columns = 50

d[qual_vars].describe()

#Convert all variables which are object dtypes to str, in pandas an series with a object class can contain entires of different types, this is both dangerous and annoying

d[qual_vars] = d[qual_vars].apply(lambda x: x.astype('str'))

#Drop alot of missing

d[qual_vars].apply(lambda x: np.sum(x == 'nan'))

d = d.drop(['verification_status_joint', 'next_pymnt_d', 'sub_grade'], axis = 1)

qual_vars = d.dtypes[d.dtypes == 'object'].index

#Drop alot that are too unique 

d = d.drop(['url','desc'], axis = 1)

qual_vars = d.dtypes[d.dtypes == 'object'].index

d[qual_vars].describe()

#Drop unhelpful variable
d = d.drop(['last_pymnt_d'], axis = 1)

#Clean dirty variables 

d['emp_title'] = d['emp_title'].str.lower()

employ = d['emp_title'].value_counts()

d.loc[d['emp_title'] == 'nan', 'emp_title'] = 'Missing'

#finding all employment names that occure less than 200 so I can set to other 
emp_other_logic = d['emp_title'].isin(employ[employ < 650].index)

d.loc[emp_other_logic, 'emp_title'] = 'Other'

d.loc[d['emp_title'] == 'rn', 'emp_title'] = 'registered nurse'
d.loc[d['emp_title'] == 'office manager', 'emp_title'] = 'manager'
d.loc[d['emp_title'] == 'project manager', 'emp_title'] = 'manager'

d['emp_title'].value_counts()


d['title'].value_counts()

d['title'] = d['title'].str.lower()

#uniting some common titles
d.loc[d['title'].str.contains('consolidation|consolidate'), 'title'] = 'consolidation'
d.loc[d['title'].str.contains('wedding'), 'title'] = 'wedding'
d.loc[d['title'].str.contains('home'), 'title'] = 'home'
d.loc[d['title'].str.contains('credit card|cc'), 'title'] = 'credit card'
d.loc[d['title'].str.contains('car'), 'title'] = 'car'
d.loc[d['title'].str.contains('personal'), 'title'] = 'personal'
d.loc[d['title'].str.contains('debt'), 'title'] = 'debt'
d.loc[d['title'].str.contains('pool'), 'title'] = 'pool'
d.loc[d['title'].str.contains('major'), 'title'] = 'major purchase'
d.loc[d['title'].str.contains('medical'), 'title'] = 'medical'
d.loc[d['title'].str.contains('moving'), 'title'] = 'moving'
d.loc[d['title'].str.contains('loan'), 'title'] = 'loan'

title = d['title'].value_counts()

title_other_logic = d['title'].isin(title[title < 1000].index)

d.loc[title_other_logic, 'title'] = 'other'

qual_vars = d.dtypes[d.dtypes == 'object'].index

d[qual_vars].describe()

###QUANT###

quant_vars = d.dtypes[d.dtypes != 'object'].index.drop('default')

quant_per_miss = d.isnull().sum() / d.shape[0] 

#Dropping variables with a lot of missing
d = d.drop(quant_per_miss[quant_per_miss > .99].index, axis = 1)

#Turning missing into dummy
dum_list = ['tot_coll_amt', 'mths_since_last_major_derog', 'mths_since_last_record', 'mths_since_last_delinq']
dum_list_nam = [v + 'is_null' for v in dum_list]

d[dum_list_nam] = pd.DataFrame(np.where(d[dum_list].isnull(), 1, 0), index = d.index)

#Correlation between co-variates

quant_vars = d.dtypes[d.dtypes != 'object'].index.drop('default')

d[quant_vars].describe()

corr_mat = d[quant_vars].corr()

corr_mat[(corr_mat > .9) | (corr_mat < -.9)]

corr_mat[(corr_mat > .9) | (corr_mat < -.9)][quant_vars[0:20]]


drop_vars = ['policy_code'
             ,'total_pymnt_inv'
             ,'total_rec_prncp'
             ,'out_prncp_inv'
             ,'member_id'
             ,'funded_amnt'
             ,'funded_amnt_inv'
             ,'installment'
             ,'id']

quant_vars = quant_vars.drop(pd.Index(drop_vars))

#Correlation between default

d[quant_vars.append(pd.Index(['default']))].corr()['default']
quant_vars = quant_vars.drop(pd.Index(['total_rec_int']))



####SIMPLE MODEL

d[qual_vars].describe()
qual_vars_simple = ['home_ownership', 'application_type'] 

d[quant_vars].describe()
quant_vars_simple = ['loan_amnt', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'tot_cur_bal'] 

simple_vars = qual_vars_simple+quant_vars_simple
simple_vars.append('default')


#training and testing
dtrain, dtest = train_test_split(d[simple_vars], test_size=0.2)

#setting up pipeline
def cook_features(quant, cat, DF):
    
    quant_steps = [Imputer(missing_values='NaN', strategy='mean', axis=0), StandardScaler()]
    
    quantTrans = [([v], quant_steps) for v in quant]
    catTrans = [(v, LabelBinarizer()) for v in cat]
    
    return DataFrameMapper(quantTrans + catTrans, df_out = DF)

simple_pipe =  Pipeline([('cook_features', cook_features(quant = quant_vars_simple, cat = qual_vars_simple, DF = False)) 
                        ,('logit', LogisticRegression())])

X = dtrain.drop(['default'], axis = 1)
y = dtrain['default']
simple_pipe.fit(X = X, y = y)

X_test = dtest.drop(['default'], axis = 1)
y_test = dtest['default']

y_hat = simple_pipe.predict(X_test)

#It's a piece of shit but who cares it will work for my purposes
roc_auc_score(y_test, y_hat)

#Let's pickle it
filename = '/Users/jacobhumber/python_projects/simple_model_api/ml_model/simple_logit.pkl'
joblib.dump(simple_pipe, filename)

unpickle = joblib.load(filename)
unpickle.predict(X_test)

X_test.head()

X_test.head().to_csv('/Users/jacobhumber/python_projects/simple_model_api/ml_model/test_head.csv', index = False)



#pd.set_option('display.height', 1000)
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)
