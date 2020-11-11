import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import category_encoders as ce
from scipy.stats import zscore
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


## Loading data from csv to a dataframe
filename = 'Hiring_Challenge.csv'
hire_df = pd.read_csv(filename)

hire_df_clean = hire_df.copy()

## checking the total ? occurences
#print("Total occurences of ?")
#hire_df_clean.isin(['?']).sum()


###Replacing all ? with nan for further cleaning
hire_df_clean = hire_df_clean.replace('?', np.nan)

### Converting continuous objects to numerical data type
#####To handle missing values in continuous variable, converting obect to numerical
hire_df_clean['C2'] = hire_df_clean['C2'].astype('float')
###C14 is of type int but is converted to float as we have NaN which is a float value
hire_df_clean['C14'] = hire_df_clean['C14'].astype('float')


####Imputing categorical variables with most frequent value and numerical values with median
#hire_df.columns()
for C in hire_df_clean:
    if(hire_df_clean[C].dtype == np.dtype('O')):
        hire_df_clean[C].fillna(hire_df_clean[C].value_counts().index[0], inplace = True)
    else:
        hire_df_clean[C].fillna(np.nanmedian(hire_df_clean[C]), inplace = True)

##converting C14 back to int as NaN values are cleared
hire_df_clean['C14'] = hire_df_clean['C14'].astype('int64')

### Separating all Categorical variables
hire_df_cat = hire_df_clean.select_dtypes(include = 'object').copy()

### Seperating all numerical variables
hire_df_num = hire_df_clean.select_dtypes(exclude = 'object').copy()


## As we are working on a supervised ML problem, we should add dependent variable to our dataframe
hire_df_cat['Hired'] = hire_df_clean.loc[hire_df_clean.index, 'Hired'].copy()

## Dropping C5 column, to avoid redundancy
hire_df_cat.drop('C5', axis = 1, inplace = True)

## Manipulating C7 so it contains only 2 categories, variable containing single letter as s and variable containing multiple letter as m
hire_df_cat['C7'] = hire_df_cat['C7'].apply(lambda x:'m' if len(x) > 1 else 's')

## Dropping C6 feature
hire_df_cat.drop('C6', axis = 1, inplace = True)

### Converting categorical variables into numerical variables ,when we have less number of unique values in that column
hire_df_cat = pd.get_dummies(hire_df_cat, columns = ['C1','C7', 'C9', 'C10', 'C12'], drop_first = True)

### for C4 and C13 few values have very less entries less than 10, these need to be dropped as well
hire_df_cat = pd.get_dummies(hire_df_cat, columns =['C4', 'C13'])

## Dropping less value and the first value
hire_df_cat.drop(['C4_l', 'C4_u', 'C13_p', 'C13_g'], axis = 1, inplace = True)


## preparing data frame to pass to power transformer
hire_df_num_out = hire_df_num.loc[:, hire_df_num.columns != 'Hired']


# Applying powertransformer on data
p = PowerTransformer(method = 'yeo-johnson')
hire_df_num_fit = p.fit_transform(hire_df_num_out)

## Converting numpy array to dataframe
#print(hire_df_num.columns[:-1])
hire_df_num_clean = pd.DataFrame(data=hire_df_num_fit, index = hire_df_num.index, columns=hire_df_num.columns[:-1])

### As we have only few outliers replacing them with median 
Q1 = hire_df_num_clean['C14'].quantile(0.25)
Q3 = hire_df_num_clean['C14'].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + (1.5 * IQR )
median = hire_df_num_clean.loc[hire_df_num_clean['C14']< upper_fence, 'C14'].median()
hire_df_num_clean.loc[hire_df_num_clean.C14 > upper_fence, 'C14'] = np.nan
hire_df_num_clean.fillna(median,inplace=True)


lastcol = hire_df.pop("Hired")
hire_df_num_clean.insert(6, 'Hired',  lastcol)
## Merging the categorical and numerical variables
hire_df_num_clean.drop('Hired', axis = 1, inplace = True)
X = pd.merge(hire_df_num_clean, hire_df_cat, left_index = True, right_index = True)

#Splitting data into features and target variables
col_name = 'Hired'
last_col = X.pop(col_name)
y = last_col

# Scaling Data, as few models work based on distance if not scaled the performance will be effected
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_scaled = scale.fit_transform(X)

## Splitting Training data and testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)


### Naive Bayes
gnb = GaussianNB()

### Logistic Regression
lr = LogisticRegression(max_iter = 2000)

### Decision Tree Classifier
dt = tree.DecisionTreeClassifier(random_state = 1)

### K Nearest Neighbors
knn = KNeighborsClassifier()

### Random Forest Classifier
rf = RandomForestClassifier(random_state = 1)

## Support Vector Machine with scaled data
svc = SVC(probability = True)

### XGBoost with scaled data
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state =1)

from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),('rf',rf),('gnb',gnb),('svc',svc),('xgb',xgb)], voting = 'soft') 
cv = cross_val_score(voting_clf,X_train,y_train,cv=5)

### Tuning Logistic Regression
lr = LogisticRegression()
param_grid = {'max_iter' : [2000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['liblinear']}

clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(X_train_scaled,y_train)


#### Tunning knn classifier
knn = KNeighborsClassifier()
param_grid = {'n_neighbors' : [3,5,7,9],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'p' : [1,2]}
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train_scaled,y_train)

#### Tuning SVC
svc = SVC(probability = True)
param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100]},
                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100]}]
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train_scaled,y_train)


#### Tuning Random Forest Classifier
rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [400,450,500,550],
               'criterion':['gini','entropy'],
                                  'bootstrap': [True],
                                  'max_depth': [15, 20, 25],
                                  'max_features': ['auto','sqrt', 10],
                                  'min_samples_leaf': [2,3],
                                  'min_samples_split': [2,3]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train_scaled,y_train)

######## Tuning XGB classifier
xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators': [450,500,550],
    'colsample_bytree': [0.75,0.8,0.85],
    'max_depth': [None],
    'reg_alpha': [1],
    'reg_lambda': [2, 5, 10],
    'subsample': [0.55, 0.6, .65],
    'learning_rate':[0.5],
    'gamma':[.5,1,2],
    'min_child_weight':[0.01],
    'sampling_method': ['uniform']
}

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train_scaled,y_train)



best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_


## Ensembling on all models 
voting_clf_all = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('xgb', best_xgb),('lr', best_lr)], voting = 'soft')
voting_clf_all.fit(X_train_scaled, y_train)
y_vc_all = voting_clf_all.predict(X_test_scaled).astype(int)

## Calculating Accuracy
report = classification_report(y_test, y_vc_all)
print("Performance Metrics of Ensemble Model")
print(report)
