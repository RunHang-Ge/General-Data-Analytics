import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt



# ---------------------------- Data Processing ----------------------------- #
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

raw_data = pd.read_excel(r'ABCD.xlsx',sheet_name = 'A')

# Select required columns
required_cols = []
MTD_cols = []
for i in raw_data.columns:
    if i.split("_")[-1] == 'MTD':
        MTD_cols.append(i)
for i in raw_data.columns[np.where(raw_data.columns == 'LAST_LOGIN_DATE')[0][0]:]:
    required_cols.append(i)
for i in raw_data.columns[:np.where(raw_data.columns == 'FAST_INBOUND_$_MTD')[0][0]]:
    required_cols.append(i)
required_cols = required_cols + MTD_cols
select_data = raw_data[required_cols]

# Create a function to convert target numerical variable into categorical variable.
def _create_datasubset(cols):
    df = select_data[['NPS'] + cols]
    df['NPS_Class'] = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        if df.loc[i, 'NPS'] <= 6:
            df.loc[i, 'NPS_Class'] = 'Low'
        elif df.loc[i, 'NPS'] <= 8:
            df.loc[i, 'NPS_Class'] = 'Mid'
        else:
            df.loc[i, 'NPS_Class'] = 'High'
    return df

# Extract special data subsets. Rest columns are all numerical.
date_cols = ['LAST_LOGIN_DATE',
             'ACCOUNT_OPEN_DATE']
date_data = _create_datasubset(date_cols)

cate_cols = ['SOURCE',
             'LOGIN_GROUP',
             'GENDER']
cate_data = _create_datasubset(cate_cols)
label_Encoder = LabelEncoder()
for i in cate_cols:
    cate_data[i] = label_Encoder.fit_transform(cate_data[i])

bool_cols = ['CUST_GENDER',
             'IS_LINK',
             'IS_REGISTERED',
             'IS_APPROVED']
bool_data = _create_datasubset(bool_cols)
label_Encoder = LabelEncoder()
for i in bool_cols:
    bool_data[i] = label_Encoder.fit_transform(bool_data[i])

# function to create dummy variables of categorical features
def _get_dummies(df, cats):
    for col in cats:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df

# If too many categorical variables, we can apply selection on categorical variables.
Cate_Select = SelectKBest(chi2, k=1).fit_transform(cate_data.drop('NPS_Class', axis=1), cate_data['NPS_Class'])

# Generate the final dataset, which includes numerical, boolean and categorical variables.
final_data = pd.concat([bool_data, # + dummy var if you want to insert category variable.
                        select_data.drop(labels = bool_cols+date_cols+['NPS','NPS_GROUP'], axis = 1)],
                        axis = 1)
final_data = _get_dummies(final_data, cate_cols)

# Fill NaN value
for i in final_data.columns:
    if pd.isna(final_data.loc[:,i]).sum() > 0:
        final_data.loc[:,i].fillna(0,inplace = True)

# Standardization
scaled_data = StandardScaler().fit_transform(final_data.drop(['NPS','NPS_Class'], axis = 1))
scaled_data = pd.DataFrame(scaled_data, columns = final_data.drop(['NPS','NPS_Class'], axis = 1).columns)
scaled_data = pd.concat([final_data[['NPS']],scaled_data],axis = 1)
scaled_data['NPS_Class'] = LabelEncoder().fit_transform(final_data.NPS_Class)



# ---------------------------- Data Analytics ----------------------------- #
import seaborn as sns
import palettable
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Correlation matrix
corr_matrix = pd.DataFrame(np.corrcoef(scaled_data.T),
                           columns = scaled_data.columns,
                           index = scaled_data.columns)
plt.figure(dpi=150)
sns.heatmap(data = corr_matrix.round(2),
           cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
           annot=True,
           annot_kws={'size':7, 'weight':'normal', 'color':'blue'},
           )
plt.title('Correlation')

# Basic analysis to find whether there may exist statistical difference.
low = date_data[date_data['NPS_Class'] == 'Low'][['LAST_LOGIN_DATE', 'NPS']].groupby(
    ['LAST_LOGIN_DATE'],dropna = True).count()
mid = date_data[date_data['NPS_Class'] == 'Mid'][['LAST_LOGIN_DATE', 'NPS']].groupby(
    ['LAST_LOGIN_DATE'],dropna = True).count()
high = date_data[date_data['NPS_Class'] == 'High'][['LAST_LOGIN_DATE', 'NPS']].groupby(
    ['LAST_LOGIN_DATE'],dropna = True).count()

print(date_data[['LAST_LOGIN_DATE', 'NPS']].groupby(['LAST_LOGIN_DATE'],dropna = True).count().sum())
plt.plot(low, color = 'Red')
plt.plot(mid, color = 'Green')
plt.plot(high, color = 'Blue')
plt.xlim(datetime.datetime(2022,12,22),datetime.datetime(2023,1,30))
plt.legend(['Low','Mid','High'])
plt.xlabel('LAST_LOGIN_DATE')
plt.ylabel('Number of clients')
plt.show()

# Calculate baseline results using all features
test_result = []
for i in range(500): # Do 500 times to reduce randomness
    x_train, x_test, y_train, y_test = train_test_split(scaled_data, y_class, test_size=0.2)
    base_GBDT = GradientBoostingClassifier(max_depth = 5).fit(x_train, y_train)
    #print(cross_val_score(tree_result,x_train, y_train, cv=10,scoring='accuracy').mean())
    test_result.append(base_GBDT.score(x_test, y_test))
np.mean(test_result)

# linear regression. We can find all varibles with p>0.05
x_addConst = sm.add_constant(scaled_data.drop('NPS_Class',axis=1))
model = sm.OLS(final_data['NPS'],x_addConst).fit()
print(model.summary())



# ---------------------------- Feature selection ----------------------------- #
"""
Feature selection contains five parts:
1. VIF selection
2. Regularization regression
3. Feature importance in ML methods
4. Mutual information
5. An ensemble voting combination of the above results.
"""
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score

# Generate initial VIF matrix.
X = scaled_data.assign(const=1)
y_num = final_data['NPS']
y_class = final_data['NPS_Class']
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns

# Create a function to delete variables which have too high VIF.
def vif_process(X_init, vif_init):
    VIF = vif_init
    # Nan means this column is all zeros. So we first delete NA
    X_init = X_init.drop(labels=VIF[VIF['VIF'].isna()]['variable'], axis=1)
    # inf means perfect correlation. Here we just randomly delete one variable among inf value.
    while VIF[VIF['VIF'] == float('inf')].shape[0] > 0:
        inf_var = VIF[VIF['VIF'] == float('inf')]['variable']
        X_init = X_init.drop(labels=(inf_var.iloc[np.random.randint(0, len(inf_var))]), axis=1)
        VIF = pd.DataFrame()
        VIF['VIF'] = [variance_inflation_factor(X_init.values, i) for i in range(X_init.shape[1])]
        VIF['variable'] = X_init.columns
    # Now we delete all variable with vif > 10
    X_init = X_init.drop(labels=(VIF[VIF['VIF'] > 10]['variable']), axis=1)
    VIF = pd.DataFrame()
    VIF['VIF'] = [variance_inflation_factor(X_init.values, i) for i in range(X_init.shape[1])]
    VIF['variable'] = X_init.columns
    return X_init, VIF

X, vif = vif_process(X, vif)

# Transform X and Y to required format.
X = X.drop(labels = 'const', axis=1)
y_class_trans = LabelEncoder().fit_transform(y_class)

# Here we want to calculate the frequency based on the following results.
feature_appearance = pd.Series(data=np.zeros(len(X.columns)), index=X.columns)

# This is basic logistic regression.
# Since this is a multi-class problem, we choose "ovr" and take average on model's outcome.
model = LogisticRegression(penalty='l1',C=1,solver='liblinear', multi_class='ovr')
model.fit(X, y_class_trans)
feature_lasso_One = X.columns[np.where(model.coef_[0] != 0)]
feature_lasso_Two = X.columns[np.where(model.coef_[1] != 0)]
feature_lasso_Three = X.columns[np.where(model.coef_[2] != 0)]
# This is the results from Logistic model with L1 penalty.
# Because it returns three models, each one we only take weight as 0.33
for i in feature_lasso_One:
    feature_appearance[i] += 0.33
for i in feature_lasso_Two:
    feature_appearance[i] += 0.33
for i in feature_lasso_Three:
    feature_appearance[i] += 0.33

"""
The core: some variables have 0 in L1 penalty, but actually have similar coefficients in L2 penalty. 
So we find those similar varibles from L2, and average their coefficients in L1.
So we can extend the variables based on L1 results.
"""
class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0, fit_intercept=True,
                 intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear',
                 max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                    intercept_scaling=intercept_scaling, class_weight=class_weight,
                                    random_state=random_state, solver=solver,
                                    max_iter=max_iter, multi_class=multi_class,
                                    verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                     intercept_scaling=intercept_scaling, class_weight=class_weight,
                                     random_state=random_state, solver=solver,
                                     max_iter=max_iter, multi_class=multi_class,
                                     verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        # Fit l1 logistic model
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        self.l2.fit(X, y, sample_weight=sample_weight)
        # Average coef
        cntOfRow, cntOfCol = self.coef_.shape
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]

                if coef != 0:
                    idx = [j]
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        if abs(coef1 - coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self

feature_L1L2 = SelectFromModel(LR(threshold=0.0001, C=0.06)).fit(X, y_class_trans)
for i in X.columns[feature_L1L2.get_support()]:
    feature_appearance[i] += 1

# Create a list of classifiers to do feature selection
estimator = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gbt', GradientBoostingClassifier()),
    ('dt', DecisionTreeClassifier(max_depth = 5)),
    ('svc', LinearSVC(random_state=42))
]
for name, estimator in estimator:
    select_model = SelectFromModel(estimator).fit(X, y_class)
    for i in X.columns[select_model.get_support()]:
        feature_appearance[i] += 1
    print(f"{name}: Selected {X.columns[select_model.get_support()]} features")
    print(f"{name}'s Feature Importance: {select_model.estimator_.feature_importances_}\n")

# This is mutual information
mutual_info_list = pd.Series()
for i in X.columns:
    mutual_info_list[i] = mutual_info_score(X[i],y_class)
feature_MI = X.columns[mutual_info_list.sort_values(ascending = False).index[:10]]
for i in feature_MI:
    feature_appearance[i] += 1

feature_appearance.sort_values(ascending=False)

# Test whether the feature selection works
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

X_Reduced = X[feature_appearance.sort_values(ascending=False).index[0:10]]
test_result = []
for i in range(500):
    x_train, x_test, y_train, y_test = train_test_split(X_Reduced, y_class, test_size=0.2)
    base_GBDT = GradientBoostingClassifier(max_depth = 5).fit(x_train, y_train)
    test_result.append(base_GBDT.score(x_test, y_test))

np.mean(test_result)



# ---------------------------- Shapley Value ----------------------------- #
import shap

# Use shapley value to see features' contribution
estimator = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gbt', GradientBoostingClassifier()),
    ('dt', DecisionTreeClassifier(max_depth = 5)),
    ('logit', LogisticRegression(penalty='l1',C=0.01,solver='liblinear')),
    ('svc', LinearSVC(random_state=42))
]

for name, estimator in estimator:
    select_model = estimator.fit(x_addConst, y_grab)
    explainer = shap.Explainer(select_model.predict, x_addConst)
    shap_values = explainer(x_addConst)
    print(shap_values)
    shap.plots.waterfall(shap_values[0])
