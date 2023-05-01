import pandas as pd
import numpy as np
import datetime
import re



# ----------------- Data Processing ----------------------- #
# Extract the account open date.
Account_openDate = []
for i in CasaDaily_data['ACCOUNT_OPEN_DATE']:
    Account_openDate.append(datetime.datetime.strptime(i.split("T")[0],"%Y-%m-%d"))
CasaDaily_data['ACCOUNT_OPEN_DATE'] = Account_openDate

# Delete repeated data, whose transaction amount is negative.
temp_index = []
for i in range(SaveBoost_data.shape[0]):
    if SaveBoost_data['TRANSACTION_AMOUNT'][i] > 0:
        temp_index.append(i)
SaveBoost_data_filter = SaveBoost_data[["VALUE_DATE", "ACCOUNT_ID",
                                        "TIMESTAMP", "BATCH_REMARKS"]].iloc[temp_index,:]

# Change text $0, $2.5, $5, $7.5, $10 into 0,1,2,3,4.
SaveBoost_data_filter['Boosts_count'] = np.zeros(SaveBoost_data_filter.shape[0])
for i in range(SaveBoost_data_filter.shape[0]):
    temp = re.split('S|in| ', SaveBoost_data_filter.iloc[i,3])
    if temp[6] == "$2.50":
        SaveBoost_data_filter.iloc[i,4] = 1
    elif temp[6] == "$5.00":
        SaveBoost_data_filter.iloc[i,4] = 2
    elif temp[6] == "$7.50":
        SaveBoost_data_filter.iloc[i,4] = 3
    elif temp[6] == "$10.00":
        SaveBoost_data_filter.iloc[i,4] = 4

SaveBoost_Date = []
for i in SaveBoost_data_filter['TIMESTAMP']:
    SaveBoost_Date.append(datetime.datetime.strptime(i.split("T")[0],"%Y-%m-%d"))
SaveBoost_data_filter['Boosts_DATE'] = SaveBoost_Date

# Delete outlier accounts which have not topped up any money
print('Number of records before deleting: ', CasaDaily_data.shape)
AlwaysZero_Number = []
for i in np.unique(CasaDaily_data['ACCOUNT_NUMBER']):
    if np.all(CasaDaily_data[CasaDaily_data['ACCOUNT_NUMBER'] == i]['TOTAL_ACCOUNT_BALANCE'] == 0):
        AlwaysZero_Number.append(i)
        CasaDaily_data = CasaDaily_data[CasaDaily_data['ACCOUNT_NUMBER'] != i]
print('Number of records after deleting: ', CasaDaily_data.shape)
print('Number of accounts who never top up money:', len(AlwaysZero_Number))
CasaDaily_data = CasaDaily_data.reset_index(drop=True)

# Get the customer subset which open accounts before 11.08
BeforeCamp_Account = []
for i in range(CasaDaily_data.shape[0]):
    if (CasaDaily_data['ACCOUNT_OPEN_DATE'][i] < datetime.datetime(2022, 11, 8, 0, 0)) & (CasaDaily_data['ACCOUNT_OPEN_DATE'][i] >= datetime.datetime(2022, 7, 28, 0, 0)):
        BeforeCamp_Account.append(i)
BeforeCamp_Daily = CasaDaily_data.iloc[BeforeCamp_Account,:]

"""
Create a new dataset, which:
1. change "Account_Open_Date" into "Days_Before".
2. Determine “Treatment” by whether a customer has received boosts or not
3. Determine "Target" by calculating the account balance difference between campaign has applied or not.
"""
# Calculate the required elements.
BeforeCamp_Change = pd.DataFrame(columns = ['Account_Number', 'Account_Open_Date', 'Source', 'Days_Before', 'Boost_Number', 'Total_Balance_Change', 'Treatment'])
BeforeCamp_Change[['Account_Number', 'Account_Open_Date']] = BeforeCamp_Daily[['ACCOUNT_NUMBER','ACCOUNT_OPEN_DATE']].groupby(['ACCOUNT_NUMBER']).mean().reset_index()
BeforeBG_TotalBalance_Ave = BeforeCamp_Daily[BeforeCamp_Daily['VALUE_DATE'] <
                 datetime.datetime(2022, 11, 8, 0, 0)].groupby(['ACCOUNT_NUMBER']).TOTAL_ACCOUNT_BALANCE.mean()
AfterBG_TotalBalance_Ave = BeforeCamp_Daily[BeforeCamp_Daily['VALUE_DATE'] >=
                 datetime.datetime(2022, 11, 8, 0, 0)].groupby(['ACCOUNT_NUMBER']).TOTAL_ACCOUNT_BALANCE.mean()
SaveBoost_MaxCount = SaveBoost_data_filter[['ACCOUNT_ID', 'Boosts_count']].groupby(['ACCOUNT_ID']).max()

# Use Loop to insert a blank dataframe.
for i in range(BeforeCamp_Change.shape[0]):
    AcNum = BeforeCamp_Change['Account_Number'][i]
    BeforeCamp_Change.loc[i, 'Source'] = BeforeCamp_Daily[BeforeCamp_Daily['ACCOUNT_NUMBER'] == AcNum].SOURCE.iloc[0]
    BeforeCamp_Change.loc[i, 'Days_Before'] = (
                datetime.datetime(2022, 11, 8) - BeforeCamp_Change.loc[i, 'Account_Open_Date']).days
    if AcNum in SaveBoost_MaxCount.index:
        BeforeCamp_Change.loc[i, 'Boost_Number'] = SaveBoost_MaxCount.loc[AcNum, 'Boosts_count']
        BeforeCamp_Change.loc[i, 'Treatment'] = 1
    else:
        BeforeCamp_Change.loc[i, 'Boost_Number'] = 0
        BeforeCamp_Change.loc[i, 'Treatment'] = 0

    BeforeCamp_Change.loc[i, 'Total_Balance_Change'] = AfterBG_TotalBalance_Ave[AcNum] - BeforeBG_TotalBalance_Ave[
        AcNum]

# Create a function to deal with categorical variables.
def _categoricals_process(df, cats): # Input target dataframe and target categorical columns.
    for col in cats:
        if len(X[data_cats].unique()) == 2:
            df[col], _ = pd.factorize(df[col])
        elif len(X[data_cats].unique()) > 2:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1).drop(labels=col,axis=1)
    return df
BeforeCamp_Change = _categoricals_process(BeforeCamp_Change, ['Source'])
BeforeCamp_Change = BeforeCamp_Change.drop(labels=['Account_Open_Date'], axis=1)

# Make sure the columns' type is correct.
BeforeCamp_Change[['Boost_Number',
                   'Days_Before',
                   'Treatment']] = BeforeCamp_Change[['Boost_Number',
                                                      'Days_Before',
                                                       'Treatment']].astype('int')
BeforeCamp_Change['Total_Balance_Change'] = BeforeCamp_Change['Total_Balance_Change'].astype('float')



# ----------------- Simple linear regression before further analysis ----------------------- #
import statsmodels.api as sm

reg_x = BeforeCamp_Change.drop(['Treatment', 'Account_Number','Total_Balance_Change'],axis=1).values
reg_x = sm.add_constant(reg_x)
reg_y = BeforeCamp_Change['Total_Balance_Change'].values

model = sm.OLS(reg_y,reg_x)
results = model.fit()
print("{0} + {1}*Source_A + {2}*Source_B + {3}*Source_C + {4}*Days_Before + {5}*Boost_Number".format(results.params[0],
                                                                                                     results.params[1],
                                                                                                     results.params[2],
                                                                                                     results.params[3],
                                                                                                     results.params[4],
                                                                                                     results.params[5]))
results.summary()



# ----------------- Average Treatment Effect by Causal Inference ----------------------- #
import dowhy

# Create customized causal graph
causal_graph = """digraph {
Days_Before;
Boost_Number;
Treatment;
Total_Balance_Change;
Source_A;
Source_B;
Source_C;
Treatment->{Total_Balance_Change};
Days_Before->{Boost_Number, Treatment, Total_Balance_Change};
Boost_Number->{Total_Balance_Change, Treatment};
Source_A->{Days_Before};
Source_B->{Days_Before};
Source_C->{Days_Before, Treatment, Total_Balance_Change};
}"""
model= dowhy.CausalModel(
        data = BeforeCamp_Change,
        graph=causal_graph.replace("\n", " "),
        treatment="Treatment",
        outcome="Total_Balance_Change")
model.view_model()

# Estimate the average treatment effect.
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.propensity_score_matching",target_units="ate")
print(estimate)

# Refute results
refute1_results = model.refute_estimate(identified_estimand, estimate,
        method_name="random_common_cause")
print(refute1_results)
res_placebo=model.refute_estimate(identified_estimand, estimate,
        method_name="placebo_treatment_refuter", placebo_type="permute")
print(res_placebo)
res_subset=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter", subset_fraction=0.8)
print(res_subset)



# ----------------- Individual Treatment Effect by Causal Inference ----------------------- #
from causalml.inference.meta import XGBTRegressor

# Data processing
treatment = BeforeCamp_Change['Treatment'].values
X = BeforeCamp_Change.drop(['Account_Number','Total_Balance_Change'],axis=1).values
y = BeforeCamp_Change['Total_Balance_Change'].values

# Estimate Individual and average treatment effect.
xg = XGBTRegressor(random_state=42)
te, lb, ub = xg.estimate_ate(X.drop('Treatment',axis=1).values, treatment, y)
print('Average Treatment Effect (XGBoost): {:.2f} (lower bound:{:.2f}, upper bound:{:.2f})'.
      format(te[0], lb[0], ub[0]))
cate_s = xg.fit_predict(X=X, treatment=treatment, y=y) #This returns CATE for each sample

# Extract those clients whose treatment effect < 0. Apply further analysis on it, like clusering.
from sklearn.cluster import KMeans

NegativeATE_Account = BeforeCamp_Change.loc[np.where(cate_s < 0)[0],]
initial_start = X.groupby(['Treatment']).mean().values
kmeans_result = KMeans(n_clusters=3, init=initial_start).fit(X.values)