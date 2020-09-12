
# %%
'''-------------------- Dependencies --------------------'''
# region Imports
# Import Dependencies
import datetime
import random
import time
import math
import warnings
from catboost import CatBoostClassifier, Pool, cv
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.model_selection import train_test_split
import catboost
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
import seaborn as sns
import missingno
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
%matplotlib inline

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}
matplotlib.rc('font', **font)
sns.set(style="ticks", context="talk")
plt.style.use('seaborn-whitegrid')
# plt.style.use("dark_background")


# Let's be rebels and ignore warnings for now
warnings.filterwarnings('ignore')  # %%


print('imports successful')

# endregion Imports


# %%
'''-------------------- Data --------------------'''
# region Read Data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
gender_submission = pd.read_csv('data/gender_submission.csv')
print('data loaded successfully')
# endregion Read Data


# %%
'''-------------------- Describe --------------------'''
train.describe()



# 51603605 
# 5750
# 340312
# %%
'''-------------------- Missing Data --------------------'''
# region Missing_data_df


def find_missing_vals(df):
    missing_df = pd.DataFrame()
    for col in df.columns:
        total_value_count = df[col].value_counts().sum()
        missing = len(df) - total_value_count
        perc_missing = '{:.2f}%'.format(missing*100 / len(df))
        row = pd.DataFrame({
            'column': [col],
            'missing_entries': [missing],
            'perc_missing': [perc_missing],
        })
        missing_df = missing_df.append(row, sort=False)

    return missing_df.reset_index(drop=True)


def print_missing_count(var_name):
    amount_missing = missing_df.loc[missing_df.column ==
                                    var_name, 'missing_entries'].values[0]
    perc_missing = missing_df.loc[missing_df.column ==
                                  var_name, 'perc_missing'].values[0]
    print(
        f'Amount of "{var_name}" Data Missing: {amount_missing} ({perc_missing})')


missing_df = find_missing_vals(train)
print('missing values per column\n'.title(), missing_df)
matrix_view = missingno.matrix(train, figsize=(10, 3))
matrix_view
# endregion Missing_data_df

# %%
# region Handling Data
# Binned
df_bin = pd.DataFrame()
# Continuous
df_con = pd.DataFrame()
# endregion Handling Data


# %%
'''-------------------- Survived Vs. Perished --------------------'''
# region Survived
df_bin['Survived'] = train.Survived
df_con['Survived'] = train.Survived
fig = plt.figure(figsize=(20, 5))
sns.countplot(y="Survived", data=train)
print(train.Survived.value_counts())
# endregion Survived


# %%
'''-------------------- Ticket Class (Pclass) --------------------'''
# region Pclass
variable = 'Pclass'
df_bin[variable] = train[variable]
df_con[variable] = train[variable]
print_missing_count(variable)
sns.distplot(train.Pclass)
# endregion Pclass


# %%
'''-------------------- Passenger Name --------------------'''
# region Name Field
variable = 'Name'
# df_bin[variable] = train[variable]
# df_con[variable] = train[variable]
print_missing_count(variable)
unique_names = train[variable].unique().tolist()
no_repeats = bool(len(unique_names) == len(train))
msg = f'No Repeat Names: {no_repeats}'
print(msg)
# endregion Name Field


# %%

'''-------------------- Passenger Sex --------------------'''
# region Sex
variable = 'Sex'
df_bin[variable] = np.where(train[variable] == 'female', 1, 0)

df_con[variable] = train[variable]
print_missing_count(variable)
fig = plt.figure(figsize=(20, 5))
sns.countplot(y=variable, data=train)
print(train[variable].value_counts())
# endregion Sex


# %%
'''-------------------- Survived vs. Sex --------------------'''

# region Surv vs Sex
fem_df = df_bin[df_bin['Sex'] == 1]
fem_total = len(fem_df)
fem_survived = len(fem_df[fem_df['Survived'] == 1])
perc_fem_surv = '{:.2f}%'.format(fem_survived*100 / fem_total)
men_df = df_bin[df_bin['Sex'] == 0]
men_total = len(men_df)
men_survived = len(men_df[men_df['Survived'] == 1])
perc_men_surv = '{:.2f}%'.format(men_survived*100 / men_total)

print(
    f'# of Female Survivors : {fem_survived} / {fem_total} ({perc_fem_surv}) \n'
    f'# of Male Survivors : {men_survived} / {men_total} ({perc_men_surv}) \n'
)
fig = plt.figure(figsize=(10, 10))
sns.distplot(df_bin.loc[df_bin['Survived'] == 1]
             ['Sex'], kde_kws={'label': 'Survived'})
sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'],
             kde_kws={'label': 'Did not Survive'})
# endregion Surv vs Sex

# %%
'''-------------------- Age --------------------'''
variable = 'Age'
print_missing_count(variable)

# %%
'''-------------------- Subplotter Func --------------------'''


def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):
    """
    Function to plot counts and distributions of a label variable and 
    target variable side by side.
    ::param_data:: = target dataframe
    ::param_bin_df:: = binned dataframe for countplot
    ::param_label_column:: = binary labelled column
    ::param_target_column:: = column you want to view counts and distributions
    ::param_figsize:: = size of figure (width, height)
    ::param_use_bin_df:: = whether or not to use the bin_df, default False
    """
    if use_bin_df:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=bin_df)
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column],
                     kde_kws={"label": "Survived"})
        sns.distplot(data.loc[data[label_column] == 0][target_column],
                     kde_kws={"label": "Did not survive"})
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=data)
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column],
                     kde_kws={"label": "Survived"})
        sns.distplot(data.loc[data[label_column] == 0][target_column],
                     kde_kws={"label": "Did not survive"})


# %%
'''-------------------- Sibling / Spouse --------------------'''
# region SibSp
variable = 'SibSp'
df_bin[variable] = train[variable]
df_con[variable] = train[variable]
print_missing_count(variable)
print(train[variable].value_counts())
plot_count_dist(
    data=train,
    bin_df=df_bin,
    label_column='Survived',
    target_column=variable,
    figsize=(20, 10)
)
# endregion SibSp

# %%
'''-------------------- Parents/Children --------------------'''
# Specifically, the number of parents/children the passenger has on board

# region Parent/Child
variable = 'Parch'
df_bin[variable] = train[variable]
df_con[variable] = train[variable]
print_missing_count(variable)
print(train[variable].value_counts())
plot_count_dist(
    data=train,
    bin_df=df_bin,
    label_column='Survived',
    target_column=variable,
    figsize=(20, 10)
)
# endregion Parent/Child


# %%

'''-------------------- Ticket Text --------------------'''
# region Ticket
variable = 'Ticket'
print_missing_count(variable)
unique_tickets = len(train[variable].unique())
print(f'\nnum of unique tickets: {unique_tickets}\n'.title())

val_count = train[variable].value_counts()
u_val_count = val_count.unique()
for count in u_val_count:
    print(
        f'{len(val_count[val_count == count])} tickets values used {count} times')
# endregion Ticket


# %%
'''-------------------- Fare --------------------'''

# region Fare
variable = 'Fare'
# Bin to Quintiles
df_bin[variable] = pd.qcut(train[variable], q=5)  # Discretised
df_con[variable] = train[variable]
print_missing_count(variable)
print(
    f'Num of Unique Fare Costs: {len(train[variable].unique())}'
)
print('\ndistribution:\n')
df_bin[variable].value_counts()


plot_count_dist(
    data=train,
    bin_df=df_bin,
    label_column='Survived',
    target_column=variable,
    figsize=(20, 10),
    use_bin_df=True,
)

# endregion Fare

# %%
'''-------------------- Cabin --------------------'''

# region Cabin
variable = 'Cabin'
print_missing_count(variable)
total = len(train[variable].dropna())
unique = len(train[variable].unique())
print(
    f'{unique}  unique values out of {total} \n'
    'Too many missing to be useful'
)
# endregion Cabin

# %%
'''-------------------- Embarked --------------------'''
# region Embarked
variable = 'Embarked'
df_bin[variable] = train[variable]
df_con[variable] = train[variable]
# Clean NaN
df_bin = df_bin.dropna(subset=[variable])
df_con = df_con.dropna(subset=[variable])
print_missing_count(variable)
total = len(train[variable].dropna())
unique = len(train[variable].dropna().unique())
print(
    '\n',
    f'\n{unique}  unique values out of {total} \n'
    '\n',
    f'breakdown:\n {train[variable].value_counts()}',
)
sns.countplot(x=variable, data=train)
# endregion Embarked


# %%
'''-------------------- Subset Compare and Create --------------------'''
# region Compare Subset
print('columns not included in subset frame:'.title())
for col in train.columns:
    if col not in df_bin.columns:
        print(f'>>> {col}')

one_hot_cols = ['Embarked', 'Sex', 'Pclass']

concats = [df_con]
for col in one_hot_cols:
    ohdf = pd.get_dummies(df_con[col],
                          prefix=col.lower())
    concats.append(ohdf)

df_con_enc = pd.concat(concats, axis=1)
df_con_enc = df_con_enc.drop(one_hot_cols, axis=1)
print('\nDummified Data:')
# df_con_enc.head(10)
X_train = df_con_enc.drop('Survived', axis=1)
y_train = df_con_enc['Survived']
print(
    'X_train set has:\n',
    '{} rows, {} columns'.format(*X_train.shape)
)
X_train.head(5)
# endregion Compare Subset


# %%

'''-------------------- Algo Func --------------------'''

# Function that runs the requested algorithm and returns the accuracy metrics


def fit_ml_algos(algos, X_train, y_train, cv):

    algo_df = pd.DataFrame()
    for algo in algos.keys():
        algo_name = algo
        algo = algos[algo]()

        start_time = time.time()
        # One Pass
        model = algo.fit(X_train, y_train)
        acc = round(model.score(X_train, y_train) * 100, 2)

        # Cross Validation
        train_pred = model_selection.cross_val_predict(algo,
                                                    X_train,
                                                    y_train,
                                                    cv=cv,
                                                    n_jobs=-1)
        # Cross-validation accuracy metric
        acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

        log_time = (time.time() - start_time)
        elapsed = datetime.timedelta(seconds=log_time)
        row = pd.DataFrame(
            {
                'Algo': algo_name,
                'Accuracy': acc,
                'Accuracy_CV_10-F': acc_cv,
                'Time_Elapsed':elapsed
            },
        index=[0]
        )
        algo_df = algo_df.append(row, sort=False)

    return algo_df.set_index('Algo', drop=True)



# %%
'''-------------------- SkitLearn Algos --------------------'''
# region SkitLearn Algos
algos = {
    'LogisticRegression':LogisticRegression,
    'KNeighborsClassifier':KNeighborsClassifier,
    'GaussianNB':GaussianNB,
    'LinearSVC':LinearSVC,
    'SGDClassifier':SGDClassifier,
    'DecisionTreeClassifier':DecisionTreeClassifier,
    'GradientBoostingClassifier':GradientBoostingClassifier,
}
algos_df = fit_ml_algos(algos, X_train, y_train, 10)
algos_df
# endregion SkitLearn Algos

# %%
'''-------------------- Catboost --------------------'''
# region Catboost
# Catboost
# Categorical Features cannot be floats
cat_features = np.where(X_train.dtypes != np.float)[0]
cat_features

train_pool = Pool(X_train, y_train, cat_features)

catboost_model = CatBoostClassifier(iterations=1000,
                                    custom_loss=['Accuracy'],
                                    loss_function='Logloss')

# Fit CatBoost model
catboost_model.fit(train_pool)
                #    plot=Flaws)

# CatBoost accuracy
acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)
# endregion Catboost


# %%

import seaborn as sns
sns.set_theme()

penguins = sns.load_dataset("penguins")

penguins

# %%

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=train, 
    kind="bar",
    x="Survived", 
    y="Fare", 
    hue="Sex",
    # ci="sd", 
    palette="dark", 
    alpha=.6,
    height=6
)
g.despine(left=True)
g.set_axis_labels("", "Body mass (g)")
g.legend.set_title("")
# %%
