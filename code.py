# --------------
# Importing Necessary libraries
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



# Load the train data stored in path variable
data_train = pd.read_csv(path)
data_train.tail()

# Load the test data stored in path1 variable
data_test = pd.read_csv(path1)
data_test.tail()

# necessary to remove rows with incorrect labels in test dataset
data_test = data_test[(data_test['Target'] == ' >50K.') | (data_test['Target']==' <=50K.')]

# encode target variable as integer
data_train.loc[data_train['Target']==' <=50K', 'Target'] = 0
data_train.loc[data_train['Target']==' >50K', 'Target'] = 1

data_test.loc[data_test['Target']==' <=50K.', 'Target'] = 0
data_test.loc[data_test['Target']==' >50K.', 'Target'] = 1


# Plot the distribution of each feature
fig = plt.figure(figsize=(25, 15))
cols = 5
rows = np.ceil(float(data_train.shape[1]) / cols)
for i, column in enumerate(data_train.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if data_train.dtypes[column] == np.object:
        data_train[column].value_counts().plot(kind="bar", axes=ax)
    else:
        data_train[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)


# convert the data type of Age column in the test data to int type
data_test['Age'] = data_test['Age'].astype(int)

# cast all float features to int type to keep types consistent between our train and test data
data_test['fnlwgt'] = data_test['fnlwgt'].astype(int)
data_test['Education_Num'] = data_test['Education_Num'].astype(int)
data_test['Capital_Gain'] = data_test['Capital_Gain'].astype(int)
data_test['Capital_Loss'] = data_test['Capital_Loss'].astype(int)
data_test['Hours_per_week'] = data_test['Hours_per_week'].astype(int)
 
# choose categorical and continuous features from data and print them
categorical_columns = [c for c in data_train.columns 
                       if data_train[c].dtype.name == 'object']
numerical_columns = [c for c in data_train.columns 
                     if data_train[c].dtype.name != 'object']

print('categorical_columns:', categorical_columns)
print('numerical_columns:', numerical_columns)

# fill missing data for catgorical columns
for c in categorical_columns:
    data_train[c].fillna(data_train[c].mode()[0], inplace=True)
    data_test[c].fillna(data_train[c].mode()[0], inplace=True)

# fill missing data for numerical columns   
for c in numerical_columns:
    data_train[c].fillna(data_train[c].median(), inplace=True)
    data_test[c].fillna(data_train[c].median(), inplace=True)

# Label encoding categorical features


le = LabelEncoder()

for x in categorical_columns:
    data_train[x] = le.fit_transform(data_train[x])
    data_test[x] = le.transform(data_test[x]) 

# one hot encode Categorical features
data_train = pd.concat([data_train[numerical_columns],
    pd.get_dummies(data_train[categorical_columns])], axis=1)

data_test = pd.concat([data_test[numerical_columns],
    pd.get_dummies(data_test[categorical_columns])], axis=1)

# Alternate way with a single line of code to encode data with categorical data without complicating if you want to use.
#pd.get_dummies(data=data_train, columns = categorical_columns).shape


# Split train and test data into X_train ,y_train,X_test and y_test data
X_train = data_train.drop(['Target'], axis=1)
y_train = data_train['Target']

X_test = data_test.drop(['Target'], axis=1)
y_test = data_test['Target']

# train a decision tree model then predict our test data and compute the accuracy
tree = DecisionTreeClassifier(max_depth=3, random_state=17)
tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test) 
print("Decision tree accuracy: ",accuracy_score(y_test,tree_predictions))

# Decision tree with parameter tuning
tree_params = {'max_depth': range(2, 11)}
locally_best_tree = GridSearchCV(DecisionTreeClassifier(random_state=17),
                                 tree_params, cv=5)                  
locally_best_tree.fit(X_train, y_train)

# Print out optimal maximum depth(i.e. best_params_ attribute of GridSearchCV) and best_score_
print("Best params:", locally_best_tree.best_params_)
print("Best cross validaton score", locally_best_tree.best_score_)

#train a decision tree model with best parameter then predict our test data and compute the accuracy
tuned_tree = DecisionTreeClassifier(max_depth=9, random_state=17)
tuned_tree.fit(X_train, y_train)
tuned_tree_predictions = tuned_tree.predict(X_test)
print("Decision tree Accuracy after tuning: ",accuracy_score(y_test, tuned_tree_predictions))



