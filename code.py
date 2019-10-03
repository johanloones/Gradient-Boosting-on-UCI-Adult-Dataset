# --------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Load Data
data_sal=pd.read_csv(path)
pd.set_option('display.max_column',None)
display(data_sal.head())


#Explore Data

# Count of Men & Women in Dataset
print('Count of Men & Women in Dataset',data_sal['sex'].value_counts())

# Average age of women
avg_women_age=data_sal.groupby(['sex'])['age'].mean()[0]
print('Average age of women :',avg_women_age)

# Percentage of German citizens
print('Percentage of German citizens :',data_sal['native-country'].value_counts()['Germany']/data_sal['native-country'].value_counts().sum(),'%')

# Mean & Standard Deviation age feature 
print('Mean people who recieve more than 50K per year',data_sal.groupby(data_sal[data_sal['salary'] == '>50K']['salary'])['age'].mean())
print('Standard Deviation people who recieve more than 50K per year',data_sal.groupby(data_sal[data_sal['salary'] == '>50K']['salary'])['age'].std())

print('Mean people who recieve less than equal to 50K per year',data_sal.groupby(data_sal[data_sal['salary'] == '<=50K']['salary'])['age'].mean())
print('Standard Deviation people who recieve less than equal to 50K per year',data_sal.groupby(data_sal[data_sal['salary'] == '<=50K']['salary'])['age'].std())

#Display the statistics of age for each gender of all the race
race_sex_age_groupby=data_sal.groupby(['race','sex'])['age']
display(race_sex_age_groupby.describe())
print('Maximum age of men of Amer-Indian-Eskimo race :',race_sex_age_groupby.max()['Amer-Indian-Eskimo']['Male'])

# Encoding the categorical features.

#Encode Salary column '>50k' we encode it to 1 else 0.
data_sal['salary']=data_sal['salary'].replace({'>50K':1,'<=50K':0})
display(data_sal['salary'].head())

#One hot encode the categorical features.
data_sal=pd.get_dummies(data_sal)
display(data_sal.head())


#Split features and target variable into X and y respectively.
X=data_sal.drop(columns='salary')
y=data_sal.salary

#Perform the following operation on dataset.
#Split the data X and y into Xtrain,Xtest,ytrain and ytest in the ratio 70:30
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=12)

#Further split the training data into train and validation in 80:20 ratio
Xtrain1,Xval,ytrain1,yval=train_test_split(Xtrain,ytrain,test_size=0.2,random_state=12)

#Then apply the base Decision Tree Classifier model and calculate the accuracy on validation data test data.
clf_dt=DecisionTreeClassifier()
clf_dt.fit(Xtrain1,ytrain1)
y_val_score=clf_dt.score(Xval,yval)
print('Validation Accuracy score',y_val_score)
y_test_score=clf_dt.score(Xtest,ytest)
print('Testing Accuracy score',y_test_score)

#Perform ensembling using the models Decision Tree Classifier and Logistic Regression, using a VotingClassifier  #parameter voting as soft and the calculate the accuracy.
clf_vt = VotingClassifier(estimators=[
    ('Decision Tree Classifier', DecisionTreeClassifier()), ('Logistic Regression', LogisticRegression())], voting='soft')
clf_vt.fit(Xtrain1, ytrain1)
y_clf_vt_val_score = clf_vt.score(Xval,yval)
print('Validation Voting Classifier Accuracy score',y_clf_vt_val_score)
y_clf_vt_test_score=clf_vt.score(Xtest,ytest)
print('Testing Voting Classifier Accuracy score',y_clf_vt_test_score)



#What is the Effect of adding more trees
#Perform the following Boosting task on models.

model_10=GradientBoostingClassifier(n_estimators=10,max_depth=6,random_state=12)
model_50=GradientBoostingClassifier(n_estimators=50,max_depth=6,random_state=12)
model_100=GradientBoostingClassifier(n_estimators=100,max_depth=6,random_state=12)

model_50.fit(Xtrain1,ytrain1)
model_100.fit(Xtrain1,ytrain1)
model_10.fit(Xtrain1,ytrain1)

#Calculate the accuracy on validation data and testing data.
y_model_10_val_score=model_10.score(Xval,yval)
y_model_10_test_score=model_10.score(Xtest,ytest)
print('Validation model_10 :',y_model_10_val_score,'Testing model_10 :',y_model_10_test_score)

y_model_50_val_score=model_50.score(Xval,yval)
y_model_50_test_score=model_50.score(Xtest,ytest)
print('Validation model_50 :',y_model_50_val_score,'Testing model_50 :',y_model_50_test_score)

y_model_100_val_score=model_100.score(Xval,yval)
y_model_100_test_score=model_100.score(Xtest,ytest)
print('Validation model_100 :',y_model_100_val_score,'Testing model_100 :',y_model_100_test_score)



#Based on the best gradient boosting classifier , plot a bar plot of the model's top 10 features with  feature importance score
feat_imp=pd.DataFrame({'importance':model_100.feature_importances_},index=Xtrain.columns)
feat_imp.sort_values(by='importance',inplace=True,ascending=False)
feat_imp[:10].sort_values(by='importance').plot.barh()




#Steps to follow:
#Step 1: Calculate the classification error for model on the training data 
train_err_10=1-model_10.score(Xtrain1,ytrain1)
train_err_50=1-model_50.score(Xtrain1,ytrain1)
train_err_100=1-model_100.score(Xtrain1,ytrain1)

#Step 2: Store the training errors into a list ( training_errors) 
training_errors= [train_err_10, train_err_50, train_err_100]

#Step 3: Calculate the classification error of each model on the validation data .
validation_err_10=1-model_10.score(Xval,yval)
validation_err_50=1-model_50.score(Xval,yval)
validation_err_100=1-model_100.score(Xval,yval)

#Step 4: Store the validation classification error into a list ( validation_errors) 
validation_errors= [validation_err_10, validation_err_50,validation_err_100]

#Step 5: Calculate the classification error of each model on the test data .
testing_err_10=1-model_10.score(Xtest,ytest)
testing_err_50=1-model_50.score(Xtest,ytest)
testing_err_100=1-model_100.score(Xtest,ytest)

#Step 6: Store the testing classification error into a list ( testing_errors) 
testing_errors = [testing_err_10, testing_err_50,testing_err_100]



#Plot the training and testing error vs. number of trees
n_trees=[10,50,100]
plt.plot(n_trees,training_errors,label='Training Error')
plt.plot(n_trees,testing_errors,label='Testing Error')
plt.legend(['Training Error','Testing Error'])
plt.title('Training and Testing Error vs. Number of Trees')


