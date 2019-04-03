import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Loan Prediction.csv")

#Check the datatypes of each column
df.dtypes

## UNIVARIATE ANALYSIS
#Categorical Data
df['Loan_Status'].value_counts(normalize = True).plot.bar() #70% were approved
df['Gender'].value_counts(normalize = True).plot.bar() #80% are male
df['Married'].value_counts(normalize = True).plot.bar() #65% are married
df['Self_Employed'].value_counts(normalize = True).plot.bar() #15% are self employed
df['Credit_History'].value_counts(normalize = True).plot.bar() #85% have credit history

#Ordinal Data
df['Education'].value_counts(normalize = True).plot.bar() #80% are graduates
df['Dependents'].value_counts(normalize = True).plot.bar() #57% have no dependents, 15% have 1 or 2 dependents
df['Property_Area'].value_counts(normalize = True).plot.bar() #40% SemiUrban, 35% Urban, 30% Rural

#Metric Data
sns.distplot(df["ApplicantIncome"]) #Highly skewed distribution
df["ApplicantIncome"].plot.box() #Identifying the outliers
df.boxplot(column = 'ApplicantIncome', by = 'Education')

sns.distplot(df["CoapplicantIncome"]) #Highly skewed distribution
df["CoapplicantIncome"].plot.box() #Identifying the outliers


#BIVARIATE ANALYSIS
Gender=pd.crosstab(df['Gender'],df['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) #Loan_Status does not much depend on the Gender

Married=pd.crosstab(df['Married'],df['Loan_Status']) 
Dependents=pd.crosstab(df['Dependents'],df['Loan_Status']) 
Education=pd.crosstab(df['Education'],df['Loan_Status']) 
Self_Employed=pd.crosstab(df['Self_Employed'],df['Loan_Status']) 
Credit_History=pd.crosstab(df['Credit_History'],df['Loan_Status'])
Property_Area=pd.crosstab(df['Property_Area'],df['Loan_Status'])

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) #It is more likely for a married person to get their loan approved
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) #No significant inference from this variable
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) #Graduate has high chances of getting their loan approved
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) #No significant inference from this variable
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) #If the customer has a positive credit history, it is very likely for him to get his loan approved
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) #Semi-urban has the maximum no of loans approved

#Correlation
matrix = df.corr()
sns.heatmap(matrix, cmap="BuPu") 
#LoanAmount and Applicant Income are highly correlated
#LoanAmount and Coapplicant Income are highly correlated
                                 
## MISSING DATA
#Dependents replace
df["Dependents"].replace("3+", 3, inplace = True)
df["Dependents"] = df["Dependents"].astype(float)

#Loan Status replace
df["Loan_Status"].replace("Y", 1, inplace = True)
df["Loan_Status"].replace("N", 0, inplace = True)

#Missing values
df.isnull().sum()
df["Gender"].fillna(df["Gender"].mode()[0], inplace = True)
df["Married"].fillna(df["Married"].mode()[0], inplace = True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace = True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace = True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace = True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace = True)
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace = True)

#Outlier treatment
df["LoanAmountLog"] = np.log(df["LoanAmount"])
sns.distplot(df["LoanAmountLog"], bins=20)

#Loan_ID is redundant
df = df.drop("Loan_ID", axis = 1)

#Feature Engineering
df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
sns.distplot(df["Total_Income"])
df["TotalIncomeLog"] = np.log(df["Total_Income"])

df["EMI"] = df["LoanAmount"]/df["Loan_Amount_Term"]
sns.distplot(df["EMI"])
df["EMILog"] = np.log(df["EMI"])

df['Balance_Income']=df['Total_Income']-(df['EMI']*1000)
sns.distplot(df['Balance_Income'])
df["Balance_IncomeLog"] = np.log(df["Balance_Income"])
df["Balance_IncomeLog"].fillna(df["Balance_IncomeLog"].median(), inplace = True)

#Dropping the variables that were used to derive new variables
df = df.drop(["EMI", "Balance_Income", "Total_Income", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"], axis = 1)

#Test sets
X = df.drop("Loan_Status", axis = 1)
Y = df.Loan_Status

#Dummy Variables
X = pd.get_dummies(X)

#Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
cm = confusion_matrix(Y_test, logreg.predict(X_test))
accuracyScore = accuracy_score(Y_test, logreg.predict(X_test)) #82% accuracy





















