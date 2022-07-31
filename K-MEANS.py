# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 12:37:24 2022

@author: Rakesh
"""
##################################Problem 1########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###loading dataset ###
airline =pd.read_excel('D:/DATA SCIENCE ASSIGNMENT/Datasets_Kmeans/EastWestAirlines.xlsx', sheet_name='data')

#lets drop ID# in columns as it doesnt contribute in dataset#
airline.drop(['ID#'], axis=1 , inplace = True)

#checking NA or Null value
airline.isna().sum()
airline.isnull().sum()

#checking for duplicate value##

dup1 = airline.duplicated()
sum(dup1)

airline_new = airline.drop_duplicates()

airline.columns

##Plotting Boxplot univariate EDA##

sns.boxplot(airline_new.Balance);plt.title('Boxplot');plt.show()
sns.boxplot(airline_new.Qual_miles);plt.title('Boxplot');plt.show()
sns.boxplot(airline_new.cc1_miles);plt.title('Boxplot');plt.show()
sns.boxplot(airline_new.cc2_miles);plt.title('Boxplot');plt.show()
sns.boxplot(airline_new.cc3_miles);plt.title('Boxplot');plt.show()
sns.boxplot(airline_new.Bonus_miles);plt.title('Boxplot');plt.show()
sns.boxplot(airline_new.Bonus_trans);plt.title('Boxplot');plt.show()
sns.boxplot(airline_new.Flight_miles_12mo);plt.title('Boxplot');plt.show()
sns.boxplot(airline_new.Flight_trans_12);plt.title('Boxplot');plt.show()
sns.boxplot(airline_new.Days_since_enroll);plt.title('Boxplot');plt.show()

#Scatterplt#

plt.scatter(airline_new["Balance"] , airline_new["Bonus_miles"])
plt.scatter(airline_new["Balance"] , airline_new["Bonus_trans"])
plt.scatter(airline_new["Flight_miles_12mo"] , airline_new["Flight_trans_12"])
plt.scatter(airline_new["Bonus_miles"] , airline_new["Bonus_trans"])

#removing outlier##
IQR = airline_new['Balance'].quantile(0.75)-airline_new['Balance'].quantile(0.25)
lower_limit_balance= airline_new['Balance'].quantile(0.25)-(IQR*1.5)
upper_limit_balance= airline_new['Balance'].quantile(0.75)+(IQR*1.5)
airline_new['Balance']=pd.DataFrame(np.where(airline_new['Balance']>upper_limit_balance,upper_limit_balance,
                                         np.where(airline_new['Balance']<lower_limit_balance,lower_limit_balance,airline_new['Balance'])))
sns.boxplot(airline_new.Balance);plt.title('Boxplot');plt.show()

IQR = airline_new['Bonus_miles'].quantile(0.75)-airline_new['Bonus_miles'].quantile(0.25)
lower_limit_Bonus_miles= airline_new['Bonus_miles'].quantile(0.25)-(IQR*1.5)
upper_limit_Bonus_miles= airline_new['Bonus_miles'].quantile(0.75)+(IQR*1.5)
airline_new['Bonus_miles']=pd.DataFrame(np.where(airline_new['Bonus_miles']>upper_limit_Bonus_miles,upper_limit_Bonus_miles,
                                         np.where(airline_new['Bonus_miles']<lower_limit_Bonus_miles,lower_limit_Bonus_miles,airline_new['Bonus_miles'])))
sns.boxplot(airline_new.Bonus_miles);plt.title('Boxplot');plt.show()

IQR = airline_new['Bonus_trans'].quantile(0.75)-airline_new['Bonus_trans'].quantile(0.25)
lower_limit_Bonus_trans= airline_new['Bonus_trans'].quantile(0.25)-(IQR*1.5)
upper_limit_Bonus_trans= airline_new['Bonus_trans'].quantile(0.75)+(IQR*1.5)
airline_new['Bonus_trans']=pd.DataFrame(np.where(airline_new['Bonus_trans']>upper_limit_Bonus_trans,upper_limit_Bonus_trans,
                                         np.where(airline_new['Bonus_trans']<lower_limit_Bonus_trans,lower_limit_Bonus_trans,airline_new['Bonus_trans'])))
sns.boxplot(airline_new.Bonus_trans);plt.title('Boxplot');plt.show()

IQR = airline_new['Flight_miles_12mo'].quantile(0.75)-airline_new['Flight_miles_12mo'].quantile(0.25)
lower_limit_Flight_miles_12mo= airline_new['Flight_miles_12mo'].quantile(0.25)-(IQR*1.5)
upper_limit_Flight_miles_12mo= airline_new['Flight_miles_12mo'].quantile(0.75)+(IQR*1.5)
airline_new['Flight_miles_12mo']=pd.DataFrame(np.where(airline_new['Flight_miles_12mo']>upper_limit_Flight_miles_12mo,upper_limit_Flight_miles_12mo,
                                         np.where(airline_new['Flight_miles_12mo']<lower_limit_Flight_miles_12mo,lower_limit_Flight_miles_12mo,airline_new['Flight_miles_12mo'])))
sns.boxplot(airline_new.Flight_miles_12mo);plt.title('Boxplot');plt.show()

IQR = airline_new['Flight_trans_12'].quantile(0.75)-airline_new['Flight_trans_12'].quantile(0.25)
lower_limit_Flight_trans_12= airline_new['Flight_trans_12'].quantile(0.25)-(IQR*1.5)
upper_limit_Flight_trans_12= airline_new['Flight_trans_12'].quantile(0.75)+(IQR*1.5)
airline_new['Flight_trans_12']=pd.DataFrame(np.where(airline_new['Flight_trans_12']>upper_limit_Flight_trans_12,upper_limit_Flight_trans_12,
                                         np.where(airline_new['Flight_trans_12']<lower_limit_Flight_trans_12,lower_limit_Flight_trans_12,airline_new['Flight_trans_12'])))
sns.boxplot(airline_new.Flight_trans_12);plt.title('Boxplot');plt.show()

IQR = airline_new['Days_since_enroll'].quantile(0.75)-airline_new['Days_since_enroll'].quantile(0.25)
lower_limit_Days_since_enroll= airline_new['Days_since_enroll'].quantile(0.25)-(IQR*1.5)
upper_limit_Days_since_enroll= airline_new['Days_since_enroll'].quantile(0.75)+(IQR*1.5)
airline_new['Days_since_enroll']=pd.DataFrame(np.where(airline_new['Days_since_enroll']>upper_limit_Days_since_enroll,upper_limit_Days_since_enroll,
                                         np.where(airline_new['Days_since_enroll']<lower_limit_Days_since_enroll,lower_limit_Days_since_enroll,airline_new['Days_since_enroll'])))
sns.boxplot(airline_new.Days_since_enroll);plt.title('Boxplot');plt.show()

##defining norm##
def norm_fun(i):
    x= (i-i.min()/ i.max()-i.min())
    return(x)

##normalizing data#

airline_norm= norm_fun(airline_new.iloc[:,0:])

##checking Nan or null value again
airline_norm.isnull().sum()
airline_norm.isna().sum()

#removing null or na value##

airline_norm1 =airline_norm.replace(to_replace=np.nan, value=0)
airline_norm1.isna().sum()
airline_norm1.isnull().sum()

from sklearn.cluster import KMeans

###K mean model building##
TWSS = []
k= list(range(2,9)) ##getting values for cluster 2 to 9
for i in k:
    kmeans= KMeans(n_clusters=1)
    kmeans.fit(airline_norm1)
    TWSS.append(kmeans.inertia_)

TWSS

##Plotting Elbow curve for chosing best cluster#
plt.plot(k, TWSS , 'ro-');plt.xlabel("No_of_clusters");plt.ylabel('total_within_SS')

##model building#

model_airlines= KMeans(n_clusters =i)
model_airlines.fit(airline_norm1)

##getting label of cluster assigned to each row###

model_airlines.labels_
cluster_airlines = pd.Series(model_airlines.labels_) ## converting numpy into pandas##
airline['cluster']= cluster_airlines

airlines = airline.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
airlines.head()

airlines.iloc[:,1:11].groupby(airlines.cluster).mean()

airlines.to_csv('new_airlines', encoding ='utf-8')

import os
os.getcwd()

##############################problems 2################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##loading dataset#
crime = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Kmeans/crime_data.csv')
crime.columns
##checking NA or Null value ##
crime.isnull().sum()
crime.isna().sum()

##checking duplicate value#
dup1 = crime.duplicated()
sum(dup1)

##plotting boxplot##univariate EDA##
sns.boxplot(crime['Murder']);plt.title('Boxplot');plt.show()
sns.boxplot(crime['Assault']);plt.title("Boxplot");plt.show()
sns.boxplot(crime['UrbanPop']);plt.title('Boxplot');plt.show()
sns.boxplot(crime['Rape']);plt.title('Boxplot');plt.show()

##acatterplot Bivariate EDA##
plt.scatter(crime['Murder'], crime['Assault'])
plt.scatter(crime['UrbanPop'], crime['Rape'])
plt.scatter(crime['Murder'], crime['Rape'])

#removing outlier ##
IQR = crime['Rape'].quantile(0.75)-crime['Rape'].quantile(0.25)
lower_limit_Rape= crime['Rape'].quantile(0.25)-(IQR*1.5)
upper_limit_Rape= crime['Rape'].quantile(0.75)+(IQR*1.5)
crime['Rape']=pd.DataFrame(np.where(crime['Rape']>upper_limit_Rape,upper_limit_Rape,
                                         np.where(crime['Rape']<lower_limit_Rape,lower_limit_Rape,crime['Rape'])))
sns.boxplot(crime.Rape);plt.title('Boxplot');plt.show()

##scaling data using min max method
def norm_fun(i):
    x= (i-i.min()/i.max()-i.min())
    return (x)

crime_data_norm= norm_fun(crime.iloc[: ,1:])
#letss check Null and Na value one more tims#

crime_data_norm.isnull().sum()
crime_data_norm.isna().sum()

from sklearn.cluster import KMeans

#model Building Kmeans#
TWSS= []
k= list(range(2,9))

for i in k:
    kmeans= KMeans(n_clusters=i)
    kmeans.fit(crime_data_norm)
    TWSS.append(kmeans.inertia_)
TWSS

##Elbow curve plotting###
plt.plot(k, TWSS , 'ro-');plt.xlabel("No_of_clusters");plt.ylabel('total_within_SS')

##model building#

model_crime= KMeans(n_clusters =3)
model_crime.fit(crime_data_norm)

##assigning labels of clusters to each row##
model_crime.labels_ 
cluster_crime= pd.Series(model_crime.labels_)
crime['cluster']=cluster_crime

#indexing column##

crime= crime.iloc[:,[5,0,1,2,3,4]]

crime.head()

crime.iloc[: , 1:].groupby(crime.cluster).mean()

##exporting ###

crime.to_csv('crime_Kmean.csv' , encoding= 'utf-8')

import os
os.getcwd()


#################################Problem 3############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##uploading dataset ###

insurance= pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Kmeans/Insurance Dataset.csv')

##checking duplicates in dataset#
dup1 = insurance.duplicated()
sum(dup1)

###checking for Na or null value##
insurance.isna().sum()
insurance.isnull().sum()
insurance.columns
##Boxplotting bivariate EDA####
sns.boxplot(insurance['Premiums Paid']);plt.title('Boxplot');plt.show()
sns.boxplot(insurance['Age']);plt.title('Boxplot');plt.shpw()
sns.boxplot(insurance['Days to Renew']);plt.title('Boxplot');plt.shpw()
sns.boxplot(insurance['Claims made']);plt.title('Boxplot');plt.shpw()
sns.boxplot(insurance['Income']);plt.title('Boxplot');plt.shpw()

#scatterplt ##bivariate EDA##
plt.scatter(insurance['Premiums Paid'],insurance['Age'])
plt.scatter(insurance['Days to Renew'],insurance['Claims made'])
plt.scatter(insurance['Income'],insurance['Premiums Paid'])

#Removing outlier using IQR#

IQR = insurance['Premiums Paid'].quantile(0.75)-insurance['Premiums Paid'].quantile(0.25)
lower_limit_Premiums_Paid= insurance['Premiums Paid'].quantile(0.25)-(IQR*1.5)
upper_limit_Premiums_Paid= insurance['Premiums Paid'].quantile(0.75)+(IQR*1.5)
insurance['Premiums Paid']=pd.DataFrame(np.where(insurance['Premiums Paid']>upper_limit_Premiums_Paid,upper_limit_Premiums_Paid,
                                         np.where(insurance['Premiums Paid']<lower_limit_Premiums_Paid,lower_limit_Premiums_Paid,insurance['Premiums Paid'])))
sns.boxplot(insurance['Premiums Paid']);plt.title('Boxplot');plt.show()

IQR = insurance['Claims made'].quantile(0.75)-insurance['Claims made'].quantile(0.25)
lower_limit_Claims_made= insurance['Claims made'].quantile(0.25)-(IQR*1.5)
upper_limit_Claims_made= insurance['Claims made'].quantile(0.75)+(IQR*1.5)
insurance['Claims made']=pd.DataFrame(np.where(insurance['Claims made']>upper_limit_Claims_made,upper_limit_Claims_made,
                                         np.where(insurance['Claims made']<lower_limit_Claims_made,lower_limit_Claims_made,insurance['Claims made'])))
sns.boxplot(insurance['Claims made']);plt.title('Boxplot');plt.show()

#scaling method using min max method#

def norm_fun(i):
    x = (i-i.min()/i.max()-i.min())
    return(x)

#normalization##
insurance_norm= norm_fun(insurance)

str(insurance_norm)
from sklearn.cluster import KMeans
##K means model buiilding ##

TWSS = []
k = list(range(2,9))
for i in k:
    kmeans= KMeans(n_clusters= i)
    kmeans.fit(insurance_norm)
    TWSS.append(kmeans.inertia_)
TWSS    
    
plt.plot(k,TWSS,'ro-'); plt.xlabel('No_of_clusters');plt.ylabel('total_within_SS')

#model building##
model_insurance = KMeans(n_clusters= 3)
model_insurance.fit(insurance_norm)

model_insurance.labels_ ##assigning labels to each cluster#
cluster_insurance=pd.Series(model_insurance.labels_) ##converting numpy into pandas ##
insurance['cluster'] = cluster_insurance

insurance= insurance.iloc[:,[5,0,1,2,3,4]]

insurance.head()

insurance.iloc[: , 1:].groupby(insurance.cluster).mean()

insurance.to_csv("Kmeans_insurance.csv", encoding = "utf-8")
import os
os.getcwd()

#######################################PROBLEM 4######################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
###loading data set##

telco = pd.read_excel('D:/DATA SCIENCE ASSIGNMENT/Datasets_Kmeans/Telco_customer_churn (1).xlsx')

telco.columns
telco.drop(['Count', 'Quarter'], axis=1 , inplace = True)

##Check for NA values##
telco.isna().sum()
telco.isnull().sum()
##no NA value##

#checkig for duplicate values ##
dup1= telco.duplicated()
dup1
sum(dup1)
##no duplicate found##

### gettting dummies variable##
new_telco = pd.get_dummies(telco)

from sklearn.preprocessing import OneHotEncoder

OH_enc = OneHotEncoder()
new_telco2 = pd.DataFrame(OH_enc.fit_transform(telco).toarray())

from sklearn.preprocessing import LabelEncoder

L_enc = LabelEncoder()
telco['Referred a Friend'] = L_enc.fit_transform(telco['Referred a Friend'])
telco['Offer'] = L_enc.fit_transform(telco['Offer'])
telco['Phone Service'] = L_enc.fit_transform(telco['Phone Service'])
telco['Multiple Lines'] = L_enc.fit_transform(telco['Multiple Lines'])
telco['Internet Service'] = L_enc.fit_transform(telco['Internet Service'])
telco['Internet Type'] = L_enc.fit_transform(telco['Internet Type'])
telco['Online Security'] = L_enc.fit_transform(telco['Online Security'])
telco['Online Backup'] = L_enc.fit_transform(telco['Online Backup'])
telco['Device Protection Plan'] = L_enc.fit_transform(telco['Device Protection Plan'])
telco['Premium Tech Support'] = L_enc.fit_transform(telco['Premium Tech Support'])
telco['Streaming TV'] = L_enc.fit_transform(telco['Streaming TV'])
telco['Streaming Movies'] = L_enc.fit_transform(telco['Streaming Movies'])
telco['Streaming Music'] = L_enc.fit_transform(telco['Streaming Music'])
telco['Unlimited Data'] = L_enc.fit_transform(telco['Unlimited Data'])
telco['Contract'] = L_enc.fit_transform(telco['Contract'])
telco['Paperless Billing'] = L_enc.fit_transform(telco['Paperless Billing'])
telco['Payment Method'] = L_enc.fit_transform(telco['Payment Method'])

##Boxplotting##
sns.boxplot(telco["Tenure in Months"]);plt.title("Boxplot");plt.show()
sns.boxplot(telco["Avg Monthly Long Distance Charges"]);plt.title("Boxplot");plt.show()

sns.boxplot(telco["Avg Monthly GB Download"]);plt.title("Boxplot");plt.show()

sns.boxplot(telco["Monthly Charge"]);plt.title("Boxplot");plt.show()
sns.boxplot(telco["Total Charges"]);plt.title("Boxplot");plt.show()

sns.boxplot(telco["Total Refunds"]);plt.title("Boxplot");plt.show()
sns.boxplot(telco["Total Extra Data Charges"]);plt.title("Boxplot");plt.show()
sns.boxplot(telco["Total Long Distance Charges"]);plt.title("Boxplot");plt.show()
sns.boxplot(telco["Total Revenue"]);plt.title("Boxplot");plt.show()

##scatter plot##
plt.scatter(telco["Tenure in Months"] , telco["Total Extra Data Charges"])
plt.scatter(telco["Monthly Charge"] , telco["Avg Monthly Long Distance Charges"])
plt.scatter(telco["Total Long Distance Charges"] , telco["Total Revenue"])

telco.columns
##Removing outlier using IQR ##

IQR = telco['Avg Monthly GB Download'].quantile(0.75)-telco['Avg Monthly GB Download'].quantile(0.25)
lower_limit_Avg_Monthly_GB_Download= telco['Avg Monthly GB Download'].quantile(0.25)-(IQR*1.5)
upper_limit_Avg_Monthly_GB_Download= telco['Avg Monthly GB Download'].quantile(0.75)+(IQR*1.5)
telco['Avg Monthly GB Download']=pd.DataFrame(np.where(telco['Avg Monthly GB Download']>upper_limit_Avg_Monthly_GB_Download,upper_limit_Avg_Monthly_GB_Download,
                                         np.where(telco['Avg Monthly GB Download']<lower_limit_Avg_Monthly_GB_Download,lower_limit_Avg_Monthly_GB_Download,telco['Avg Monthly GB Download'])))
sns.boxplot(telco['Avg Monthly GB Download']);plt.title('Boxplot');plt.show()

IQR = telco['Total Refunds'].quantile(0.75)-telco['Total Refunds'].quantile(0.25)
lower_limit_Total_Refunds= telco['Total Refunds'].quantile(0.25)-(IQR*1.5)
upper_limit_Total_Refunds= telco['Total Refunds'].quantile(0.75)+(IQR*1.5)
telco['Total Refunds']=pd.DataFrame(np.where(telco['Total Refunds']>upper_limit_Total_Refunds,upper_limit_Total_Refunds,
                                         np.where(telco['Total Refunds']<lower_limit_Total_Refunds,lower_limit_Total_Refunds,telco['Total Refunds'])))
sns.boxplot(telco['Total Refunds']);plt.title('Boxplot');plt.show()

IQR = telco['Total Extra Data Charges'].quantile(0.75)-telco['Total Extra Data Charges'].quantile(0.25)
lower_limit_Total_Extra_Data_Charges= telco['Total Extra Data Charges'].quantile(0.25)-(IQR*1.5)
upper_limit_Total_Extra_Data_Charges= telco['Total Extra Data Charges'].quantile(0.75)+(IQR*1.5)
telco['Total Extra Data Charges']=pd.DataFrame(np.where(telco['Total Extra Data Charges']>upper_limit_Total_Extra_Data_Charges,upper_limit_Total_Extra_Data_Charges,
                                         np.where(telco['Total Extra Data Charges']<lower_limit_Total_Extra_Data_Charges,lower_limit_Total_Extra_Data_Charges,telco['Total Extra Data Charges'])))
sns.boxplot(telco['Total Extra Data Charges']);plt.title('Boxplot');plt.show()

IQR = telco['Total Revenue'].quantile(0.75)-telco['Total Revenue'].quantile(0.25)
lower_limit_Total_Revenue= telco['Total Revenue'].quantile(0.25)-(IQR*1.5)
upper_limit_Total_Revenue= telco['Total Revenue'].quantile(0.75)+(IQR*1.5)
telco['Total Revenue']=pd.DataFrame(np.where(telco['Total Revenue']>upper_limit_Total_Revenue,upper_limit_Total_Revenue,
                                         np.where(telco['Total Revenue']<lower_limit_Total_Revenue,lower_limit_Total_Revenue,telco['Total Revenue'])))
sns.boxplot(telco['Total Revenue']);plt.title('Boxplot');plt.show()

###scaling method using min max method#
def std_fun(i):
    x= (i-i.mean())/(i.std())
    return(x)

telco_norm = std_fun(new_telco)

str(telco_norm)    
##K mean model building#

from sklearn.cluster import KMeans
TWSS= []
k = list(range(2,9))

for i in k:
    kmeans= KMeans(n_clusters=i)
    kmeans.fit(telco_norm)
    TWSS.append(kmeans.inertia_)
TWSS    

plt.plot(k,TWSS,'ro-'); plt.xlabel('No_of_clusters');plt.ylabel('total_within_SS')

model_telco= KMeans(n_clusters=3)
model_telco.fit(telco_norm)    

model_telco.labels_ ##getting label of assigned to each row#
cluster_telco = pd.Series(model_telco.labels_) ##converting numpy array into pandas series
telco['cluster'] = cluster_telco


telco.head()

telco.iloc[: , :].groupby(telco.cluster).mean()

telco.to_csv('Kmeans_telco.csv', encoding='utf-8')

import os
os.getcwd()
###############################Problem 5#############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
##loading dataset##
auto_data= pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Kmeans/AutoInsurance (1).csv')

auto_data.drop(['Customer'], axis = 1 ,inplace = True)

new_auto_data = auto_data.iloc[:,1:]

##checkind duplicate , na  value##
dup1 = new_auto_data.duplicated()
sum(dup1)
new_auto_data = new_auto_data.drop_duplicates()

new_auto_data.isna().sum()


##getting dummy variables##

dummy_auto_data = pd.get_dummies(new_auto_data)

new_auto_data.columns

##Boxplotting ##
sns.boxplot(new_auto_data["Customer Lifetime Value"]);plt.title("Boxplot");plt.show()

sns.boxplot(new_auto_data["Income"]);plt.title("Boxplot");plt.show()

sns.boxplot(new_auto_data["Monthly Premium Auto"]);plt.title("Boxplot");plt.show()

sns.boxplot(new_auto_data["Months Since Last Claim"]);plt.title("Boxplot");plt.show()
sns.boxplot(new_auto_data["Months Since Policy Inception"]);plt.title("Boxplot");plt.show()

sns.boxplot(new_auto_data["Total Claim Amount"]);plt.title("Boxplot");plt.show()

#scatterplott##
plt.scatter(new_auto_data["Customer Lifetime Value"] , new_auto_data["Income"])
plt.scatter(new_auto_data["Monthly Premium Auto"] ,new_auto_data["Months Since Last Claim"])
plt.scatter(new_auto_data["Months Since Policy Inception"] , new_auto_data["Total Claim Amount"])

new_auto_data.columns
##removing outlier#

IQR = new_auto_data['Customer Lifetime Value'].quantile(0.75)-new_auto_data['Customer Lifetime Value'].quantile(0.25)
lower_limit_Customer_Lifetime_Value= new_auto_data['Customer Lifetime Value'].quantile(0.25)-(IQR*1.5)
upper_limit_Customer_Lifetime_Value= new_auto_data['Customer Lifetime Value'].quantile(0.75)+(IQR*1.5)
new_auto_data['Customer Lifetime Value']=pd.DataFrame(np.where(new_auto_data['Customer Lifetime Value']>upper_limit_Customer_Lifetime_Value,upper_limit_Customer_Lifetime_Value,
                                         np.where(new_auto_data['Customer Lifetime Value']<lower_limit_Customer_Lifetime_Value,lower_limit_Customer_Lifetime_Value,new_auto_data['Customer Lifetime Value'])))
sns.boxplot(new_auto_data['Customer Lifetime Value']);plt.title('Boxplot');plt.show()

IQR = new_auto_data['Monthly Premium Auto'].quantile(0.75)-new_auto_data['Monthly Premium Auto'].quantile(0.25)
lower_limit_Monthly_Premium_Auto= new_auto_data['Monthly Premium Auto'].quantile(0.25)-(IQR*1.5)
upper_limit_Monthly_Premium_Auto= new_auto_data['Monthly Premium Auto'].quantile(0.75)+(IQR*1.5)
new_auto_data['Monthly Premium Auto']=pd.DataFrame(np.where(new_auto_data['Monthly Premium Auto']>upper_limit_Monthly_Premium_Auto,upper_limit_Monthly_Premium_Auto,
                                         np.where(new_auto_data['Monthly Premium Auto']<lower_limit_Monthly_Premium_Auto,lower_limit_Monthly_Premium_Auto,new_auto_data['Monthly Premium Auto'])))
sns.boxplot(new_auto_data['Monthly Premium Auto']);plt.title('Boxplot');plt.show()

IQR = new_auto_data['Total Claim Amount'].quantile(0.75)-new_auto_data['Total Claim Amount'].quantile(0.25)
lower_limit_Total_Claim_Amount= new_auto_data['Total Claim Amount'].quantile(0.25)-(IQR*1.5)
upper_limit_Total_Claim_Amount= new_auto_data['Total Claim Amount'].quantile(0.75)+(IQR*1.5)
new_auto_data['Total Claim Amount']=pd.DataFrame(np.where(new_auto_data['Total Claim Amount']>upper_limit_Total_Claim_Amount,upper_limit_Total_Claim_Amount,
                                         np.where(new_auto_data['Total Claim Amount']<lower_limit_Total_Claim_Amount,lower_limit_Total_Claim_Amount,new_auto_data['Total Claim Amount'])))
sns.boxplot(new_auto_data['Total Claim Amount']);plt.title('Boxplot');plt.show()


#scaling method using min max method#

def norm_fun(i):
    x = (i-i.min()/i.max()-i.min())
    return(x)

#normalization##
auto_data_norm= norm_fun(dummy_auto_data)

from sklearn.cluster import KMeans
##kmeans model building##

TWSS=[]
k= list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters= i)
    kmeans.fit(auto_data_norm)
    TWSS.append(kmeans.inertia_)
TWSS

plt.plot(k,TWSS,'ro-'); plt.xlabel('No_of_clusters');plt.ylabel('total_within_SS')

model_auto =KMeans(n_clusters=3)
model_auto.fit(auto_data_norm)

model_auto.labels_
cluster_auto=pd.Series(model_auto.labels_)
auto_data['clusters']=cluster_auto

auto_data = auto_data.iloc[:,[23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]

auto_data.head()

auto_data.iloc[: ,:].groupby(auto_data.clusters).mean()

auto_data.to_csv('Kmeans_auto_data.csv' , encoding = 'utf-8')

import os
os.getcwd()







