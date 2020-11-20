#!/usr/bin/env python
# coding: utf-8

# Source of data: https://www.kaggle.com/saurabhtayal/diabetic-patients-readmission-prediction
# This dataset is about the diabetes cases taking into account 130 hospitals in the US for years 1999-2008

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

df = pd.read_csv("datasource/diabetic_data.csv")


# ************************************************ DATA PREPROCESSING ***********************************************************

# Get to know the data

# In[ ]:


#Get number of rows and columns
df.shape


# In[ ]:


#Get column names
df.columns


# In[ ]:


#Get the first 10 line
df.head(5)


# Refactor & looking for missing data

# In[ ]:


#Replace question marks with NaN
df.replace("?", np.nan, inplace=True)
df.sample(5)


# In[ ]:


#Get columns where can be found NaN value
columns_with_nan = df.columns[pd.isnull(df).sum() > 0].tolist()
columns_with_nan


# In[ ]:


#Get for these columns the number of NaN value and the percentage of the occurrence of NaN
dict_cols_with_nan = {column: [] for column in columns_with_nan}

for column in columns_with_nan:
    count = df[column].isnull().sum()
    percentage = round(count / df.shape[0] * 100, 1)
    dict_cols_with_nan[column] = [count, percentage]

data_of_NaN = pd.DataFrame(dict_cols_with_nan)
data_of_NaN.index=["num_of_NaN_rows", "percentage"]
data_of_NaN


# Clean the dataset

# In[ ]:


#Remove rows from race with NaN value
df.drop(df[df["race"].isnull()].index, inplace=True)

#Also, remove rows with NaN from "diag_1", "diag_2", "diag_3"
df.drop(df[df["diag_1"].isnull()].index, inplace=True)
df.drop(df[df["diag_2"].isnull()].index, inplace=True)
df.drop(df[df["diag_3"].isnull()].index, inplace=True)

#Remove redundant columns
df.drop(columns=["payer_code", "medical_specialty"], axis=1, inplace=True)

#Remove columns from "diag_1" to "citoglipton" and from "glyburide-metformin" to "diabetesMed"
df.drop(df.loc[:, 'diag_1':'citoglipton'].columns, axis = 1, inplace=True) 
df.drop(df.loc[:, 'glyburide-metformin':'diabetesMed'].columns, axis = 1, inplace=True) 


# Filling the NaN values in weight column
# At this point the weight column has 97% of the row with NaN value. Just for the sake of practice, I decided to keep this column still and attempt to fill in the flaws. I was trying to fill out the missing values according to the gender and age category.

# In[ ]:


#Get how many females and males are in the dataset
df["gender"].value_counts()


# In[ ]:


#Get the index of the unknown values
idx_of_unknown_v = []

for index, row in df.iterrows():
    if row[3] not in ["Male", "Female"]:
        idx_of_unknown_v.append(index)

df.drop(idx_of_unknown_v, inplace=True)
df["gender"].value_counts()

#Reset indexes
df.reset_index(drop=True, inplace=True)


# In[ ]:


#Get the rows in weight column, which are not NaN values
df_notna_in_weight = df[df['weight'].notna()]

#Get how many row is this
df_notna_in_weight.shape[0]


# In[ ]:


#Save in another variable the non NaN weight rows and certain columns of those rows
df_checkweight = df_notna_in_weight[["weight","gender","age"]]


# In[ ]:


#Define a reusable function to get the weights by categories both for females and males

def get_weights_by_age_categories(age_list, data):
    """
    age_list: the age categories what can be found for particular genders
    data: is a subset of the dataframe from the df_chackweight with columns weight and age
    """
    #Define a dictionary    
    weight_by_age = {}
    
    #Loop through on the unique age categories
    for age in age_list:
        #Get weights from rows where the age category is matching and create a set of arrays from them
        weights_by_current_age = data[data["age"] == age].weight
        
        unique_weights_by_current_age = set(weights_by_current_age.values)
        """
        Add item to the weight_by_age dictionary with current age as key
        And a nested dictionary as value which has the current weight 
        as key and the number of occurrence of those weights in that age category
        """
        #Get the item of the dictionary - multidimensional: key=age category, value={key=weight, value=amount of the key}
        weight_by_age[age] = {weight: weights_by_current_age[weights_by_current_age.values == weight].count()                 for weight in unique_weights_by_current_age}
    
    return weight_by_age


# In[ ]:


#Get rows with non NaN values only for females
females = df_checkweight[df_checkweight['gender'] == 'Female']
f = females[["weight","age"]]


# In[ ]:


#Get all type of age category wich has non NaN value
f_age_list = list(dict.fromkeys(f["age"].tolist()))
f_age_list


# In[ ]:


#Get unique age categories and corresponding weights for females:
f_weights = get_weights_by_age_categories(f_age_list, f)
f_weights


# In[ ]:


#Get how many percentage is this part of the dataset which has non NaN weight value for females
round(females.shape[0] / df[df["gender"]== "Female"].shape[0] * 100,1)


# In[ ]:


#Doing the same with males
males = df_checkweight[df_checkweight['gender'] == 'Male']
m = males[["weight","age"]]
m_age_list = list(dict.fromkeys(m["age"].tolist()))

#Get unique age categories and corresponding weights for males:
m_weights = get_weights_by_age_categories(m_age_list, m)

#Get how many percentage is this part of the dataset which has non NaN weight value for males
round(males.shape[0] / df[df["gender"]== "Male"].shape[0] * 100,1)
m_weights


# In[ ]:


#Define a funtion which fills the NaN values with the most common weight category in the current age category
def fill_in(gender, age_list, weights):
    """
    gender: since we cant to get this function for females and males 
    we have to determine for which gender do we want to get the fill out
    
    age_list: the age categories can be found for a particular gender
    
    weights: dictionary, which has the age_categories and in that age category different weight categories with number of occurrence
    
    """
    for index, row in df.iterrows():
        #If the value of weight is nan we proceed    
        if str(row[5])=="nan" and row[3] == gender:
            for age_cat in age_list:
                #Find from the proper dictionary the age category needed
                if row[4] == age_cat:
                    #Get from dictionary the most common weight in the current age category
                    max_key = max(weights[age_cat], key=weights[age_cat].get)
                    #Replace weight with the max key
                    df.loc[index, "weight"] = max_key


# In[ ]:


#Use the function for both females and males 
#(NOTE: THIS MIGHT BROKE THE JUPITER; SO I USED SPYDER3 FOR TESTING...ANYWAY IT TAKES SOME TIME)
fill_in("Female", f_age_list, f_weights)
fill_in("Male", m_age_list, m_weights)

df[["gender","age","weight"]].head(10)


# In[ ]:


#Check for remaining null values in weight column
df[df["weight"].isnull()]


# In[ ]:


#Turns out from the result that the values are still NaN for the weight in certain rows
#This is because there is no weight value for any row in age category "[0-10)" in females
df[df["age"]=="[0-10)"]


# In[ ]:


#Decided to get rid of these rows 
null_columns=df.columns[df.isnull().any()]
df[df["weight"].isnull()][null_columns]

#Check if the deletion was successful
df.drop(df[df["weight"].isnull()].index, inplace=True)
df[null_columns].isnull().sum()


# In[ ]:


#In case something would break, I saved the result into a file called "out.csv". Find this file in the attachment. 
#Write out data - using the following lines
df.to_csv(index=False)
compression_opts = dict(method='zip',
                        archive_name='out.csv')  
df.to_csv('out.zip', index=False,
          compression=compression_opts)  


# Looking for outliers using isolation forest

# In[ ]:


#Read the result csv
df2 = pd.read_csv("out/out.csv")

# 'if' ~ isolation forest
df2_if = df2.copy()

df2.head(5)


# In[ ]:


#Estimate contamination
contamination = 0.05

# Label encoding 
for col in df2_if.columns:
    if df2_if[col].dtype == 'object':
        le = LabelEncoder()
        df2_if[col].fillna("None", inplace=True)
        le.fit(list(df2_if[col].astype(str).values))
        df2_if[col] = le.transform(list(df2_if[col].astype(str).values))
    else:
        df2_if[col].fillna(-999, inplace=True)

model = IsolationForest(contamination=contamination, n_estimators=1000)
model.fit(df2_if)

#Add new column to dataset containing 1 and -1
# -1 indicates outliers in row according to the program
df2_if["iforest"] = pd.Series(model.predict(df2_if))

#Get the contamination ratio and comper with the estimation
print(df2_if["iforest"].value_counts())
outlier_ratio = df2_if[df2_if["iforest"] == -1].shape[0] / df2_if.shape[0]   # Indeed 5%
outlier_ratio


# AFTER PREPROCESSING
# 
# remaining rows and columns:
#     rows: 97988 / 101765
#     columns: 48 / 50
# 
# Decided to keep the weight column after I succeeded to fill the NaN values, 
# however these values are not certainly reliable.
# The column has been filled up according to the data of appr. 3000 rows 
# taking into account the age and gender

# ******************************************** Data Objects & Attribute Types **************************************************

# In[ ]:


#print out datatypes of columns
df2.dtypes


# In[ ]:


#NOMINALS
nominal_columns = df2.loc[:, df2.dtypes == object].columns
nominal_columns


# In[ ]:


#Columns with integer values
columns_int = df2.loc[:, df2.dtypes != object].columns
print(columns_int)
print("")

#NOMINAL columns with integer value:
nominal_cols_int = ["encounter_id", "patient_nbr", "admission_type_id", "discharge_disposition_id", "admission_source_id"]
print("NOMINAL columns with integer value:")
print(nominal_cols_int)
print("")

#Numeric attributes
print("Numeric attributes")
for column in columns_int:
    if column not in nominal_cols_int:
        print(column)


# ********************************************** Measurement of the central Tendency ********************************************

# In[ ]:


#RANGE OF num_medications
num_med_range = df2["num_medications"].max() - df2["num_medications"].min()
print("Range for number of medications is for a person is: " + str(num_med_range))


# In[ ]:


#Quantiles
print(df2['num_medications'].quantile([0.6]))
print(df2['num_medications'].quantile([0.7]))


# In[ ]:


#Interquartiles
df2["num_medications"].describe()
Q1 = df2.num_medications.quantile(0.25, interpolation = 'midpoint')
Q3 = df2.num_medications.quantile(0.75, interpolation = 'midpoint')
IQR = Q3-Q1
print("The Interquantile range is: ", IQR)


# In[ ]:


#SKEWNESS
#num_medications
sns.distplot(df2["num_medications"])
plt.show()

"""
The result shows that extreme values are located on the right, so this distribution is positively skewed.
The diagram shows that in most cases the number of medication needed is approximately between 10 and 20.
"""


# In[ ]:


#time in hospital
sns.distplot(df2["time_in_hospital"])
plt.show()
"""
Same type of diagram here: positively skewed
"""


# In[ ]:


#ages using encoded ages
#Get the encoded version of ages from the df used for isolation forest
age_codes = df2_if[["age"]]

#Add to the other dataframe (used for visualization)
df2["age_code"] = age_codes
sns.distplot(df2["age_code"])
plt.show()

"""
This diagram shows the that the extreme values are on the left so this distribution is negatively skewed.
The most patients are in age in between 60-80 years.
"""


# In[ ]:


#Check age categories by codes
age_translator = df2[["age","age_code"]].drop_duplicates()

age_translator.sort_values(by = ["age_code"], inplace =True)
age_translator_dict = {}
for index, row in age_translator.iterrows():
    age_translator_dict[row[1]] = row[0]
age_translator_dict


# In[ ]:


#KUSTOSIS
#num_medications
print("Kurtosis of number of medications:", df2['num_medications'].kurtosis())
"""
result: 3.4963398143243185
Leptokurtic - positive values use to indicate that more values are close to the mean, 
which makes the diagram highly peaked
"""


# In[ ]:


#ages
print("Kurtosis of time in hospital:", df2['time_in_hospital'].kurtosis())
"""
result: 0.8162859340700073
Leptokurtic

close to
Mesokurtic - values close to zero use to indicate that the distribution is normal, 
which makes the diagram medium peaked height
"""


# In[ ]:


#DEVIATION: low value denotes that data values are close to mean 
#num_medications
df2['num_medications'].std()
"""
result:  8.107139550425275
"""


# In[ ]:


#age
df2['time_in_hospital'].std()
"""
result:  2.993512211107218
"""


# In[ ]:


#Get result for all columns
df2.std(axis = 0)


# SOME VISALIZATIONS

# DIAGRAM1 - Show the number of medications by age categories indicating genders and weight by changing the size of the points
# Find the result of this diagram in the attachments (diagram1.jpg)

# In[ ]:


#Get weights in int format 
weights = df2[["weight"]]

#Create a dictionary to assing numbers to age categories in ascending order
weights_to_numeric = {"[0-25)": 1, "[25-50)": 2, "[50-75)": 3, "[75-100)": 4, "[100-125)": 5, "[125-150)": 6, "[150-175)": 7, "[175-200)": 8, ">200": 9}
#Replace values according to the dictionary
weights.replace(weights_to_numeric, inplace=True)

#Rename weight to weight_code and add column to the Dataframe
df2["weight_code"] = weights

#Set labels
plt.ylabel("Age categories")
plt.xlabel("Number of medication")
plt.title("Num of medications by ages indicating weights and gender")
tick_val = [0,2,4,6,8]
tick_lab = ['[0-10)','[20-30)','[40-50)','[60-70)','[80-90)']
plt.yticks(tick_val, tick_lab)

#Get numpy array of the converted weights and power the up in order to increase the difference between the values
#to get more clearly the different weights in the diagram
np_weights = np.array(weights)
np_weights = np.power(np_weights, 2.5)

#Set dictionary for colors
d = {
    'Female':'red',
    'Male':'blue',
}

#Get a Series of colors accordin to the dataframe gender column
col = df2["gender"].replace(d)

#Add legends
red_patch = mpatches.Patch(color='red', label='Female')
blue_patch = mpatches.Patch(color='blue', label='Male')
plt.legend(handles=[red_patch, blue_patch], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

#Create diagram
plt.scatter(df2["num_medications"],df2["age_code"], s = np_weights, c = col, alpha = 0.2)


# DIAGRAM2 - Show the sum of readmissions by ages indicating genders
# Find the result of this diagram in the attachments (diagram2.jpg)

# In[ ]:


#Get 1 for readmitted regardless of how many days after the patient has been readmitted
readmitted_codes = df2_if[["readmitted"]]
df2["readmitted_code"] = readmitted_codes
r_d = {1 : 1, 2 : 0, 0 : 1}
df2["readmitted_code"].replace(r_d, inplace=True)
df2.head(5)

#Save into an array the sum of readmission by the increasing age categories for females
females_df = df2[df2["gender"] == "Female"]
np_readmitted_sum_by_age_f = np.array(females_df[["age_code", "readmitted_code"]].groupby("age_code").sum())

#Save into an array the sum of readmission by the increasing age categories for males
males_df = df2[df2["gender"] == "Male"]
np_readmitted_sum_by_age_m = np.array(males_df[["age_code", "readmitted_code"]].groupby("age_code").sum())

#use reshape to convert array to 2d array
np_age_codes = np.reshape(np.array(age_translator["age_code"]), (-1, 1))

#For females there's no readmitted_code for age_code 0 so we have to delete the first age code here
np_age_codes_f = np.reshape(np.delete(np.array(age_translator["age_code"]), 0), (-1, 1))

#Create datafram which contains the age categories and the sum of readmission for the proper age category
#FEMALES
females_df2 = pd.DataFrame(np.concatenate((np_readmitted_sum_by_age_f,np_age_codes_f), axis=1), columns=["Sum_of_readmitted", "age_code"])

#MALES
males_df2 = pd.DataFrame(np.concatenate((np_readmitted_sum_by_age_m,np_age_codes),axis=1), columns=["Sum_of_readmitted", "age_code"])

#Add a new column to the dataframes with the proper color we want to use in the diagram
females_df2["color"] = "red"
males_df2["color"] = "blue"

#Now that the dataframes contains the colors as well, we can concatenate the two
f_m_df = pd.concat([females_df2, males_df2], ignore_index=True)

#Set labels
plt.ylabel("Sum of readmission")
plt.xlabel("Age categories")
plt.title("Num of readmissions by ages indicating genders")
xtick_val = [0,2,4,6,8]
xtick_lab = ['[0-10)','[20-30)','[40-50)','[60-70)','[80-90)']
plt.xticks(xtick_val, xtick_lab)
xtick_val = [0,2,4,6,8]
xtick_lab = ['[0-10)','[20-30)','[40-50)','[60-70)','[80-90)']
plt.xticks(xtick_val, xtick_lab)

#Set legends
red_patch = mpatches.Patch(color='red', label='Female')
blue_patch = mpatches.Patch(color='blue', label='Male')
plt.legend(handles=[red_patch, blue_patch], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

#Create diagram
plt.scatter(f_m_df["age_code"], f_m_df["Sum_of_readmitted"], s=100, c=f_m_df["color"], alpha = 0.6 )

