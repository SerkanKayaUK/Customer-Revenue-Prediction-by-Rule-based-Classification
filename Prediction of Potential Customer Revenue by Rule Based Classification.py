########################################################################################################
#                                                                                                      #
#  This project has been performed in Data Science Machine Learning Bootcamp at VBO , (c) MIUUL        #
#                                                                                                      #
#           -     Participant: Dr. Serkan KAYA  , 14 December 2021                                     #
#                                                                                                      #
########################################################################################################


###########################################################################################
#                                          TASK 1                                         #
###########################################################################################
#  Descriptive Statistics
########################################################################################
# Q1: Read the persona.csv file and show the general data statistics about the dataset.
########################################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('/Users/mac/Desktop/DATA_SCIENCE/VERIOKULU/DERSLER/2_HAFTA/Ders Oncesi Notlar/persona.csv')

def check_df(dataframe, head=5):
    print('#################### Shape #########################')
    print(dataframe.shape)
    print('#################### Type ##########################')
    print(dataframe.dtypes)
    print('#################### Head #########################')
    print(dataframe.head(head))
    print('##################### Tail #########################')
    print(dataframe.tail(head))
    print('###################### NA ########################')
    print(dataframe.isnull().sum())
    print('###################### Quantile ########################')
    print(dataframe.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

check_df(df)
df.shape

########################################################################################
# Q2: How many unique SOURCE are there? What are their frequencies?
########################################################################################

df['SOURCE'].unique()  # Just to see the type of source
df['SOURCE'].nunique()
df['SOURCE'].value_counts()

########################################################################################
# Q3: How many unique PRICEs are there?
########################################################################################

df['PRICE'].nunique()

########################################################################################
# Q4: How many sales were made from which PRICE?
########################################################################################

df['PRICE'].value_counts()

########################################################################################
# Q5: How many sales occured based on the country?
########################################################################################

df['COUNTRY'].value_counts()

########################################################################################
# Q6: How much was totally earned by each country?
########################################################################################

df.groupby('COUNTRY')['PRICE'].sum()

########################################################################################
# Q7: What are the sales numbers according to SOURCE types?
########################################################################################

df['SOURCE'].value_counts()

########################################################################################
# Q8: What are the PRICE averages by country?
########################################################################################

df.groupby(['COUNTRY'])['PRICE'].mean()
# df.groupby(['COUNTRY']).agg({'PRICE':'mean'})

########################################################################################
# Q9: What are the PRICE averages according to SOURCEs?
########################################################################################

df.groupby('SOURCE')['PRICE'].mean()

########################################################################################
# Q10: What are the PRICE averages in the COUNTRY-SOURCE breakdown?
########################################################################################

df.groupby(['COUNTRY','SOURCE'])['PRICE'].mean()


###########################################################################################
#                                          TASK  2                                        #
#       What are the average earnings in breakdown of COUNTRY, SOURCE, SEX, AGE?          #
###########################################################################################

df.head(3)
df.groupby(['COUNTRY','SOURCE','SEX','AGE']).aggregate({'PRICE':'mean'})

###########################################################################################
#                                          TASK 3                                         #
# Sort the output by PRICE ? To better see the output in the previous question, apply     #
# the sort_values method in descending order of PRICE.                                    #
###########################################################################################
# First, I just want to see the results of the command below..
df.groupby(['COUNTRY','SOURCE','SEX','AGE']).aggregate({'PRICE':'mean'}).sort_values(by='PRICE',ascending=False)
# now let us assign this to a variable, agg_df
agg_df=df.groupby(['COUNTRY','SOURCE','SEX','AGE']).aggregate({'PRICE':'mean'}).sort_values(by='PRICE',ascending=False)


###########################################################################################
#                                          TASK 4                                         #
#                       Convert the names in the index to variable names.                 #
###########################################################################################

agg_df.reset_index(inplace=True)
agg_df

###########################################################################################
#                                          TASK  5                                        #
#              Convert age variable to categorical variable and add it to agg_df.         #
###########################################################################################

my_labels = ['0_18', '19_23', '24_30', '31_41', f'42_{agg_df.AGE.max()}']
agg_df["AGE_CUT"]=pd.cut(agg_df["AGE"],bins=[0,18,23,30,41,agg_df.AGE.max()],labels=my_labels)


###########################################################################################
#                                          TASK 6                                        #
#              Yeni seviye tabanlı müşterileri (persona) tanımlayınız                     #
###########################################################################################

agg_df["customers_level_based"]=[(i[0]+"_"+i[1]+"_"+i[2]+"_"+i[5]).upper() for i in agg_df.values]
agg_df=agg_df.groupby("customers_level_based").mean()
agg_df.reset_index(inplace=True)
agg_df

###########################################################################################
#                                          TASK  7                                        #
#                             Segment new customers (personas).                           #
###########################################################################################

agg_df['SEGMENT']=pd.qcut(agg_df['PRICE'],4,labels=['D','C','B','A'])
agg_df.head(5)
agg_df.columns

## Describe segments (Group by segments and get price mean, max, sum etc)

agg_df.groupby('SEGMENT').aggregate({'PRICE':['min','max','mean','sum']})

## Analyze C segment (Only extract C segment from dataset and analyze)

agg_df[agg_df['SEGMENT']=='C'].describe().T

###########################################################################################
#                                          TASK 8                                         #
#    Classify new customers by segment and estimate how much revenue they can generate    #
###########################################################################################

## What segment does a 33-year-old Turkish woman using ANDROID belong to and how much income is expected on average?


new_user="TUR_ANDROID_FEMALE_31_41"
agg_df[agg_df['customers_level_based']==new_user]

## In which segment and on average how much income would a 35-year-old French woman using IOS expect to earn?

new_user="FRA_IOS_FEMALE_31_41"
agg_df[agg_df['customers_level_based']==new_user]

