#!/usr/bin/env python
# coding: utf-8

# In[1]:


#======================================
# Libraries Needed
#======================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras


# # A. Loading the Data
# The first part of this project will be to download and view some basic information pertaining to our dataset. This will allow us to have a general idea as to how to move forward with the project as well as understand how the initial data is strucutured. 

# In[2]:


#Training Data
train = pd.read_csv('train.csv')
train_labels = pd.read_csv('train_labels.csv')
test = pd.read_csv('test.csv')


# In[3]:


train.head() #Test to see if you can read the data


# In[4]:


train_labels.head() #Test to see if you can read the data


# In[5]:


train.info()


# #### Summary of Initial Finding
# Even though we have just loaded the data, there are several key features that can be idenitified. These will need to be addressed moving forward and will be vital to the creation of model:
# 1. As shown by the `train.info()` the dataset that we are using is massive:
#     * Before we begin dealing with any type of feature analysis we must reduce the amount of memory used up by our system. 
#     * The result of this is to improve the performance of the runtime of our model <br><br>
# 2. The dataset, while large, consisits of several observations pertaining to the same `session_id`. 
#     * This implies that we will need to perform some aggreagation prior to any modeling. 

# # B. Reducing Memory Usage of the Dataset
# Having now downloaded data, we will make it more manegagble to use by reducing the memory required to process it.

# In[11]:


# Function to reduce memory
def reduce_memory(df):   
    for col in df.columns:
        col_type = df[col].dtype.name
        
        #Only focuses on numerical data (categorical data is handled later)
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')): #DateTime object and Category object
            if (col_type != 'object'): #Object type
                col_min = df[col].min()
                col_max = df[col].max()

                #Only focuses on if the type of the attribute is of type 'int'
                # np.iinfo() finds the Machine Limits for the data type
                if str(col_type)[:3] == 'int':
                    #Case 1: If the Machine Limits of the attribute fall between those of type int8
                    if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8) #Changes the type to int8
                    #Case 2: If the Machine Limits of the attribute fall between those of type int16
                    elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16) #Changes the type to int16
                    #Case 3: If the Machine Limits of the attribute fall between those of type int32
                    elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32) #Changes the type to int32
                    #Case 4: If the Machine Limits of the attribute fall between those of type int64
                    elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64) #Changes the type to int64

                #Only focuses on if the type of the attribute is of type 'float'
                # np.finfo() finds the Machine Limits for the data type
                else:
                    #Case 1: If the Machine Limits of the attribute fall between those of type float16
                    if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    #Case 1: If the Machine Limits of the attribute fall between those of type float32
                    elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    #All other cases doesn;t change
                    else:
                        pass
            
            #If the attribute is an object than it will change its type to category
            else:
                df[col] = df[col].astype('category')
    
    return df


# In[10]:


train_df = reduce_memory(train)
train_df.info()


# In[ ]:




