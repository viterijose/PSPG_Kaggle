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


# In[2]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os
import pandas as pd
import re

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# # A. Loading the Data
# The first part of this project will be to download and view some basic information pertaining to our dataset. This will allow us to have a general idea as to how to move forward with the project as well as understand how the initial data is strucutured. 

# In[3]:


#Training Data
train = pd.read_csv('train.csv')
train_labels = pd.read_csv('train_labels.csv')
test = pd.read_csv('test.csv')


# In[4]:


train.head() #Test to see if you can read the data


# In[5]:


train_labels.head() #Test to see if you can read the data


# In[6]:


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

# In[7]:


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


# In[8]:


train_df = reduce_memory(train)
train_df.info()


# As the above illustrates, we have reduced the amount of memory used from `3.9+ GB` to `1.0 GB`. This is about a 75% decrease! We also will reduce the memory usage of the labels (as shown below).  

# In[9]:


labels_df = reduce_memory(train_labels)
labels_df.info()


# In[10]:


labels_df['user_id']=labels_df.session_id.str.split("_", expand = True)[0]


# In[11]:


del(train) #saves memory


# # C. Preprocessing the Dataset
# Having reduced the memory required to download the data, we will now prepocess the data and prepare it for the model.

# In[12]:


# Creating a summary of the dataset
def summary(df):
    summary_df = pd.DataFrame(df.dtypes, columns=['data type'])
    summary_df['#missing'] = df.isnull().sum().values * 100      #Calculates the number of missing values
    summary_df['#unique'] = df.nunique().values                  #Caluclates the number of unique values   
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summary_df['min'] = desc['min'].values                       #For numerical data, calculates the min value
    summary_df['max'] = desc['max'].values                       #For numerical data, calulcates the max value
    
    return summary_df #displays summary dataframe


# In[13]:


summary_table = summary(train_df)
summary_table


# Using the summary table above, we were able to decide on perform the following tasks to modify our original dataset:
# 1. We can one-hot encode the attributes `event_name`, `name`, and `room_fqid` to chang ethe categorical data into numerical  
# 
# 2. Although the attributes `fullscreen`, `hq`, and `music` contain numerical elements, they are categorical attributes in disguise. Therefore, we will also one-hot encode these attributes aswell. 
# 
# 3. The numerical attributes `elasped_time` and `hover_duration` appear to have outliers (based on the range between min and max values). Therefore, we must remove outliers prioir to the creation ofour model. 

# ### a. One-Hot Encoding Categorical Variables
# The following converts all of the categorical variables listed above (excluding `text` columns) into numerical ones:

# In[14]:


#========================================================================
# One-Hot Encoding Categorical Variables
#========================================================================

cat_att = ['event_name', 'name', 'room_fqid', 'fullscreen', 'hq', 'music'] #categorical attributes (from above)

for column in cat_att:
    temp_df = pd.get_dummies(train_df[column], prefix=column)
    
    train_df = pd.merge(
        left = train_df,
        right = temp_df,
        left_index = True,
        right_index = True,
    )
    
train_df.head()


# In[15]:


# Getting shape of the df
shape = train_df.shape
  
# Printing Number of columns
print('Number of columns :', shape[1])


# In[16]:


#dropping the unnessary attributes
train_df = train_df.drop(columns = cat_att, axis = 1)


# In[17]:


# Getting shape of the df
shape = train_df.shape
  
# Printing Number of columns
print('Number of columns :', shape[1])


# Having changed some initial categorical attributes into numerical ones, we will address the outliers.
# 
# ### b. Handling Outliers (Winzorization Method)
# To handle the following outliers, we will cap the data values that pertain to the numerical attributes. By capping the data values, we maintain the same number of observations while still dealing with outliers. Our caps will be at the 5th percentile and 95th percentile. 

# In[18]:


#===================================================================
# Confirming Outliers
#===================================================================

num_attr = ['elapsed_time', 'hover_duration'] #numerical attributes listed above

def CountOutliers(attributes):
    num_outliers = 0
    
    for col in attributes:
        #Finds some Summary Statistics
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
    
        num_outliers = ((train_df[col]<(Q1-1.5*IQR)) | (train_df[col]>(Q3+1.5*IQR))).sum()
    
        print("There are "+ str(num_outliers) + " outliers in the " + col + " column.")
        
CountOutliers(num_attr) #Test

#Sets up a zoomed in version of the boxplot
# plt.rcParams["figure.figsize"] = [16.0, 4.0] #sets the graph sizes
# plt.xlim(0,4000000) #Displays a zoomed in version of the boxplot
# train_df.boxplot(column= 'elapsed_time', return_type='axes',vert = False) #elasped Time


# In[56]:


from scipy.stats.mstats import winsorize #library neeeded
#========================================================================================
# Removing Outliers
#========================================================================================

# Winzorization Method resticts the data of elasped_time column up to the 90% percentile
train_df['elapsed_time_winsr'] = winsorize(train_df['elapsed_time'], limits=[None, 0.09]) #elasped time

# Winzorization Method resticts the data of elasped_time column up to the 85% percentile
mask = ~train_df['hover_duration'].isna() #displays which elements are NaN in the hover_duration column
train_df.loc[mask, 'hover_duration_winsr'] = winsorize(train_df['hover_duration'].loc[mask], limits=[None, 0.15])


# In[57]:


num_attr = ['elapsed_time_winsr', 'hover_duration_winsr']
CountOutliers(num_attr) #Test


# In[58]:


train_df.boxplot(column= 'hover_duration_winsr', return_type='axes', vert = False)


# In[60]:


train_df.boxplot(column= 'elapsed_time_winsr', return_type='axes', vert = False)


# In[61]:


#Drops the unnecessary columns
train_df = train_df.drop(columns = ['elapsed_time', 'hover_duration'], axis = 1)


# In[62]:


train_df.head()


# 

# ### c. Modifying the Text Data
# As `train_df` shows, there is an attribute called `text`. We believe that the text that is played at each observation (if applicable) is important. We assume that particular responses are choosen depending on whether or not the player was able to correctly answer the question. Therefore, prior to any model selection, we must first perform some text mining and language processing. 

# # STILL NEED TO FINISH THIS SECTION THEN OFF TO FEATURE SELECTION WITH RANDOM FORREST

# In[ ]:





# In[35]:


type(train_df['room_coor_x'])
type(train_df['elapsed'])


# In[32]:


type(winsorize(train_df['elapsed_time'], limits=[None, 0.05]))


# In[ ]:


df2 = pd.DataFrame(d2)


# In[ ]:





# In[ ]:


for col in df:
    print(df[col].unique())


# In[47]:


print(train_df['name'].unique())


# In[48]:


print(train_df['text'].unique())


# In[61]:


l = train_df['text_fqid'].unique()


# In[62]:


for i in l:
    print(i)


# In[64]:


m = train_df['text'].unique()
for i in m:
    print(i)


# In[ ]:





# In[17]:


labels_df.head()


# In[20]:


labels_df["level"] = labels_df.session_id.str.split("_", expand = True)[1]
labels_df["level"] = labels_df["level"].apply(lambda x : re.sub("\D", "",x)) 
labels_df["level"] = pd.to_numeric(labels_df["level"])
labels_df["user_id"] = pd.to_numeric(labels_df["user_id"])
labels_df["session_level"] = labels_df["level"].apply(lambda x: 0 if x <= 4 else 1 if x >= 5 and x <= 12 else 2)


# In[21]:


labels_df.head()


# In[22]:


# Questions 1-4 belong to level 1, 5-12 to level 2, 13 - 22 to level 3
labels_df.level.unique()


# In[23]:


print("Number of unique users: ",len(labels_df.user_id.unique()))
print("Number of unique sessions: ",len(labels_df.session_id.unique()))


# In[24]:


train_df.isnull().sum()


# In[25]:


numeric_feature_names = ['session_id', 'index', 'elapsed_time', 'level',
       'page', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y',
       'hover_duration', 'fullscreen', 'hq', 'music']
numeric_features = train_df[numeric_feature_names].copy()
numeric_features.head()


# In[26]:


# Based on the data described in the notebook, this is an MNAR type, meaning, the value is missing not at random 
numeric_features.isnull().sum()


# In[27]:


numeric_features.shape,labels_df.shape


# In[29]:


len(numeric_features['session_id'].unique()),len(train_labels['user_id'].unique())


# In[30]:


numeric_features['hover_duration'].describe()


# In[32]:


# Creating a copy of my labels so I can modify the column names and keep the raw dataset intact
labels_df_cp= labels_df.copy()
labels_df_cp.rename(columns = {'session_id':'session_res','user_id':'session_id'}, inplace = True)
labels_df_cp.head()
# There is no level 0 in the training labels provided, how should we handle this?
train_df_cp = train_df.copy()
df_full = pd.merge(train_df_cp, labels_df_cp, how='inner',on=['session_id','level'])
df_full.head()
df_full.shape


# In[33]:


scaler = MinMaxScaler()
scaler.fit(df_full[['elapsed_time', 'fullscreen','room_coor_x','room_coor_y','screen_coor_x',
                    'screen_coor_y','hover_duration']])
training_data_scaled = scaler.transform(df_full[['elapsed_time', 'fullscreen','room_coor_x','room_coor_y','screen_coor_x',
                                                 'screen_coor_y','hover_duration']])
training_data_scaled = pd.DataFrame(df_full, columns=['elapsed_time_scaled', 'fullscreen_scaled','room_coor_x_scaled',
                                                      'room_coor_y_scaled','screen_coor_x_scaled','screen_coor_y_scaled',
                                                      'hover_duration_scaled'])
df_full = pd.concat([df_full, training_data_scaled], axis=1)
print('Dataset shape: ',df_full.shape,'\n')
df_full.head()


# In[34]:


training_data=df_full[['elapsed_time_scaled','fullscreen_scaled']]
label_data=df_full[['correct']]
print('Training data shape: ',training_data.shape,'\n','Label data shape: ',label_data.shape)
x_train,x_val = training_data[:int(len(training_data)*.8)],training_data[int(len(training_data)*.8):]
y_train,y_val = label_data[:int(len(label_data)*.8)],label_data[int(len(label_data)*.8):]
print('X train shape: ',x_train.shape,'\n','X valid shape: ',x_val.shape)
model = keras.models.Sequential([
    keras.layers.Dense(10, input_shape=(2,),activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])


# In[35]:


model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])


# In[36]:


history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_val, y_val))


# In[ ]:




