#!/usr/bin/env python
# coding: utf-8

# # Predicting the Sale Price of Bulldozers using ML
# 
# ## 1. Problem Defination 
# 
# > Predict future price of Bulldozer , based on characteristics 
# 
# ## 2. Data 
# 
# > Data downloaded from https://www.kaggle.com/c/bluebook-for-bulldozers
# 
# The data for this competition is split into three parts:
# 
# * Train.csv is the training set, which contains data through the end of 2011.
# * Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
# * Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.
# 
# The key fields are in train.csv are:
# 
# * SalesID: the uniue identifier of the sale
# * MachineID: the unique identifier of a machine.  A machine can be sold multiple times
# * saleprice: what the machine sold for at auction (only provided in train.csv)
# * saledate: the date of the sale
# 
# ## 3. Evaluation
# 
# The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.
# 
# Goal : Minimize the Error RMSLE
# 
# ## 4. Features
# 
# Check Data Dictionary
# https://www.kaggle.com/competitions/bluebook-for-bulldozers/data
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[2]:


#Import training and validation sets
df = pd.read_csv("TrainAndValid.csv",low_memory=False)


# In[3]:


df.info()


# In[4]:


df.isna().sum()


# In[5]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000],df["SalePrice"][:1000])


# In[6]:


df.saledate[:1000]


# In[7]:


df.SalePrice.plot.hist()


# ### Parsing Dates
# 
# Tell pandas which col has dates in it using parse dates

# In[8]:


#Import data again but this time parse dates
df = pd.read_csv("TrainAndValid.csv",low_memory=False,parse_dates=["saledate"])
df.saledate.dtype


# In[9]:


df.saledate[:1000]


# In[10]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000],df["SalePrice"][:1000])


# In[11]:


df.head()


# In[12]:


df.head().T


# In[13]:


df.saledate.head(20)


# ## Sort DataFrame by saledate

# In[14]:


df.sort_values(by=["saledate"],inplace=True, ascending=True)
df.saledate.head(20)


# In[15]:


df.head()


# ### Make Copy of original data frame

# In[16]:


df_tmp = df.copy()


# ### Add datetime parameters for saledate column

# In[17]:


df_tmp[:1].saledate.dt.year


# In[18]:


df_tmp[:1].saledate.dt.day


# In[19]:


df_tmp[:1].saledate


# In[20]:


df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayOfWeek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayOfYear"] = df_tmp.saledate.dt.dayofyear


# In[21]:


#Dropping Saledate as we enriched our existing features
df_tmp.drop("saledate", axis=1, inplace=True)


# In[22]:


df_tmp.state.value_counts()


# ## Modelling
# model driven EDA

# In[23]:


from sklearn.ensemble import RandomForestRegressor

#model = RandomForestRegressor(n_jobs=-1,
#                             random_state=42)
#model.fit(df_tmp.drop("SalePrice", axis=1),df_tmp["SalePrice"])


# In[24]:


df.info()


# ## Convert String to categories
# 
# https://pandas.pydata.org/pandas-docs/version/1.4/reference/api/pandas.api.types.is_string_dtype.html

# In[25]:


pd.api.types.is_string_dtype(df_tmp["UsageBand"])


# In[26]:


#find the cols which contain strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)


# In[27]:


#random test code to explain 
random_dict = {"key1" : "hello",
               "key2" : "world"}

for key, value in random_dict.items():
    print(f"this is a key: {key}",
          f"this is a value: {value}")


# In[28]:


for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()


# In[29]:


df_tmp.info()


# In[30]:


df_tmp.state.cat.categories


# In[31]:


df_tmp.state.cat.codes


# In[32]:


df_tmp.isnull().sum()/len(df_tmp)


# ### Save preprocessed data

# In[33]:


df_tmp.to_csv("train_tmp.csv", index=False)


# In[34]:


df_tmp = pd.read_csv("train_tmp.csv",low_memory=False)
df_tmp.head().T


# ## Fill Missing Values
# 
# ### Fill Numerical Missing values first 

# In[35]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)


# In[36]:


df_tmp.ModelID


# In[37]:


#check for null columns
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[38]:


# Fill numeric rows with the median
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells us if the data was missing or not
            df_tmp[label+"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with median
            df_tmp[label] = content.fillna(content.median())


# In[39]:


#demonstrate how median is more robust than mean
hundreds = np.full((1000,),100)
hundreds_billion = np.append(hundreds,1000000000)
np.mean(hundreds),np.mean(hundreds_billion),np.median(hundreds),np.median(hundreds_billion)


# In[40]:


# Check if there's any null numeric values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# ###  Fill Categorical Missing values 

# In[41]:


#check for columns which not numerical
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[42]:


df.isna().sum()


# In[43]:


df_tmp.isna().sum()


# In[44]:


#check for columns which not numerical
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        #Add binary column to indicate whether sample had missing value 
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        #Tur categories into numbers and add+1
        df_tmp[label] = pd.Categorical(content).codes + 1 #missing is assigned neg 1 so + 1


# In[45]:


pd.Categorical(df_tmp["state"]).codes


# In[46]:


pd.Categorical(df_tmp["UsageBand"]).codes+1


# In[47]:


df_tmp.info()


# In[48]:


df_tmp.head().T


# In[49]:


df_tmp.isna().sum()[:10]


# ## FIT the MODEL

# In[50]:


get_ipython().run_cell_magic('time', '', '# Instantiate model \nmodel = RandomForestRegressor(n_jobs=-1,\n                             random_state=42)\n\n#Fit the model\nmodel.fit(df_tmp.drop("SalePrice",axis=1),df_tmp["SalePrice"])\n')


# In[51]:


#Score the model - Nonrelaiable - cause we are testing on train data
model.score(df_tmp.drop("SalePrice",axis=1),df_tmp["SalePrice"])


# ## Splitting data into train/validation set

# In[52]:


# We'll be working on train and valid sets


# In[53]:


df_tmp.saleYear


# In[54]:


df_tmp.saleYear.value_counts()


# In[55]:


# Split data into train and validation set
# Year 2012 - Valid set , before that is all train set

df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

len(df_val), len(df_train)


# In[56]:


# Split data into X & y
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# In[57]:


y_train


# ### Building an evaluation function 

# In[58]:


# Create evaluation function (the competition uses RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_preds):
    """
    Caculates root mean squared log error between predictions and
    true labels.
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate model on a few different levels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
              "Valid MAE": mean_absolute_error(y_valid, val_preds),
              "Training RMSLE": rmsle(y_train, train_preds),
              "Valid RMSLE": rmsle(y_valid, val_preds),
              "Training R^2": r2_score(y_train, train_preds),
              "Valid R^2": r2_score(y_valid, val_preds)}
    return scores


# ## Testing our model on subset (to tune the hyperparameters)

# In[59]:


# # this takes lots time 
#%%time
#model = RandomForestRegressor(n_jobs=-1,
#                             random_state=42)
#model.fit(X_train, y _train)


# In[60]:


#model.fit(X_train[:10000], y _train[:10000])


# In[61]:


# Change max_samples value
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              max_samples=10000)


# In[62]:


get_ipython().run_cell_magic('time', '', '# Cutting down on the max number of samples each estimator can see improves training time\nmodel.fit(X_train, y_train)\n')


# In[63]:


show_scores(model)


# ### Hyperparameter tuning with Randomized SeacrhCV

# In[64]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import RandomizedSearchCV\n\n#Different RandomForestRegressor hyperparameters\nrf_grid = {"n_estimators": np.arange(10, 100, 10),\n           "max_depth": [None, 3, 5, 10],\n           "min_samples_split": np.arange(2, 20, 2),\n           "min_samples_leaf": np.arange(1, 20, 2),\n           "max_features":[0.5, 1,"sqrt", "auto"],\n           "max_samples": [10000]}\n\nrs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,\n                                                   random_state=42),\n                             param_distributions=rf_grid,\n                             n_iter=2,\n                             cv=5,\n                             verbose=True)\n\nrs_model.fit(X_train,y_train)\n')


# In[65]:


# Find the best hyperparameters 
rs_model.best_params_


# In[66]:


# Evaluate the RansomizedSearch Model
show_scores(rs_model)


# ## Train a model with the best hyperparameters
# 
#  Note: These were found after 100 iterations of RandomizedSearchCV

# In[68]:


get_ipython().run_cell_magic('time', '', '\n#Most Ideal hyperparameters\nideal_model = RandomForestRegressor(n_estimators=40,\n                                    min_samples_leaf=1,\n                                    min_samples_split=14,\n                                    max_features=0.5,\n                                    n_jobs=-1,\n                                    max_samples=None,\n                                    random_state=42)\n\n#Fit the model\nideal_model.fit(X_train, y_train)\n')


# In[69]:


show_scores(ideal_model)


# ## Make predictions on test data

# In[101]:


#Import test data
df_test = pd.read_csv("Test.csv",
                     low_memory=False,
                     parse_dates=["saledate"])

df_test.head()


# In[71]:


#test data has NA , Non numerical values and different column size

#MAke prediction on test dataset
#test_preds = ideal_model.predict(df_test)


# ## Preprocessing test data

# In[102]:


def preprocess_data(df):
    """
    Performs transformations on df and 
    """
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear
    
    df.drop("saledate",axis=1,inplace=True)
    
    # Fill numeric rows with the median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing or not
                df[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                df[label] = content.fillna(content.median())
                    
        # Fill categorical with num codes
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            #we add +1
            df[label] = pd.Categorical(content).codes+1
        
    return df      


# In[103]:


#process test data
df_test = preprocess_data(df_test)
df_test.head()


# In[104]:


#Make predictions on updated test data 
#-> No of cols in train data set is 102 and test is 101
#-> So this throws error

#Find how the columns differ using sets
set(X_train.columns) - set(df_test.columns)


# In[105]:


#Manually adjust df_test to have autioneerigID_is_misisng column
df_test["auctioneerID_is_missing"] = False
df_test.head()


# In[106]:


pd.set_option('display.max_columns', None)
X_train.head()
X_train.columns.get_loc("auctioneerID_is_missing")


# In[112]:


#i had to do this cause tool was trowing error for aution.. being differently indexed for df_test and X_train
cols = list(df_test)
cols.insert(56,cols.pop(cols.index('auctioneerID_is_missing')))
cols
df_test=df_test.reindex(columns=cols)
df_test.columns.get_loc("auctioneerID_is_missing")


# In[113]:


test_preds = ideal_model.predict(df_test)


# In[114]:


test_preds


# In[115]:


#Format predictions into same format Kaggle wants
df_preds = pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds
df_preds


# In[116]:


#Export
df_preds.to_csv("test_predictions.csv",index=False)


# ## Feature Importance
# 
# Figure out which attributes of data were most important for predicting target value

# In[117]:


ideal_model.feature_importances_


# In[118]:


len(ideal_model.feature_importances_)


# In[120]:


X_train.shape


# In[127]:


#Helper function for plotting importance

def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features":columns,
                        "feature_importances": importances})
         .sort_values("feature_importances",ascending=False)
         .reset_index(drop=True))
    
    #plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n],df["feature_importances"][:20])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()


# In[128]:


plot_features(X_train.columns, ideal_model.feature_importances_)


# In[129]:


df["ProductSize"].value_counts()


# In[ ]:




