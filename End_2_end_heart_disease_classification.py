#!/usr/bin/env python
# coding: utf-8

# # Predict heart disease using ML
# 
# 1. Problem Defination
# 2. Data
# 3. Evaluation 
# 4. Features
# 5. Modelling
# 6. Experimentation
# 

# ## 1. Problem Definination
# 
# > Given clinical parameters , can we predict , heart disease or not
# 
# ## 2. Data
# 
# The original data is from Cleaveland data from UCI Machine Learning Reposititory.
# https://archive.ics.uci.edu/dataset/45/heart+disease
# 
# ## 3. Evaluation 
# 
# If we reach 95% accuracy  during proof of conceot , we'll pursue.
# 
# ## 4. Features
# 
# Only 14 attributes used:
#       1. #3  (age)       
#       2. #4  (sex)       
#       3. #9  (cp)        
#       4. #10 (trestbps)  
#       5. #12 (chol)      
#       6. #16 (fbs)       
#       7. #19 (restecg)   
#       8. #32 (thalach)   
#       9. #38 (exang)     
#       10. #40 (oldpeak)   
#       11. #41 (slope)     
#       12. #44 (ca)        
#       13. #51 (thal)      
#       14. #58 (num)       (the predicted attribute)
#       
#  age - age in years
# sex - (1 = male; 0 = female)
# cp - chest pain type
# 0: Typical angina: chest pain related decrease blood supply to the heart
# 1: Atypical angina: chest pain not related to heart
# 2: Non-anginal pain: typically esophageal spasms (non heart related)
# 3: Asymptomatic: chest pain not showing signs of disease
# trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
# chol - serum cholestoral in mg/dl
# serum = LDL + HDL + .2 * triglycerides
# above 200 is cause for concern
# fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# '>126' mg/dL signals diabetes
# restecg - resting electrocardiographic results
# 0: Nothing to note
# 1: ST-T Wave abnormality
# can range from mild symptoms to severe problems
# signals non-normal heart beat
# 2: Possible or definite left ventricular hypertrophy
# Enlarged heart's main pumping chamber
# thalach - maximum heart rate achieved
# exang - exercise induced angina (1 = yes; 0 = no)
# oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
# slope - the slope of the peak exercise ST segment
# 0: Upsloping: better heart rate with excercise (uncommon)
# 1: Flatsloping: minimal change (typical healthy heart)
# 2: Downslopins: signs of unhealthy heart
# ca - number of major vessels (0-3) colored by flourosopy
# colored vessel means the doctor can see the blood passing through
# the more blood movement the better (no clots)
# thal - thalium stress result
# 1,3: normal
# 6: fixed defect: used to be defect but ok now
# 7: reversable defect: no proper blood movement when excercising
# target - have disease or not (1=yes, 0=no) (= the predicted attribute)
# 
# 

# In[1]:


import sklearn
sklearn.__version__


# ## Prepping tools
# 
# We will use Pandas Matplotlib Numpy for data analysis and manipulation

# In[2]:


#Import all tools 
#Regular EDA (Explanantory Data Analysis) and plotting libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
#For making plots appear inside notebook 

#Models from Scikit Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Model evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay


# ## Load Data 

# In[3]:


df = pd.read_csv("heart-disease.csv")
df 


# In[4]:


df.shape #(rows,columns)


# ## Data Exploration (EDA)
# 
# 1. Problem Statement
# 2. Data type
# 3. Missing data 
# 4. Outliers : outlandish data point
# 5. Add , Change , Remove 
# 
# 

# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df["target"].value_counts()  #shows no of unique data


# In[8]:


df["target"].value_counts().plot(kind="bar", color=["salmon","lightblue"]);


# In[9]:


df.info()


# In[10]:


df.isna().sum() #check missing data


# In[11]:


df.describe() #numerical details about features


# ### Heart Disease Frequency according to Sex

# In[12]:


df.sex.value_counts()


# In[13]:


# Compare target column with sex column
pd.crosstab(df.target, df.sex)


# In[14]:


# Create a plot of crosstab
pd.crosstab(df.target,df.sex).plot(kind="bar",
                                  figsize=(10,6), ##?? TODO : why ?
                                  color=["salmon","lightblue"]);

plt.title("Heart Diesease Frequency of Sex")
plt.xlabel("0 = No Dieasese, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female","Male"]);
plt.xticks(rotation=0);


# In[15]:


df["thalach"].value_counts()


# ## Age vs. Max Heart Rate for Heart Disease

# In[16]:


#Create another figure
plt.figure(figsize=(10,6))

#Scatter with positive examples
plt.scatter(df.age[df.target==1],
           df.thalach[df.target==1],
           c="salmon");

#Scatter with negetive examples
plt.scatter(df.age[df.target==0],
           df.thalach[df.target==0],
           c="lightblue");
plt.title("Heart Diesease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease","No Disease"]);


# In[17]:


# Check the distribution of the age column with histogram
df.age.plot.hist();


# ## Heart Disease Frequency per chest pain type

# cp - chest pain type 
# 0: Typical angina: chest pain related decrease blood supply to the heart 
# 1: Atypical angina: chest pain not related to heart 
# 2: Non-anginal pain: typically esophageal spasms (non heart related) 
# 3: Asymptomatic: chest pain not showing signs of disease

# In[18]:


pd.crosstab(df.cp,df.target)


# In[19]:


pd.crosstab(df.cp,df.target).plot(kind="bar",
                                 figsize=(10,6),
                                 color=["salmon","lightblue"])

plt.title("Heart Disease Frequency per chest pain type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No Disease","Disease"])
plt.xticks(rotation=0); #TODO what is this ?


# In[20]:


#Correlation Matrix 

df.corr()


# In[21]:


corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,
                annot=True,
                linewidth=0.5,
                fmt=".2f",
                cmap="YlGnBu");


# In[22]:


#Split data into X and y
X = df.drop("target", axis=1)
y = df["target"]

np.random.seed(42)

#split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# We are going to try 3 diff models:
# 1. Logistic Regression
# 2. K Nearest Neighbours
# 3. Random Forest Classifier

# In[23]:


models = {"Logistic Regression" : LogisticRegression(),
         "KNN": KNeighborsClassifier(),
         "Random Forest" : RandomForestClassifier()}

#Create a function to fit and score models
def fit_and_acore(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates ML models
    models : dict of Scikit Learn ML models
    X_train : training data (no labels)
    X_test : testing data (no_labels)
    y_train : traning labels
    y_test : test labels
    """
    
    #set random seed
    np.random.seed(42)
    #make dict to keep model scores
    model_scores = {}
    #Loop through models
    for name, model in models.items():
        #Fit model to data
        model.fit(X_train, y_train)
        #Evaluate
        model_scores[name] = model.score(X_test, y_test)
    return model_scores    


# In[24]:


model_scores = fit_and_acore(models=models,
                            X_train= X_train,
                            X_test= X_test,
                            y_train= y_train,
                            y_test= y_test)

model_scores


# # Model Comparison

# In[25]:


model_compare = pd.DataFrame(model_scores,index=["accuracy"])
model_compare.T.plot.bar();


# We'll explore the below :
# * Hyperparameter tuning
# * Feature importance 
# * Confusion Matrix
# * Cross Validation 
# * Precision 
# * Recall
# * F1 Score
# * Classification Report
# * ROC Curve
# * Area Under Curve 
# 
# ## Hyperparameter tuning
# 

# In[26]:


#Tune KNN

train_scores = []
test_scores = []

neighbors = range(1,21)

knn = KNeighborsClassifier()

for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    #Fit the algo
    knn.fit(X_train, y_train)
    
    train_scores.append(knn.score(X_train, y_train))
    
    test_scores.append(knn.score(X_test, y_test))


# In[27]:


train_scores


# In[28]:


test_scores


# In[29]:


plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1,21,1))
plt.xlabel("Numner of neighbors")
plt.ylabel("Model Score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# ## Hyperparameter tuning for RandomizedSearchCV

# #Tuning :
# LogisticRegression()
# RandomForestClassifier()
# 
# using RandomizedSearchCV

# In[30]:


#For Logistic Regression

log_reg_grid = {"C":np.logspace(-4, 4, 20),
               "solver": ["liblinear"]}


# In[31]:


np.logspace(-4, 4,20)


# In[32]:


#Create a hyperparameter grid for RandomForestClassifier

rf_grid = {"n_estimators": np.arange(10,1000,50),
          "max_depth": [None, 3, 5, 10],
          "min_samples_split": np.arange(2, 20, 2),
          "min_samples_leaf": np.arange(1, 20, 2)}


# In[33]:


#Tune Logistic Regression Model 

np.random.seed(42)

#random hyperparameter search for Logistic Regression

rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                               param_distributions=log_reg_grid,
                               cv=5,
                               n_iter=20,
                               verbose=True)

#Fit random hyperparameter sarch model for Logistic Regression

rs_log_reg.fit(X_train,y_train)



# In[34]:


rs_log_reg.best_params_


# In[35]:


rs_log_reg.score(X_test,y_test)


# Now turning RandomForest Classifier

# In[36]:


np.random.seed(42)

#setup random hyperarameter

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                          verbose=True)

rs_rf.fit(X_train,y_train)


# In[37]:


rs_rf.best_params_


# In[38]:


rs_rf.score(X_test, y_test)


# In[39]:


model_scores


# # Hyperparameter Tuning with GridSearchCV
# 
# Since LogisticRegression is best here so tune it using GridSearchCV

# In[40]:


log_reg_grid = {"C":np.logspace(-4, 4, 30),
               "solver":["liblinear"]}

gs_log_reg = GridSearchCV(LogisticRegression(),
                         param_grid=log_reg_grid,
                         cv=5,
                         verbose=True)

gs_log_reg.fit(X_train,y_train)


# In[41]:


#Check best hyperparameter

gs_log_reg.best_params_


# In[54]:


#Evaluate gridsearch Logistic Regression
gs_log_reg.score(X_test, y_test)


# # Evaluating Models

# * ROC Curve and AUC score
# * Confusion Matrix
# * Classification Report
# * Precision
# * Recall
# * F1-Score
# 
# we want to use cross-validation where ever applicable

# In[42]:


y_preds = gs_log_reg.predict(X_test)


# In[43]:


y_preds


# In[44]:


y_test


# In[48]:


# ROC Curve and AUC Score
#Plot ROC cruve and calculate AUC Metric

#THIS WONT WORK # plot_roc_curve(gs_log_reg, X_test, y_test)
# Chat GPT helped here :
roc_display = RocCurveDisplay.from_estimator(gs_log_reg, X_test, y_test)
#roc_display.plot()


# In[49]:


# COnfusion Matrix

print(confusion_matrix(y_test, y_preds))


# In[52]:


sns


# In[58]:


# usinf seaborn for better visualization 
#sns.set(font_size=1.5)  <- didnt work again Chat GPT helped

sns.set_context("notebook", font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    """
    Plots a nice looking conf matrix using Seaborn's heatmap
    """
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test,y_preds),
                    annot=True,
                    cbar=False)
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
plot_conf_mat(y_test, y_preds)  


# Now Let's get classification Report and cross validated , precision , recall and F1 Score

# In[59]:


print(classification_report(y_test, y_preds))


# ### Calculate evaluation metrics using cross validation

# In[62]:


#Cross_val_score
#Check best hyper parameter
gs_log_reg.best_params_


# In[63]:


#Create new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418, solver="liblinear")


# In[64]:


#cross-validated accuracy
cv_acc = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring="accuracy")
cv_acc


# In[65]:


cv_acc = np.mean(cv_acc)
cv_acc


# In[66]:


#Cross-validated precision
cv_precision = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring="precision")
cv_precision = np.mean(cv_precision)
cv_precision


# In[67]:


#Cross-validated recall
cv_recall = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring="recall")
cv_recall = np.mean(cv_recall)
cv_recall


# In[68]:


#Cross-validated f1 score
cv_f1 = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring="f1")
cv_f1 = np.mean(cv_f1)
cv_f1


# In[71]:


#Visualize cross validated metrics

cv_metrics = pd.DataFrame({"Accuracy":cv_acc,
                          "Precision": cv_precision,
                          "Recall": cv_recall,
                          "F1":cv_f1},
                         index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics",legend= False)


# ### Feature Importance 
# 
# Which features contribued most to the outcomes of the model

# In[73]:


clf = LogisticRegression(C=0.20433597178569418, solver="liblinear")

clf.fit(X_train,y_train);


# In[74]:


clf.coef_


# In[75]:


#Match coef of features to columns

feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# In[76]:


# Visualize feature importance
feature_df = pd.DataFrame(feature_dict,index=[0])
feature_df.T.plot.bar(title="Feature Importamce",legend=False)


# In[77]:


pd.crosstab(df["sex"],df["target"])


# In[78]:


pd.crosstab(df["slope"],df["target"])


# In[ ]:




