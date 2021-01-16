#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/Logos/organization_logo/organization_logo.png" width="300" alt="cognitiveclass.ai logo"  />
# </center>
# 
# # Analyzing Where Do People Drink?
# 
# Estimated time needed: **30** minutes
# 
# ## Objectives
# 
# After completing this lab you will be able to:
# 
# -   Be confident about your data analysis skills
# 

# This Dataset is from the story <a href=https://fivethirtyeight.com/features/dear-mona-followup-where-do-people-drink-the-most-beer-wine-and-spirits/>  Dear Mona Followup: Where Do People Drink The Most Beer, Wine And Spirits? </a>  The dataset contains Average serving sizes per person such as average wine, spirit, beer servings. As well as several other metrics. You will be asked to analyze the data and predict the total liters served given the servings. See how to share your lab at the end.
# 

# You will need the following libraries:
# 

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# <b>Importing the Data</b>
# 

# Load the csv:
# 

# In[43]:


df= pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/edx/project/drinks.csv')


# We use the method  <code>head()</code>  to display the first 5 columns of the dataframe:
# 

# In[44]:


df.head()


# <b>Question 1</b>:  Display the data types of each column using the attribute dtype.
# 

# In[45]:


df.dtypes


# <b>Question 2</b> Use the method <code>groupby</code> to get the number of wine servings per continent:
# 

# In[46]:


df_group_wine = df[['continent','wine_servings']]
df_group_wine = df_group_wine.groupby(['continent'], as_index=False).sum()
df_group_wine


# <b>Question 3:</b> Perform a statistical summary and analysis of beer servings for each continent:
# 

# In[47]:


df_group_beer = df[['continent','beer_servings']]
df_group_beer = df_group_beer.groupby(['continent'], as_index=False).describe()
df_group_beer


# <b>Question 4:</b> Use the function boxplot in the seaborn library to produce a plot that can be used to show the number of beer servings on each continent.
# 

# In[48]:


import seaborn as sns
sns.boxplot(x = "continent", y = 'beer_servings', data = df)
plt.show()


# <b>Question 5</b>: Use the function <code> regplot</code> in the seaborn library to determine if the number of wine servings is
# negatively or positively correlated with the number of beer servings.
# 

# In[49]:


import seaborn as sns
sns.regplot(x="wine_servings",y="beer_servings",data=df)


# In[50]:


df.corr()


# <b> Question 6:</b> Fit a linear regression model to predict the <code>'total_litres_of_pure_alcohol'</code> using the number of <code>'wine_servings'</code> then calculate $R^{2}$:
# 

# In[59]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
# independent variable
x = df[['wine_servings']]
# dependent variable
y = df[['total_litres_of_pure_alcohol']]
lm.fit(x,y)
lm.score(x,y)

#OutPut
print(("R^2 = "), lm.score(x,y))


# <br>
# <b>Note:</b> Please use <code>test_size = 0.10</code> and <code>random_state = 0</code> in the following questions.
# 

# <b>Question 7: </b>Use list of features to predict the <code>'total_litres_of_pure_alcohol'</code>, split the data into training and testing and determine the $R^2$ on the test data, using the provided code:
# 

# In[20]:


# listing all the features to deal with


features =["country", "beer_servings","wine_servings" ,"spirit_servings" ,"continent"]     


# In[66]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_data = df[['beer_servings','wine_servings','spirit_servings']]
y_data = df['total_litres_of_pure_alcohol']

# Train_Test_Split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.1, random_state=0)



lr = LinearRegression()
lr.fit(x_train, y_train)

#R^2 on both test and train data set
#OutPut
print("Train data set R^2:", lr.score(x_train, y_train))
print("Test data set R^2:", lr.score(x_test, y_test))


# <b>Question 8 :</b> Create a pipeline object that scales the data, performs a polynomial transform and fits a linear regression model. Fit the object using the training data in the question above, then calculate the R^2 using. the test data. Take a screenshot of your code and the $R^{2}$. There are some hints in the notebook:
# 

# <code>'scale'</code>
# 
# <code>'polynomial'</code>
# 
# <code>'model'</code>
# 
# The second element in the tuple contains the model constructor
# 
# <code>StandardScaler()</code>
# 
# <code>PolynomialFeatures(include_bias=False)</code>
# 
# <code>LinearRegression()</code>
# 

# In[67]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=0)

input =[('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False,degree=2)),('model',LinearRegression())]
pipe = Pipeline(input)
pipe.fit(x_train, y_train)
yhat = pipe.predict(x_data)

#OutPut
print("R^2 using Test data is", pipe.score(x_test, y_test))
print("R^2 using Training data is", pipe.score(x_train, y_train))


# <b>Question 9</b>: Create and fit a Ridge regression object using the training data, setting the regularization parameter to 0.1 and calculate the $R^{2}$ using the test data. Take a screenshot of your code and the $R^{2}$
# 

# In[68]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=0)

#Ridge Regression

RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)

#OutPut
print("Test R^2:",RidgeModel.score(x_test, y_test))
print("Train R^2:",RidgeModel.score(x_train, y_train))


# <b>Question 10 </b>: Perform a 2nd order polynomial transform on both the training data and testing data.  Create and fit a Ridge regression object using the training data, setting the regularization parameter to 0.1. Calculate the $R^{2}$ utilizing the test data provided. Take a screen-shot of your code and the $R^{2}$. 
# 

# In[69]:


pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel_pr = Ridge(alpha=0.1)
RidgeModel_pr.fit(x_train_pr, y_train)

#OutPut
print("RidgeModel Test data R^2: ",RidgeModel_pr.score(x_test_pr, y_test))
print("RidgeModel Train data R^2: ",RidgeModel_pr.score(x_train_pr, y_train))


# <a href="https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/share-notebooks.html\" > CLICK HERE </a>  to see how to share your notebook
# 

# <b>Sources</b>
# 

# <a href=https://fivethirtyeight.com/features/dear-mona-followup-where-do-people-drink-the-most-beer-wine-and-spirits/> Dear Mona Followup: Where Do People Drink The Most Beer, Wine And Spirits?</a> by By Mona Chalabi , you can download the dataset <a href=https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption>here</a>.
# 

# ### Thank you for completing this lab!
# 
# ## Author
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/" target="_blank">Joseph Santarcangelo</a>
# 
# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By | Change Description                 |
# | ----------------- | ------- | ---------- | ---------------------------------- |
# | 2020-08-27        | 2.0     | Lavanya    | Moved lab to course repo in GitLab |
# 
# <hr>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 
