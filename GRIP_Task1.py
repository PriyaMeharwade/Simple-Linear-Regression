#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Spark Foundation
# 

# # Data science and Business Analytics Intern

# ## Author: Priya Meharwade

# ## Task1- Prediction using Supervised Machine Learning

# In this task we have to predict the percentage score of a student based on number of hours studied,This task has two variables where the feature is no.of hours studied and target value is percentage score, this can be solved using Simple linear Regression model

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


# Reading data from remote link
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
df


# In[28]:


df.head()


# In[4]:


df.shape


# In[5]:


#to check the null values
df.info()


# In[6]:


#to check mean,median,count
df.describe()


# In[29]:


#data visualization
df.plot(x='Hours', y='Scores', kind='scatter')


# In[8]:


#In this graph we can see there is positive linear relationship between x and y vaiables


# In[12]:


#splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Hours, df.Scores, test_size=0.2, random_state=40)


# In[13]:


#Visualizing train and test data set

plt.scatter(X_train, y_train, label =  'Training_Data', color='b')
plt.scatter(X_test, y_test, label = 'Testing_Data', color='r')
plt.legend()
plt.title("Model_Visualization")
plt.show()


# In[19]:


#using Linear Regression Model

from sklearn.linear_model import LinearRegression
Lr = LinearRegression()
Lr.fit(X_train.values.reshape(-1,1),y_train.values)


# In[20]:


#predicting for test dataset
pred = Lr.predict(X_test.values.reshape(-1,1))
pred


# In[24]:


pred1 = pd.DataFrame(pred)
pred1


# In[25]:


#plotting on test data
plt.plot(X_test, pred, label='LinearRegression', color='b')
plt.scatter(X_test, y_test, label= 'Test_data', color='r')
plt.legend()
plt.show()


# In[26]:


Lr.predict([[9.25]])


# In[32]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, pred1)) 


# In[ ]:




