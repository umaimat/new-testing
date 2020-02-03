#!/usr/bin/env python
# coding: utf-8

# # Loading Libraries and Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


dataset = pd.read_csv('student_scores.csv')


# # EDA

# In[3]:


dataset.shape


# In[4]:


dataset.head()


# In[5]:


dataset.describe()


# In[6]:


dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# # Preparing the data

# In[18]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
X
y


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Training the Algorithm

# In[9]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[10]:


print(regressor.intercept_)


# In[11]:


# Coefficient of X
print(regressor.coef_)


# This means that for every one unit of change in hours studied, the change in the score is about 9.91%. Or in simpler words, if a student studies one hour more than they previously studied for an exam, they can expect to achieve an increase of 9.91% in the score achieved by the student previously.

# # Making Predictions

# In[12]:


y_pred = regressor.predict(X_test)
y_test
y_pred


# In[13]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# # Evaluating the Algorithm

# In[14]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# You can see that the value of root mean squared error is 4.64, which is less than 10% of the mean value of the percentages of all the students i.e. 51.48. This means that our algorithm did a decent job.

# In[15]:


regressor.score(X_train, y_train)


# 95.14% accuracy

# # Plot our results

# In[16]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# In[ ]:




