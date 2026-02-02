#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The data is available at [https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv](https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv)

# To complete this assignment, your code will need to contain the following:

# - One valid model. This can be one model of any kind covered in class to this point (see list above). For your model, be sure that you structure the model code as follows:

#     - A forecasting algorithm named `model` using the implementation of one of the four models covered in weeks 1 to 3 (don't use other libraries, since I can't keep track of all of them). This model will use the number of trips in an hour as the dependent variable, and may or may not use exogenous variables from the remainder of the dataset.
#     - A fitted model named `modelFit`. This should be a model instance capable of generating forecasts by incorporating new data in the same shape as the data used in part (1).
#     - A vector of forecasts using the data from the test period named `pred`. You should predict each hour in January of the year following our training data (for 744 total predicted hours).


# In[2]:


import pandas as pd 
from prophet import Prophet


# In[3]:


data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")


# In[4]:


data.head()


# In[5]:


data_p = data[['Timestamp', 'trips']]
data_p.columns = ['ds', 'y']
data_p.head()


# In[7]:


model = Prophet()
modelFit = model.fit(data_p)


# In[8]:


# future = m.make_future_dataframe(periods=365)
# forecast = m.predict(future)

# forecast.head()


# In[9]:


test = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")


# In[10]:


test.head()


# In[11]:


test_p = test[['Timestamp']]


# In[12]:


test_p.head()


# In[13]:


test_p.columns=['ds']


# In[14]:


forecast = modelFit.predict(test_p)


# In[15]:


pred = forecast['yhat'].values


# In[16]:


forecast['yhat'].head()


# In[17]:


print({len(forecast)})


# In[ ]:




