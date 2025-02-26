#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
url='https://raw.githubusercontent.com/Fxisxl/Two-Wheeler-Price-Prediction/refs/heads/main/CleanedDataNew.csv'
df=pd.read_csv(url)
df.head()


# In[2]:


df=df.drop('Unnamed: 0',axis=1)
df=df.drop('Owner_Type',axis=1)


# In[3]:


df


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

x=df.drop(columns=['price'],errors='ignore')
y=df['price']

categorical_variable=['bike_name','brand']
numerical_variable=['kms_driven','age']

categorical_transform=OneHotEncoder(handle_unknown='ignore')
numerical_transform=StandardScaler()

transforms=[('num',numerical_transform,numerical_variable),
                                         ('cat',categorical_transform,categorical_variable)]

preprocessor=ColumnTransformer(transforms)


model=Pipeline(steps=[('preprocessor',preprocessor),
                     ('Linear_regression',LinearRegression())])


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(model)


# In[5]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[6]:


model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[7]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))


# In[8]:


print(mae,mse,rmse)


# In[9]:


df.head()


# In[10]:


# bike_name=str(input("enter bike name to be sold : "))
# kms_driven=eval(input("enter the kms driven : "))
# age=eval(input("enter age of bike : "))
# brand=str(input("enter brand of bike : "))

# new_data = pd.DataFrame({
#     'bike_name': [bike_name],
#     'kms_driven': [kms_driven],
#     'age': [age],
#     'brand':[brand]
# })

# print(new_data)

# predicted_data=model.predict(new_data)
# print(predicted_data)
    


# In[11]:


y_pred.shape


# In[12]:


x_train


# In[13]:


y_train


# In[14]:


x_test


# In[15]:


y_test


# In[17]:


y_pred.shape


# In[ ]:





# In[ ]:





# In[ ]:




