#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df = pd.read_csv(r"C:\Users\tejad\Downloads\Housing.csv")
df.head()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.dtypes


# In[9]:


df['mainroad'].value_counts()
# yes    468
# no      77
# after Label Encoding
# 1      468
# 0      77


# In[10]:


df['guestroom'].value_counts()


# In[11]:


df['hotwaterheating'].value_counts()


# In[12]:


df['airconditioning'].value_counts()


# In[13]:


df['basement'].value_counts()


# In[14]:


df['prefarea'].value_counts()


# In[16]:


df['furnishingstatus'].value_counts()


# In[17]:


from sklearn.preprocessing import LabelEncoder


# In[18]:


cat_cols = [i for i in df.columns if df[i].dtypes=='object']
print(cat_cols)


# In[19]:


lb = LabelEncoder()
for i in cat_cols:
    df[i] = lb.fit_transform(df[i])


# In[20]:


df.dtypes


# In[21]:


df['mainroad'].value_counts()


# In[22]:


df['guestroom'].value_counts()


# In[23]:


df['basement'].value_counts()


# In[24]:


df['airconditioning'].value_counts()


# In[25]:


df['furnishingstatus'].value_counts()


# In[26]:


df['prefarea'].value_counts()


# In[27]:


df['hotwaterheating'].value_counts()


# In[28]:


plt.boxplot(df['price'])
plt.show()


# In[29]:


plt.hist(df['price'])
plt.show()


# In[30]:


df.columns


# ### Correlation
# 1) It gives the strength and direction of linear relationship between 2 or more variables.<br>
# 2) Correlation varies between -1 and +1.<br>
# 3) Positive Correlation between x and y implies<br>
#      
#        a) if x increases => y also increases
#        b) if x decreases => y also decreases
#        
# 4) Negative Correlation between x and y implies<br>
# 
#        a) if x increases => y decreases
#        b) if x decreases => y increases
#        
# 5) Corr(x,y) = (Sum((xi-xmean) * (yi-ymean))/(sqrt(sum(xi-xmean)^2) * sum(yi-ymean)^2))
# 6) Note<br>
# corr(x,y) = corr(y,x)<br>
# corr(x,x) = corr(y,y) = 1<br>

# In[33]:


corr = df.corr()
corr


# In[34]:


plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True,cmap='RdBu')
plt.show()


# In[35]:


corr['price'].sort_values(ascending=False)


# In[36]:


x = df.iloc[:,1:-1]
y = df['price']
print(x.columns)


# In[37]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# #### Build the Model

# In[40]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[41]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[42]:


res = pd.DataFrame(columns=['Model','MAE','MSE','RMSE','R2_Score'])
res.head()


# In[43]:


def gen_metrics(y_test,ypred):
    mae = mean_absolute_error(y_test,ypred)
    mse = mean_squared_error(y_test,ypred)
    rmse = np.sqrt(mean_squared_error(y_test,ypred))
    r2s = r2_score(y_test,ypred)
    return mae,mse,rmse,r2s


# ### LinearRegression

# In[44]:


m1 = LinearRegression()
m1.fit(x_train,y_train)


# In[45]:


# R2 score
print('Training score',m1.score(x_train,y_train))
print('Testing score',m1.score(x_test,y_test))


# In[46]:


ypred_m1 = m1.predict(x_test)


# In[47]:


mae,mse,rmse,r2s = gen_metrics(y_test,ypred_m1)
print('MAE',mae)
print('MSE',mse)
print('RMSE',rmse)
print('R2_Score',r2s)


# In[48]:


res.head()


# In[56]:


m1p = {"Model":"Linear Reg","MAE":mae,"MSE":mse,"RMSE":rmse,"R2_Score":r2s}
r1 = pd.DataFrame(m1p,index=[0])
r1

