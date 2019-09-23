# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:54:56 2019

@author: vkovvuru
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor


train=pd.read_csv('train.csv')

test=pd.read_csv('test.csv')

meal_info=pd.read_csv('meal_info.csv')

center_info=pd.read_csv('fulfilment_center_info.csv')

df1=pd.merge(train,meal_info, on='meal_id')

df=pd.merge(df1,center_info,on='center_id')

label_encode_columns = [
                        'center_type', 
                        'category', 
                        'cuisine']

le = LabelEncoder()

for col in label_encode_columns:
    le.fit(df[col])
    df[col + '_encoded'] = le.transform(df[col])
    
df_train=df.drop(['center_type','category','cuisine'],axis=1)
    
x=df_train.drop(['num_orders'],axis=1)  
  
y=df_train['num_orders']

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.15, random_state=123)

rf = RandomForestRegressor(n_estimators=200)

# fit rf_model_on_full_data on all data from the 
rf.fit(X_train, y_train)

pred=rf.predict(X_test)
log_error=mean_squared_log_error(y_test, pred)
print("Mean_Squared_Log_error: ", log_error )


df1_test=pd.merge(test,meal_info, on='meal_id')

df_test=pd.merge(df1_test,center_info,on='center_id')    
    
      
label_encode_columns = [
                        'center_type', 
                        'category', 
                        'cuisine']

le = LabelEncoder()

for col in label_encode_columns:
    le.fit(df_test[col])
    df_test[col + '_encoded'] = le.transform(df_test[col])

   
df_test=df_test.drop(['center_type','category','cuisine'],axis=1)
    
out=rf.predict(df_test)   
    
output = pd.DataFrame({'id': df_test.id,
                       'num_orders': out})

output.to_csv('submission_rf.csv', index=False)    
    




















