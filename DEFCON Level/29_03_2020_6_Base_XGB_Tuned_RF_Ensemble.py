import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

infile=r'C:\Users\718728\Desktop\Old_Bkp\PraveenT\PycharmProjects\Trial\Hackerearth\DEFCON\train.csv'
testfile=r'C:\Users\718728\Desktop\Old_Bkp\PraveenT\PycharmProjects\Trial\Hackerearth\DEFCON\test.csv'
sample=r'C:\Users\718728\Desktop\Old_Bkp\PraveenT\PycharmProjects\Trial\Hackerearth\DEFCON\sample_submission.csv'
outfile=r'C:\Users\718728\Desktop\Old_Bkp\PraveenT\PycharmProjects\Trial\Hackerearth\DEFCON\output.csv'
compfile=r'C:\Users\718728\Desktop\Old_Bkp\PraveenT\PycharmProjects\Trial\Hackerearth\DEFCON\comp.csv'
trialfile=r'C:\Users\718728\Desktop\Old_Bkp\PraveenT\PycharmProjects\Trial\Hackerearth\DEFCON\trial.csv'
teststack=r'C:\Users\718728\Desktop\Old_Bkp\PraveenT\PycharmProjects\Trial\Hackerearth\DEFCON\teststack.csv'


df=pd.read_csv(infile)

df['Diplomatic_Meetings_Set']=df['Diplomatic_Meetings_Set'].astype(str)
df['Aircraft_Carriers_Responding']=df['Aircraft_Carriers_Responding'].astype(str)

df1=pd.get_dummies(df['Diplomatic_Meetings_Set'],prefix='Diplomatic_Meetings_Set')
df1=df1.drop('Diplomatic_Meetings_Set_2',axis=1)
df2=pd.get_dummies(df['Aircraft_Carriers_Responding'],prefix='Aircraft_Carriers_Responding')

df=df.drop(['Diplomatic_Meetings_Set','Aircraft_Carriers_Responding'],axis=1)
df=pd.concat([df,df1,df2],axis=1)

df['Percent_Of_Forces_Mobilized']=df['Percent_Of_Forces_Mobilized']*100
scaler=MinMaxScaler(feature_range=(-3,3))
df['Active_Threats']=scaler.fit_transform(df['Active_Threats'].values.reshape(-1,1))
df['Inactive_Threats']=scaler.fit_transform(df['Inactive_Threats'].values.reshape(-1,1))

X=df.drop('DEFCON_Level',axis=1).values
y=df['DEFCON_Level'].values


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=20,test_size=0.2,stratify=y)

model1=XGBClassifier(min_samples_split=2,max_depth=10,min_samples_leaf=5,learning_rate=0.1)
model1.fit(X_train,y_train)
pred1=model1.predict(X_test)
print(f1_score(y_test,pred1,average='weighted'))

model2=RandomForestClassifier()
model2.fit(X_train,y_train)
pred2=model2.predict(X_test)
print(f1_score(y_test,pred2,average='weighted'))

v=VotingClassifier(estimators=[('xgb',model1),('rfp',model2)],voting='soft')
v.fit(X_train,y_train)
pred=v.predict(X_test)

mod=pd.read_csv(compfile)
mod['y_test']=y_test
mod['xgb']=pred1
mod['rfc']=pred2
mod['voting']=pred
mod.to_csv(trialfile,index=False)
print('F1 Score before Stack : '+str(f1_score(y_test,pred,average='weighted')))

stk=pd.read_csv(trialfile)

X_stk=stk.iloc[:,1:2]
y_stk=stk.iloc[:,0]


xgb=XGBClassifier()
xgb.fit(X_stk,y_stk)
xgb.predict(X_stk)
pred_stk=np.round(xgb.predict(X_stk),0)

print('F1 Score after Stack : '+str(f1_score(y_test,pred_stk,average='weighted')))

test=pd.read_csv(testfile)

test['Diplomatic_Meetings_Set']=test['Diplomatic_Meetings_Set'].astype(str)
test['Aircraft_Carriers_Responding']=test['Aircraft_Carriers_Responding'].astype(str)

test1=pd.get_dummies(test['Diplomatic_Meetings_Set'],prefix='Diplomatic_Meetings_Set')
test1=test1.drop('Diplomatic_Meetings_Set_2',axis=1)
test2=pd.get_dummies(test['Aircraft_Carriers_Responding'],prefix='Aircraft_Carriers_Responding')

test=test.drop(['Diplomatic_Meetings_Set','Aircraft_Carriers_Responding'],axis=1)
test=pd.concat([test,test1,test2],axis=1)

test['Percent_Of_Forces_Mobilized']=test['Percent_Of_Forces_Mobilized']*100
scaler=MinMaxScaler(feature_range=(-3,3))
test['Active_Threats']=scaler.fit_transform(test['Active_Threats'].values.reshape(-1,1))
test['Inactive_Threats']=scaler.fit_transform(test['Inactive_Threats'].values.reshape(-1,1))


test_X=test.values
test_xgb_pred=model1.predict(test_X)
test_rf_pred=model2.predict(test_X)
test_v_pred=v.predict(test_X)

mod1=pd.read_csv(compfile)
mod1['xgb']=test_xgb_pred
mod1['rf']=test_rf_pred
mod1['voting']=test_v_pred

test_X_stk=mod1.iloc[:,1:2]

test_pred_stk=np.round(xgb.predict(test_X_stk),0)

submis=pd.read_csv(testfile,usecols=['ID'])
submis['ID']=test['ID']
submis['DEFCON_Level']=test_pred_stk
submis.to_csv(outfile,index=False)