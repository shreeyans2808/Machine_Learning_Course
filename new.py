import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #helps in splitting dataset into train et and test set using stratify and ratio
from sklearn.impute import SimpleImputer #helps in substituting value in cells with numm values using strategy
from sklearn.preprocessing import OrdinalEncoder #this makes an array of all cases in the text and assigns a number
from sklearn.preprocessing import OneHotEncoder #this makes a boolean by categorising multiple cases to one
from sklearn.preprocessing import FunctionTransformer #to create custom transformers
from sklearn.linear_model import LinearRegression

#extracting data
housing=pd.read_csv("housing.csv")

#finding train set and test set using cases and ratio
housing["house_income"]=pd.cut(housing["median_income"],bins=[0.,1.5,3,4.5,6.,np.inf],labels=[1,2,3,4,5])
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42,stratify=housing["house_income"])

#plotting graph and checking correlation
housing=train_set.copy()
#housing.plot(kind="scatter",x="longitude",y="latitude",grid=True,s=housing["population"]/100,c="median_house_value",colorbar=True,cmap="jet")
housing["people_in_house"]=housing["population"]/housing["households"]
housing["bedroom_ratio"]=housing["total_bedrooms"]/housing["total_rooms"]
corrm=housing.corr(numeric_only=True)
#print(corrm["median_house_value"].sort_values(ascending=False))

#substituing null values to median using imputer
imputer=SimpleImputer(strategy="median")
housing_num=housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
x=imputer.transform(housing_num)
housing_tr=pd.DataFrame(x,columns=housing_num.columns,index=housing_num.index)

#converting text formats to boolean values for machine to accept it
housing_cat=housing[["ocean_proximity"]]
cat_encoder=OneHotEncoder(sparse_output=False)
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
#print(housing_cat_1hot)
#b=housing["population"].apply(np.log).hist(bins=50,grid=True)
#plt.ylim(0,4000)

#custom transformer
log_transformer=FunctionTransformer(np.log,inverse_func=np.exp)
log_pop=log_transformer.transform(housing["population"])
#log_pop.hist(bins=50,grid=True)
model=LinearRegression()
model.fit(housing[["median_income"]],housing)
new_data=housing[["median_income"]].iloc[:5]
predict=model.predict(new_data)
print(predict)







