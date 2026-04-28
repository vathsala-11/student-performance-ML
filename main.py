import pandas as pd
df=pd.read_csv(r"C:\Users\VATHSALA M D\Downloads\archive (3)\student_data.csv")
print(df.head())
print(df.columns)

def convert_grade(x):
    if x>=15:
        return "High"
    elif x>=10:
        return "Medium"
    else:
        return "Low"
df["performance"]=df["G3"].apply(convert_grade)

print(df[["G3","performance"]].head())

df=df.drop("G3",axis=1)
x=df.drop("performance",axis=1)
y=df["performance"]

y=y.map({"Low":0,"Medium":1,"High":2})
x=pd.get_dummies(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


import pandas as pd

importance = model.feature_importances_
feature_names = x.columns

feat_imp = pd.Series(importance, index=feature_names)
print(feat_imp.sort_values(ascending=False).head(10))

print("columns used in the model:")
print(list(x.columns))

import pickle

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save column names
pickle.dump(x.columns, open("columns.pkl", "wb"))


