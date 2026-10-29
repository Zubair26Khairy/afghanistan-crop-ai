"""Reproduce the demonstration model registry from the bundled SYNTHETIC dataset."""
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor

BASE=Path(__file__).resolve().parent
DATA=BASE/'data/crops.csv'; OUT=BASE/'models'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(DATA)
features=['Year','Month','Province','Crop','Temp_C','Rain_mm','Fertilizer_kg_ha','Irrigation_Type','Land_Hectares']
num=['Year','Month','Temp_C','Rain_mm','Fertilizer_kg_ha','Land_Hectares']; cat=['Province','Crop','Irrigation_Type']
X=df[features]; y=df['Yield_Category']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
pre=ColumnTransformer([('numeric',SimpleImputer(strategy='median'),num),('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))]),cat)])
scaled=ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))]),cat)])
dense=ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
classifiers={
 'naive_bayes':('Gaussian Naive Bayes',dense,GaussianNB()),
 'linear_svm':('Linear SVM',scaled,LinearSVC(C=1,max_iter=20000,random_state=42)),
 'rbf_svm':('RBF SVM',scaled,SVC(kernel='rbf',C=10,gamma='scale')),
 'knn_31':('KNN (K=31)',scaled,KNeighborsClassifier(n_neighbors=31,n_jobs=-1)),
 'logistic_regression':('Logistic Regression',scaled,LogisticRegression(C=1,solver='lbfgs',max_iter=5000,random_state=42)),
 'decision_tree':('Decision Tree (Tuned)',pre,DecisionTreeClassifier(criterion='entropy',max_depth=10,min_samples_leaf=5,min_samples_split=2,random_state=42)),
 'random_forest_classifier':('Random Forest Classifier',pre,RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_leaf=5,random_state=42,n_jobs=-1)),
}
registry={}
for key,(name,prep,est) in classifiers.items():
    print('Training',name,flush=True)
    model=Pipeline([('preprocessing',prep),('classifier',est)])
    model.fit(Xtr,ytr)
    yp=model.predict(Xte)
    joblib.dump(model,OUT/(key+'.joblib'))
    registry[key]={'name':name,'task':'classification','accuracy':round(100*accuracy_score(yte,yp),2),'macro_f1':round(f1_score(yte,yp,average='macro'),4),'file':key+'.joblib'}

# Numeric regression with same stratified split for consistent hybrid evaluation.
tr_yield=df.loc[Xtr.index,'Yield_tons_per_hectare']; te_yield=df.loc[Xte.index,'Yield_tons_per_hectare']
regressors={
 'random_forest_regressor':('Random Forest Regression',pre,RandomForestRegressor(n_estimators=150,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1)),
 'linear_regression':('Linear Regression',pre,LinearRegression()),
 'polynomial_regression':('Polynomial Regression',pre,Pipeline([('poly',PolynomialFeatures(degree=2,include_bias=False)),('ridge',Ridge(alpha=100))])),
}
for key,(name,prep,est) in regressors.items():
    print('Training',name,flush=True)
    model=Pipeline([('preprocessing',prep),('regressor',est)])
    model.fit(Xtr,tr_yield)
    pred=model.predict(Xte)
    joblib.dump(model,OUT/(key+'.joblib'))
    registry[key]={'name':name,'task':'regression','mae':round(mean_absolute_error(te_yield,pred),4),'r2':round(r2_score(te_yield,pred),4),'file':key+'.joblib'}

# Hybrid class conversion uses thresholds derived from TRAINING data ONLY.
thresholds={}
for crop in sorted(df.Crop.unique()):
    sub=df.loc[Xtr.index]; sub=sub[sub.Crop==crop]
    lo=sub.loc[sub.Yield_Category=='Low','Yield_tons_per_hectare']; med=sub.loc[sub.Yield_Category=='Medium','Yield_tons_per_hectare']; hi=sub.loc[sub.Yield_Category=='High','Yield_tons_per_hectare']
    thresholds[crop]=[(float(lo.max())+float(med.min()))/2,(float(med.max())+float(hi.min()))/2]
def to_cat(crop,v):
    a,b=thresholds[crop]; return 'Low' if v<a else 'Medium' if v<b else 'High'
rf=joblib.load(OUT/'random_forest_regressor.joblib'); hyb=[to_cat(c,v) for c,v in zip(Xte.Crop,rf.predict(Xte))]
registry['rf_regression_category']={'name':'RF Regression → Category','task':'hybrid','accuracy':round(100*accuracy_score(yte,hyb),2),'macro_f1':round(f1_score(yte,hyb,average='macro'),4),'regressor_key':'random_forest_regressor'}
(OUT/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
(OUT/'registry.json').write_text(json.dumps(registry,indent=2))
print(json.dumps(registry,indent=2),flush=True)
