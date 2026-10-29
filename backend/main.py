from functools import lru_cache
from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE=Path(__file__).resolve().parent
META=json.loads((BASE/'models/metadata.json').read_text())
REGISTRY=json.loads((BASE/'models/registry.json').read_text())
THRESHOLDS=json.loads((BASE/'models/thresholds.json').read_text())
FEATURES=['Year','Month','Province','Crop','Temp_C','Rain_mm','Fertilizer_kg_ha','Irrigation_Type','Land_Hectares']
app=FastAPI(title='Afghanistan Crop AI API',version='2.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST'],allow_headers=['*'])

class Input(BaseModel):
    Year:int=Field(ge=2010,le=2040)
    Month:int=Field(ge=1,le=12)
    Province:str
    Crop:str
    Temp_C:float=Field(ge=-30,le=65)
    Rain_mm:float=Field(ge=0,le=3000)
    Fertilizer_kg_ha:float=Field(ge=0,le=1000)
    Irrigation_Type:str
    Land_Hectares:float=Field(gt=0,le=100000)
    classification_model:str='rbf_svm'
    regression_model:str='random_forest_regressor'

@lru_cache(maxsize=20)
def load_model(key):
    info=REGISTRY[key]
    if 'file' not in info: raise ValueError('Hybrid model is calculated dynamically')
    return joblib.load(BASE/'models'/info['file'])

def validate(row):
    for key,allowed in [('Province','provinces'),('Crop','crops'),('Irrigation_Type','irrigation_types')]:
        if row[key] not in META[allowed]: raise HTTPException(422,f'Unknown {key}: {row[key]}')
    c=row['classification_model']; r=row['regression_model']
    if c not in REGISTRY or REGISTRY[c]['task'] not in ('classification','hybrid'): raise HTTPException(422,'Unknown classification model')
    if r not in REGISTRY or REGISTRY[r]['task']!='regression': raise HTTPException(422,'Unknown regression model')

def category_from_yield(crop,yield_value):
    a,b=THRESHOLDS[crop]
    return 'Low' if yield_value<a else 'Medium' if yield_value<b else 'High'

def predict(row):
    validate(row)
    frame=pd.DataFrame([row],columns=FEATURES)
    c=row['classification_model']; r=row['regression_model']
    yield_value=max(0.,float(load_model(r).predict(frame)[0]))
    if c=='rf_regression_category':
        class_yield=float(load_model('random_forest_regressor').predict(frame)[0])
        category=category_from_yield(row['Crop'],class_yield)
    else:
        category=str(load_model(c).predict(frame)[0])
    ci=REGISTRY[c]; ri=REGISTRY[r]
    return {'category':category,'yield_tons_per_hectare':round(yield_value,3),'total_production_tons':round(yield_value*row['Land_Hectares'],3),
      'classification_model':{'id':c,'name':ci['name'],'test_accuracy_percent':ci['accuracy'],'test_macro_f1':ci['macro_f1']},
      'regression_model':{'id':r,'name':ri['name'],'test_mae_tons_per_hectare':ri['mae'],'test_r2':ri['r2']},
      'disclaimer':'Historical held-out test metrics on synthetic data; not a confidence or correctness guarantee for this individual prediction.'}

@app.get('/api/health')
def health(): return {'status':'ok','model_count':len(REGISTRY)}
@app.get('/api/metadata')
def metadata():
    return {**META,'classification_models':[{'id':k,**v} for k,v in REGISTRY.items() if v['task'] in ('classification','hybrid')],
      'regression_models':[{'id':k,**v} for k,v in REGISTRY.items() if v['task']=='regression']}
@app.post('/api/predict')
def prediction(data:Input): return predict(data.model_dump())
@app.post('/api/recommend')
def recommend(data:Input):
    base=data.model_dump(); rows=[]
    for crop in META['crops']:
        rows.append({'crop':crop,**predict({**base,'Crop':crop})})
    rows.sort(key=lambda x:x['yield_tons_per_hectare'],reverse=True)
    return {'comparisons':rows,'disclaimer':'Hypothetical yield comparison, NOT a suitability recommendation. Raw mass yields are not directly comparable for economic or agronomic suitability.'}
@app.post('/api/irrigation')
def irrigation(data:Input):
    base=data.model_dump()
    return {'comparisons':[{'irrigation_type':t,**predict({**base,'Irrigation_Type':t})} for t in META['irrigation_types']],
      'disclaimer':'Hypothetical model scenarios, not measured water savings or irrigation prescriptions.'}
@app.get('/api/models')
def model_comparison():
    return {'classification':sorted([{'id':k,**v} for k,v in REGISTRY.items() if v['task'] in ('classification','hybrid')],key=lambda x:-x['accuracy']),
      'regression':sorted([{'id':k,**v} for k,v in REGISTRY.items() if v['task']=='regression'],key=lambda x:x['mae']),
      'note':'All scores re-evaluated using the bundled synthetic dataset and a fixed held-out split. These are aggregate test-set metrics, not per-prediction confidence. The hybrid uses training-derived crop thresholds.'}
