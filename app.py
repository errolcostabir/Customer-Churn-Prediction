from email.mime import application
from unittest import result
from flask import Flask, request, render_template
#import numpy as np
#import pandas as pd

#from sklearn.preprocessing import StandardScaler
#from src.pipeline import predict_pipeline
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

##route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            SeniorCitizen=int(request.form.get('SeniorCitizen')),
            Partner=request.form.get('Partner'),
            Dependents=request.form.get('Dependents'),
            PhoneService=request.form.get('PhoneService'),
            MultipleLines=request.form.get('MultipleLines'),
            InternetService=request.form.get('InternetService'),
            OnlineSecurity=request.form.get('OnlineSecurity'),
            OnlineBackup=request.form.get('OnlineBackup'),
            DeviceProtection=request.form.get('DeviceProtection'),
            TechSupport=request.form.get('TechSupport'),
            StreamingTV=request.form.get('StreamingTV'),
            StreamingMovies=request.form.get('StreamingMovies'),
            Contract=request.form.get('Contract'),
            PaperlessBilling=request.form.get('PaperlessBilling'),
            PaymentMethod=request.form.get('PaymentMethod'),
            MonthlyCharges=float(request.form.get('MonthlyCharges')),
            TotalCharges=float(request.form.get('TotalCharges')),
            tenure=float(request.form.get('tenure'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        print(results)
        if int(results[0])== 1:
            results_txt="Customer will Churn"
        else: 
            results_txt="Customer will not Churn"
        return render_template('home.html',resultss=results_txt)
    
    
if __name__=="__main__":
    app.run(debug=True)
    
    
    
