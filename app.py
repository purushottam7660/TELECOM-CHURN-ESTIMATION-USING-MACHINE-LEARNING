from flask import Flask,render_template,request
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report

df = pd.read_csv(r'Dataset/churn-bigml-80.csv')

### For checking null values
df.isnull().sum()
## removing Highly correlated features from the data set
def corelation(file):

    file = file.drop(['State', 'Area code', 'Total day charge', 'Total eve charge',
               'Total night charge', 'Total intl charge','Account length'],axis=1)
    return file

def cat_num(file):
    vmp = file['Voice mail plan'].replace(to_replace = {'Yes':1,'No':0})
    ip = file['International plan'].replace(to_replace={'Yes':1,'No':0})

    return vmp,ip
def concat(file):
    file['International plan']= ip
    file['Voice mail plan']=vmp
    return file
train = corelation(df)
vmp,ip = cat_num(train)
print(vmp)
print(ip)
train = concat(train)
print(train)

def splitting(file):
    X = file.drop(['Churn'],axis = 1)
    y = file.Churn
    return X,y

def visualisations(file):
    plot = file.corr()
    sns.heatmap(plot,annot=True)
    plt.show()

visualisations(df)


app = Flask(__name__)
app.config['upload folder']= r'C:\Final Year Project\CODE\Dataset'
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload',methods = ['POST','GET'])
def upload():
    if request.method == "POST":
        file = request.files['file']
        filetype = os.path.splitext(file.filename)[1]
        if filetype == '.csv':
            path = os.path.join(app.config['upload folder'], file.filename)
            file.save(path)
            print(path)
            df = pd.read_csv(path)
            return render_template('view.html',col_name =df.columns,row_val = list(df.values.tolist()))
        elif filetype != '.csv':
            return render_template('upload.html',msg = 'invalid')
        # return render_template('upload.html')

    return render_template('upload.html')

test = pd.read_csv(r'Dataset/churn-bigml-20.csv')
test = corelation(test)
vmp,ip = cat_num(test)
test = concat(test)
test,act = splitting(test)

@app.route('/viz')
def viz():

    return render_template('visualisations.html')


@app.route('/model',methods = ["POST","GET"])
def model():
    if request.method == 'POST':
        model = int(request.form['model'])
        X,y = splitting(train)
        print(X.columns)
        if model == 1:
            model1 = SVC()
            model1.fit(X,y)
            file = r'Models/SVC.h5'
            pickle.dump(model1,open(file,'wb'))
            pred = model1.predict(test)
            print(test.columns)
            score = accuracy_score(act,pred)
            print(score)
            return render_template('model.html',msg = 'success',score = score)
        elif model == 2:
            model2 = RandomForestClassifier()
            model2.fit(X,y)
            file = r'Models/RFC.h5'
            pickle.dump(model2,open(file,'wb'))
            pred = model2.predict(test)
            score = accuracy_score(act,pred)
            print(score)
            return render_template('model.html',msg = 'success',score = score)
    return render_template('model.html')

@app.route('/prediction',methods = ["POST","GET"])
def pred():
    if request.method == "POST":
        print('a')
        a = (request.form['f1'])
        b = (request.form['f2'])
        c = int(request.form['f3'])
        d = int(request.form['f4'])
        e = int(request.form['f5'])
        f = int(request.form['f6'])
        g = int(request.form['f7'])
        h = int(request.form['f8'])
        i = int(request.form['f9'])
        j = int(request.form['f10'])
        k = int(request.form['f11'])
        l = int(request.form['f12'])
        values = [int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h),int(i),int(j),int(k),int(l)]
        print(values)
        file = r'Models/RFC.h5'
        pred_model = pickle.load(open(file,'rb'))
        res = pred_model.predict([values])
        print(res)
        if res ==[False]:
            a='No Churn'
        else:
            a='Churn'
        return render_template('predictions.html',res=res)

    return render_template('predictions.html')





if __name__ == '__main__':
    app.run(debug=True)