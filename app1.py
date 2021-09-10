import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import pandas as pd
df=pd.read_csv("int.csv",encoding="ISO-8859-1")



app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int1=[x for x in request.form.values()]
    def text_seperation (x):
        return x.split()
    bow_transformer = CountVectorizer(analyzer=text_seperation).fit(df['SKILL'])

    messages_bowt = bow_transformer.transform(int1)
    tfidf_transformert = TfidfTransformer().fit(messages_bowt)
    messages_tfidft = tfidf_transformert.transform(messages_bowt)
    predict= model.predict(messages_tfidft)
    p=int(np.asarray(predict))
    df1=df[df["JOBLABELS"]==p]
    df2=df1.reset_index()
    output=df2["INTERNSHIPS"].tolist()
    return render_template('index.html', prediction_text='YOUR INTERNSHIPS {}'.format(output))
        
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
