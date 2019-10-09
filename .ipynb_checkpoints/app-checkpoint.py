from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.externals import  joblib

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        review = (request.form['review'])
        data = []
        data.append(review)
        # for i in range(len(data)):
        classifier = joblib.load('class.pkl')
        # # result = classifier.predict([[2.1, 4.3, 3.8, 1.5]])
        result = classifier.predict([data[-1:]])
        return result
        # # data = request.get_json(force = True)
        

        # result = result[0]
        # return data
        # return result
        # return 'The flower classification is ' + result
#prediction = knn.predict([[np.array(data['exp'])]])
if __name__ == "__main__":
    app.run(port =5000, debug= True)