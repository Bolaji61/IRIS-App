from flask import Flask, render_template, request, jsonify
import numpy as np
import  joblib

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        review = request.values.getlist('review')
        review = np.array(review).reshape(1,-1)
        classifier = joblib.load('class.pkl')
        result = classifier.predict(review)
        result = result[0]
        return 'The flower classification is ' + result
if __name__ == "__main__":
    app.run(debug= True)