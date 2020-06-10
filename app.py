from flask import Flask, render_template, request, jsonify
import numpy as np
import  joblib

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        form_input = request.values.getlist('review')
        form_input = np.array(form_input).astype(np.float64)
        form_input = np.array(form_input).reshape(1,-1)
        classifier = joblib.load('class.pkl')
        result = classifier.predict(form_input)
        result = result[0]
        return 'The flower classification is ' + result

if __name__ == "__main__":
    app.run(port = 5000, debug= True)