from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.externals import  joblib
import traceback

app = Flask(__name__)

@app.route('/predict', methods = ['POST', 'GET'])
# def main():
#     if request.method == 'GET':
#         return render_template('index.html')
#     if request.method == 'POST':
#         review = (request.values.getlist('review'))
#         classifier = joblib.load('class.pkl')
#         # # result = classifier.predict([[2.1, 4.3, 3.8, 1.5]])
#         result = classifier.predict([review])
     
#         # # data = request.get_json(force = True)
        
#         result = result[0]
#         return 'The flower classification is ' + result
# #prediction = knn.predict([[np.array(data['exp'])]])
# if __name__ == "__main__":
#     app.run(port =5000, debug= True)


def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            # query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    # try:
    #     port = int(sys.argv[1]) # This is for a command-line input
    # except:
    #     port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("class.pkl") # Load "model.pkl"
    print ('Model loaded')
    # model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    # print ('Model columns loaded')

    app.run(port=5000, debug=True)