from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine
from dotenv import load_dotenv


import os
import pandas as pd


app = Flask(__name__)
model = LogisticRegression(max_iter=500)

load_dotenv()
database_url = os.getenv("DATABASE_URL")

engine = create_engine(database_url)
connection = engine.connect()

query = 'select * from tableName'
df = pd.read_sql_query(query, connection)

#Build ML Model Here
df.iloc[:,6] = df.iloc[:,6].replace(["Rented", "Owned"], [0, 1])
df.iloc[:,4] = df.iloc[:,4].replace(["Single", "Married"], [0, 1])
df.iloc[:,3] = df.iloc[:,3].replace(["High School Diploma", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctorate"], [0, 1, 2, 3, 4])
df.iloc[:,1] = df.iloc[:,1].replace(["Male", "Female"], [0, 1])

x = df.iloc[:,0:-1]
y = df.iloc[:,-1]

model.fit(x, y)


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    input_data = pd.DataFrame([input_data])
    predictions = model.predict(input_data)
    
    response = {'predictions': predictions.tolist()}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
