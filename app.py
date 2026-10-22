from flask import Flask, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)

with open('pipeline.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)


@app.post('/predecir')
def predecir():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'El cuerpo debe ser un objeto JSON'}), 400

    required = {'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'}
    faltantes = sorted(required - data.keys())
    if faltantes:
        return jsonify({'error': 'Faltan campos', 'campos': faltantes}), 400

    prediccion = modelo.predict(pd.DataFrame([data]))
    return jsonify({'Survived': int(prediccion[0])})


if __name__ == '__main__':
    app.run(debug=True)
