"""
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from weather import weather_fetch
from rain import rain_info
from plot import topCrops
import os
from weather import weather_fetch
from rain import rain_info
from info import info_range
import json
from flask import Response
import pandas as pd
from flask_cors import CORS
from keras import metrics
#import pickle
#import joblib
rainfall_data = rain_info()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json
app = Flask(__name__)
CORS(app) 



crop_recommendation_model_path = './models/DeepNN.h5' 

crop_recommendation_model = load_model(crop_recommendation_model_path, custom_objects={'mse': metrics.mean_squared_error}) 

crops = np.load('crops.npy', allow_pickle=True) 

#crop_recommendation_model_path = './models/RandomForest.pkl' ---> hdchi pour les models dial .pkl


#crop_recommendation_model = joblib.load(crop_recommendation_model_path) ---> hdchi pour les models dial .pkl

crops = np.load('crops.npy', allow_pickle=True)


# Initialisation de l'application Flask
app = Flask(__name__)

# Load rainfall data and the DNN model
rainfall_data = rain_info()
#crop_yield_model_path = "./models/DecisionTree.pkl" --> hadchi bach truni models lakhrin
crop_yield_model_path = "./models/Yield_DNN.h5"

# Load the model
#crop_yield_model = joblib.load(crop_yield_model_path) -->hadchi bach truni models lakhrin

crop_yield_model = load_model(crop_yield_model_path, custom_objects={'mse': metrics.mean_squared_error}) 

# Load the columns used in the original training
with open('columns.json', 'r') as f:
    data = json.load(f)


def crop_yield(formdata):
    crop = formdata["crop"]
    area = int(formdata["area"])
    season = formdata["season"]
    city = formdata["city"]

    # Fetch weather information
    temperature, humidity = weather_fetch(city)

    # Get rainfall data for the city and season
    rainfall = rainfall_data[rainfall_data["DIST"] == city][season].values[0]

    # Prepare the input DataFrame for the model
    columns = [index for index in data]  # Ensure the columns are in the correct order
    df = pd.DataFrame(columns=columns)

    df.loc[0] = 0  # Initialize all values to zero

    # Set input feature values
    df["Year"] = 2016  # Ensure this column matches the trained model
    if city in df.columns:
        df[city] = 1
    if season in df.columns:
        df[season] = 1
    if crop in df.columns:
        df[crop] = 1
    df["Area"] = area
    df["Temperature"] = temperature
    df["Rainfall"] = rainfall

    # Debugging step: Print columns and data types to ensure everything is correct
    print(f"Colonnes du DataFrame : {df.columns}")
    print(f"Types des colonnes : {df.dtypes}")
    
    # Align columns with those expected by the model (make sure the correct number of columns)
    df = df[columns]  # Reorder and ensure no extra columns

    # Check the shape of the DataFrame before predicting
    print(f"Shape du DataFrame : {df.shape}")

    input_features = df.values.astype(np.float32)  # Convert to numpy array

    # Debugging step: Print the shape of the input features
    print(f"Input shape for the model: {input_features.shape}")

    # Make prediction using the DNN model
    my_prediction = crop_yield_model.predict(input_features)
    
    # Convert the prediction to float before returning
    prediction = float(my_prediction[0][0])  # Ensure it's a float

    return prediction, temperature, humidity, rainfall


def ensure_serializable(obj):
    
    if isinstance(obj, dict):
        return {key: ensure_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return float(obj)  # Convert numpy types to Python float
    elif isinstance(obj, (np.int8, np.int16, int)):
        return int(obj)  # Convert numpy int to Python int
    return obj

# Route pour la prédiction du rendement des cultures
@app.route('/crop-yield-predict', methods=['POST'])
def crop_yield_prediction():
    data = request.json
    formdata = data['formdata']
    
    # Obtenir la prédiction
    prediction, temperature, humidity, rainfall = crop_yield(formdata)
    
    
    # Traitement des données de l'année et de la saison
    rainfall = round(rainfall, 2)
    year_yield, season_yield, temp_yield, rain_yield, humid_yield = info_range(formdata, temperature, humidity, rainfall)
    year_yield = {int(year): float(value) for year, value in year_yield.items()}
    
    # Préparer la réponse
    pred = {
        "prediction": prediction,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "year_yield": year_yield,
        "season_yield": season_yield,
        "temp_yield": temp_yield,
        "rain_yield": rain_yield,
        "humid_yield": humid_yield,
    }
    pred = ensure_serializable(pred)
    
    # Si aucune culture ne peut être cultivée
    if pred == '':
        response = {
            "status": "error",
            "result": pred,
            "message": "No crop can be grown in this region"
        }
    else:
        response = {
            "status": "success",
            "result": pred,
            "message": "Crop Yield fetched successfully"
        }
    print(response)
    return jsonify(response)

def crop_recommendation(formdata):
    # Récupérer les données météorologiques et de précipitations
    rainfall_data = rain_info()
    N = formdata['nitrogen']
    P = formdata['phosphorous']
    K = formdata['pottasium']
    ph = formdata['ph']
    season = formdata['season']
    city = formdata['city']
    
    # Récupérer la température et l'humidité pour la ville
    temperature, humidity = weather_fetch(city)
    
    # Vérifier si la ville et la saison existent dans les données de précipitations
    if city not in rainfall_data["DIST"].values or season not in rainfall_data.columns:
        return ["Invalid city or season"], temperature, humidity, None, None
    
    rainfall = rainfall_data[rainfall_data["DIST"] == city][season].values[0]
    
    # Préparer les données pour le modèle
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Vérification des dimensions et prétraitement si nécessaire
    try:
        my_prediction = crop_recommendation_model.predict(data)
    except Exception as e:
        return [f"Model prediction failed: {str(e)}"], temperature, humidity, rainfall, None

    # Identifier la culture avec la probabilité la plus élevée
    max_prob = max(my_prediction[0])  # Trouver la probabilité maximale
    max_index = my_prediction[0].tolist().index(max_prob)  # Obtenir l'index correspondant
    best_crop = crops[max_index]  # Culture correspondante à la probabilité maximale

    # Si aucune probabilité valide, retourner une culture par défaut
    if max_prob < 0.5:  # Seuil configurable
        best_crop = 'No crop'

    # Générer les données pour les graphiques
    try:
        chart_data = topCrops(crops, data)
    except Exception as e:
        chart_data = None  # Gérer les erreurs dans la génération des graphiques

    # Retourner uniquement la meilleure recommandation avec les données associées
    return [best_crop], temperature, humidity, rainfall, chart_data

    
def temp_list(formdata, prediction, temperature, humidity, rainfall):
    prediction = prediction.lower()
    highest_crop_list = {}
    crop_list = {}
    temp_list = [-4, -2, 0, 2, 4]
    for item in temp_list:
        data = crop_recommendation(
            formdata, temperature+item, humidity, rainfall)
        data1 = list(data.keys())
        if prediction not in data.keys():
            crop_list[item+temperature] = (0)
        else:
            crop_list[item+temperature] = (data[prediction])
        highest_crop_list[item+temperature] = (data1[0], data[data1[0]])
    return highest_crop_list, crop_list


def humid_list(formdata, prediction, temperature, humidity, rainfall):
    prediction = prediction.lower()
    highest_crop_list = {}
    crop_list = {}
    h_list = [-20, -10, 0, 10, 20]
    for item in h_list:
        data = crop_recommendation(
            formdata, temperature, humidity+item, rainfall)
        data1 = list(data.keys())
        if prediction not in data.keys():
            crop_list[item+humidity] = (0)
        else:
            crop_list[item+humidity] = (data[prediction])
        highest_crop_list[item+humidity] = (data1[0], data[data1[0]])
    return highest_crop_list, crop_list


def rain_list(formdata, prediction, temperature, humidity, rainfall):
    highest_crop_list = {}
    prediction = prediction.lower()
    crop_list = {}
    r_list = [-200, -100, 0, 100, 200]
    for item in r_list:
        data = crop_recommendation(
            formdata, temperature, humidity, rainfall+item)
        data1 = list(data.keys())
        if prediction not in data.keys():
            crop_list[item+rainfall] = (0)
        else:
            crop_list[item+rainfall] = (data[prediction])
        highest_crop_list[item+rainfall] = (data1[0], data[data1[0]])
    return highest_crop_list, crop_list


# Route pour la prédiction des cultures
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    data = request.get_json()  # Récupère les données JSON envoyées par le client
    formdata = data.get('formdata')  # Récupère les données du formulaire
    if not formdata:
        return jsonify({"status": "error", "message": "Missing form data"}), 400

    # Appel de la fonction de recommandation de culture
    prediction = crop_recommendation(formdata)

    prediction = ensure_serializable(prediction)

    # Préparation de la réponse
    if prediction == ['No crop']:
        response = {
            "status": "error",
            "result": prediction,
            "message": "No crop can be grown in this region"
        }
    else:
        response = {
            "status": "success",
            "result": prediction,
            "message": "Crop recommendation fetched successfully"
        }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

"""
from flask import Flask, request, jsonify
import numpy as np
from weather import weather_fetch
from rain import rain_info
from plot import topCrops
import os
import json
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Chargement des modèles et des données
crop_recommendation_model_path = './models/RandomForest.pkl'
crop_yield_model_path = "./models/DecisionTree.pkl"
crop_recommendation_model = joblib.load(crop_recommendation_model_path)
crop_yield_model = joblib.load(crop_yield_model_path)
crops = np.load('crops.npy', allow_pickle=True)

with open('columns.json', 'r') as f:
    data = json.load(f)

rainfall_data = rain_info()

# Fonction de prédiction du rendement des cultures
def crop_yield(formdata):
    crop = formdata["crop"]
    area = int(formdata["area"])
    season = formdata["season"]
    city = formdata["city"]

    # Récupérer température et humidité
    temperature, humidity = weather_fetch(city)

    # Récupérer la pluviométrie
    rainfall = rainfall_data[rainfall_data["DIST"] == city][season].values[0]

    # Charger les colonnes attendues
    columns = [index for index in data]
    df = pd.DataFrame(columns=columns)
    df.loc[0] = 0  # Initialiser avec des zéros

    # Remplir les données dans le DataFrame
    if city in df.columns:
        df[city] = 1
    if season in df.columns:
        df[season] = 1
    if crop in df.columns:
        df[crop] = 1
    df["Area"] = area
    df["Temperature"] = temperature
    df["Rainfall"] = rainfall

    # S'assurer que les colonnes sont alignées
    df = df[columns]
    input_features = df.values

    # Faire la prédiction
    try:
        prediction = crop_yield_model.predict(input_features)[0]
    except Exception as e:
        prediction = f"Error in prediction: {str(e)}"

    return prediction, temperature, humidity, rainfall
# Fonction de recommandation des cultures
def crop_recommendation(formdata):
    N = formdata['nitrogen']
    P = formdata['phosphorous']
    K = formdata['pottasium']
    ph = formdata['ph']
    season = formdata['season']
    city = formdata['city']

    temperature, humidity = weather_fetch(city)

    if city not in rainfall_data["DIST"].values or season not in rainfall_data.columns:
        return ["Invalid city or season"], temperature, humidity, None, None

    rainfall = rainfall_data[rainfall_data["DIST"] == city][season].values[0]
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    try:
        probs = crop_recommendation_model.predict_proba(data)
        crop_probs = {crops[i]: float(probs[0][i]) for i in range(len(crops))}
        top_crops = dict(sorted(crop_probs.items(), key=lambda x: x[1], reverse=True)[:5])
    except Exception as e:
        return [f"Model prediction failed: {str(e)}"], temperature, humidity, rainfall, None

    return top_crops, temperature, humidity, rainfall, crop_probs

# Route pour la prédiction des rendements
@app.route('/crop-yield-predict', methods=['POST'])
def crop_yield_prediction():
    data = request.json
    formdata = data['formdata']

    prediction, temperature, humidity, rainfall = crop_yield(formdata)

    response = {
        "status": "success",
        "result": {
            "prediction": prediction,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
        },
        "message": "Crop Yield fetched successfully"
    }

    return jsonify(response)

# Route pour la recommandation des cultures
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    data = request.json
    formdata = data['formdata']

    top_crops, temperature, humidity, rainfall, crop_probs = crop_recommendation(formdata)

    response = {
        "status": "success" if top_crops else "error",
        "result": {
            "top_crops": top_crops,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "crop_probabilities": crop_probs
        },
        "message": "Crop recommendation fetched successfully" if top_crops else "No crop can be grown in this region"
    }

    return jsonify(response)



# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)

