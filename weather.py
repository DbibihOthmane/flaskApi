import os
import requests
from dotenv import load_dotenv

def weather_fetch(city):
    
    api_key = "5bf37fa25d9937fe92946c7ee097754b"
    
    # Construire l'URL pour récupérer la géolocalisation de la ville
    base_url = "http://api.openweathermap.org/geo/1.0/direct?q={city_name}&appid={API_key}&limit=1"
    complete_url = base_url.format(city_name=city, API_key=api_key)
    
    # Faire la requête à l'API
    response = requests.get(complete_url)
    x = response.json()
    
    # Vérifier si la réponse contient des données
    if not x:
        raise ValueError(f"Aucune donnée trouvée pour la ville: {city}")
    
    # Extraire les coordonnées (latitude et longitude)
    try:
        lat = x[0]["lat"]
        lon = x[0]["lon"]
    except KeyError:
        raise ValueError(f"Les coordonnées pour la ville {city} n'ont pas pu être extraites.")
    
    # Construire l'URL pour obtenir les données météorologiques
    base_url = "https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_key}"
    complete_url = base_url.format(lat=lat, lon=lon, API_key=api_key)
    
    # Faire la requête pour les données météorologiques
    response = requests.get(complete_url)
    data = response.json()
    
    # Vérifier si la réponse contient des données météo valides
    if response.status_code != 200 or 'main' not in data:
        raise ValueError("Erreur lors de la récupération des données météorologiques.")
    
    # Extraire les informations de température et d'humidité
    temperature = round((data["main"]["temp"] - 273.15), 2)  # Convertir de Kelvin à Celsius
    humidity = data["main"]["humidity"]
    
    return temperature, humidity
