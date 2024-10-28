import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
# from dotenv import load_dotenv
from datetime import datetime
from geopy.distance import geodesic
from xgboost import XGBClassifier


# load_dotenv()

#  initialize the OpenAI client with GROQ API
client = OpenAI(
    base_url = "https://api.groq.com/openai/v1",
    api_key ="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"
""
)

def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)    

xgboost_model = load_model('xgb_model.pkl')

naive_bayes_model = load_model('nb_model.pkl')

random_forest_model = load_model('rf_model.pkl')

decision_tree_model = load_model('dt_model.pkl')

lr_model = load_model('lr_model.pkl')

xgboost_model_SMOTE = load_model('xgboost_model_SMOTE.pkl')

gbc_model = load_model('gbc_model.pkl')    

with open("merchant_label_encoder", "rb") as file:
   merchant_label_encoder = pickle.load(file)

with open("state_label_encoder.pkl", "rb") as file:
   state_label_encoder = pickle.load(file)

# function to calculate age from dob
def calculate_age(dob):
   today = datetime.today()
   return today.year - dob.year

# function to calculate distance between two lat-long coordinates
def calc_distance(lat, long, merch_lat, merch_long):
   return geodesic((lat,long), (merch_lat,merch_long)).miles

# Prepare the input data for the functions
def prepare_input(trans_date_trans_time, amt, merchant, category, gender, city_pop, dob, lat, long, merch_lat, merch_long, state):
  
  merchant_encoded = merchant_label_encoder.transform([merchant])[0]
  state_encoded = state_label_encoder.transform([state])[0]

  age = calculate_age(dob)
  distance = calc_distance(lat, long, merch_lat, merch_long)  

  transaction_datetime = datetime.strptime(trans_date_trans_time, "%Y-%m-%d %H:%M:%S")  
  trans_hour = transaction_datetime.hour
  trans_day_of_week = transaction_datetime.weekday()

  input_dict = {
    'amt': amt,
    'merchant_encoded': merchant_encoded,
    'state_encoded': state_encoded,
    'category_food_dining': 1 if category == "food_dining" else 0,
    'category_gas_transport': 1 if category == "gas_transport" else 0,
    'category_grocery_net': 1 if category == "grocery_net" else 0,
    'category_grocery_pos': 1 if category == "grocery_pos" else 0,
    'category_health_fitness': 1 if category == "health_fitness" else 0,
    'category_home': 1 if category == "home" else 0,
    'category_kids_pets': 1 if category == "kids_pets" else 0,
    'category_misc_net': 1 if category == "misc_net" else 0,
    'category_misc_pos': 1 if category == "misc_pos" else 0,
    'category_personal_care': 1 if category == "personal_care" else 0,
    'category_shopping_net': 1 if category == "shopping_net" else 0,
    'category_shopping_pos': 1 if category == "shopping_pos" else 0,
    'category_travel': 1 if category == "travel" else 0,
    'age': age,
    'distance': distance,
    'city_pop' : city_pop,
    'trans_hour': trans_hour,
    'trans_day_of_week': trans_day_of_week,
    'gender_M': 1 if gender == "M" else 0,
    'gender_F': 1 if gender == "F" else 0
  }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict