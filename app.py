import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

@st.cache_resource
def load_pipe():
    return joblib.load("model.pkl")

pipe = load_pipe()

st.title("🚗 Car MSRP Prediction")
st.write("Enter the car information in the sidebar to predict it's MSRP")

st.sidebar.header("Enter Car Information")

year = st.sidebar.slider("Year", min_value=1990, max_value=2026, value=2015)

engine_hp = st.sidebar.number_input("Engine HP", min_value=0.0, value=200.0)

engine_cylinders = st.sidebar.number_input("Engine Cylinders", min_value=0.0, max_value=16.0, value=4.0)

number_of_doors = st.sidebar.number_input("Number of Doors", min_value=2.0, max_value=4.0, value=4.0)

highway_mpg = st.sidebar.number_input("Highway MPG", min_value=0.0, value=30.0)

city_mpg = st.sidebar.number_input("City MPG", min_value=0.0, value=22.0)

popularity = st.sidebar.number_input("Popularity", min_value=0, value=1000)

data = pd.read_csv("data.csv")
car_brands = sorted(data["Make"].dropna().unique())
make = st.sidebar.selectbox("Make", car_brands)

fuel_types = sorted(data["Engine Fuel Type"].dropna().unique())
engine_fuel_type = st.sidebar.selectbox("Engine Fuel Type", fuel_types)

transmission_types = sorted(data["Transmission Type"].dropna().unique())
transmission_type = st.sidebar.selectbox("Transmission Type", transmission_types)   

driven_wheels_list = sorted(data["Driven_Wheels"].dropna().unique()) 
driven_wheels = st.sidebar.selectbox("Driven Wheels", driven_wheels_list)

vehicle_sizes = sorted(data["Vehicle Size"].dropna().unique())
vehicle_size = st.sidebar.selectbox("Vehicle Size", vehicle_sizes) 

vehicle_styles = sorted(data["Vehicle Style"].dropna().unique())
vehicle_style = st.sidebar.selectbox("Vehicle Style", vehicle_styles)  

#......
if engine_cylinders == 0:
    hp_per_cylinder = 0
else:
    hp_per_cylinder = engine_hp / engine_cylinders 

mpg_average = (highway_mpg + city_mpg) / 2

#....

new_data = {"Year": year, "Engine HP": engine_hp, "Engine Cylinders": engine_cylinders, "Number of Doors": number_of_doors, "highway MPG": highway_mpg, "city mpg": city_mpg, "Popularity": popularity, "HP per Cylinder": hp_per_cylinder, "MPG Average": mpg_average, "Make": make, "Engine Fuel Type": engine_fuel_type, "Transmission Type": transmission_type, "Driven_Wheels": driven_wheels, "Vehicle Size": vehicle_size, "Vehicle Style": vehicle_style}

new_data_df = pd.DataFrame(new_data, index=[0])

#......
button = st.button("predict")

if button:
    result = pipe.predict(new_data_df)

    st.write("Predict MSRP:")
    st.write(f"${result[0]:,.2f}") 



