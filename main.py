import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pydeck as pdk  # Para visualización en mapas
import pickle
import xgboost as xgb

# Función para cargar el modelo y el scaler
@st.cache_resource
def load_model_and_scaler():
    model = xgb.XGBRegressor()
    model.load_model("xgboost_model.json")
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_parquet('stream2.parquet')

# Cargar el modelo y el scaler
best_model, scaler = load_model_and_scaler()

# Cargar los datos
df_main = load_data()
df_main = df_main[df_main['main category'] != 'Misc']

# 1. Análisis de Sentimiento
df_main['avg_sentiment'] = df_main[['reviews_sentiment', 'tips_sentiment']].mean(axis=1).fillna(0)

# 2. Agrupación por cluster (latitud y longitud)
X_geo = df_main[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=5, random_state=42)
df_main['cluster'] = kmeans.fit_predict(X_geo)

# 3. Combinar puntaje por stars y análisis de sentimiento
df_main['combined_score'] = df_main['stars'] + df_main['avg_sentiment'] * 7.5

# 4. Normalización del combined_score entre 1 y 10
min_combined_score = df_main['combined_score'].min()
max_combined_score = df_main['combined_score'].max()
df_main['combined_score'] = 1 + (df_main['combined_score'] - min_combined_score) * (10 - 1) / (max_combined_score - min_combined_score)

# 5. Agrupar negocios por 'main category', 'categories' y 'cluster'
df_grouped = df_main.groupby(['main category', 'categories', 'cluster']).agg({
    'combined_score': 'mean',
    'stars': 'mean',
    'avg_sentiment': 'mean',
    'business_id': 'count',
    'latitude': 'mean',
    'longitude': 'mean',
    'postal_code': 'first',
}).reset_index()

# 6. Predicción de puntaje con el modelo cargado
X_for_prediction = df_grouped[['latitude', 'longitude', 'stars', 'avg_sentiment', 'postal_code']]
X_for_prediction_scaled = scaler.transform(X_for_prediction)
df_grouped['predicted_combined_score'] = best_model.predict(X_for_prediction_scaled)

# Función Haversine para calcular la distancia
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 1000  # Convertir a metros

# 7. Seleccionar las 5 categorías con el mejor combined_score predicho
top_5_categories = df_grouped.groupby('main category').agg({
    'predicted_combined_score': 'mean'
}).reset_index().nlargest(5, 'predicted_combined_score')

# Mostrar las top 5 categorías con formato de dos decimales
st.title("🏢 Comercios de interés en Filadelfia")

# Ajustar índices de las "Top 5 Categorías" para que empiecen desde 1
top_5_categories = top_5_categories.reset_index(drop=True)  # Reiniciar los índices
top_5_categories.index = top_5_categories.index + 1  # Iniciar desde 1

st.subheader("📊 Top 5 Categorías")
st.dataframe(top_5_categories[['main category', 'predicted_combined_score']].rename(columns={
    'main category': 'Categoría Principal',
    'predicted_combined_score': 'Puntaje'
}).style.format({"Puntaje": "{:.2f}"}).set_properties(**{'text-align': 'center'}))

# Dividir el df_grouped en categorías individuales
df_filtered_categories = df_grouped.copy()
df_filtered_categories['categories'] = df_filtered_categories['categories'].str.split(',\s*')
df_filtered_categories = df_filtered_categories.explode('categories')

# Eliminar categorías no deseadas
categorias_a_eliminar = [
    "Ear Nose & Throat", "Plastic Surgeons", "Scavenger Hunts", "Holistic Animal Care", 
    "Pet Cremation Services", "Education", "Street Vendors", "Marketing", 
    "Professional Services", "Cooking Schools", "Badminton", "Gold Buyers", 
    "Commercial Real Estate", "Window Washing", "Hotels & Travel", "Tours"
]
df_filtered_categories = df_filtered_categories[~df_filtered_categories['categories'].isin(categorias_a_eliminar)]

# Seleccionar una de las top 5 categorías principales
st.subheader("📋 Selecciona una Categoría Principal")
selected_main_category = st.selectbox("Categoría Principal", top_5_categories['main category'])

# Filtrar el DataFrame para la categoría seleccionada
df_filtered = df_filtered_categories[df_filtered_categories['main category'] == selected_main_category]

# Agrupar por 'categories' dentro de la 'main category' seleccionada
top_2_subcategories = df_filtered.groupby('categories').agg({
    'predicted_combined_score': 'mean'
}).reset_index().nlargest(2, 'predicted_combined_score')

# Ajustar índices de las "Top 2 Subcategorías" para que empiecen desde 1
top_2_subcategories = top_2_subcategories.reset_index(drop=True)  # Reiniciar los índices
top_2_subcategories.index = top_2_subcategories.index + 1  # Iniciar desde 1

# Mostrar las 2 mejores subcategorías con formato de dos decimales
st.subheader(f"Top 2 Subcategorías para '{selected_main_category}'")
st.dataframe(top_2_subcategories.rename(columns={
    'categories': 'Subcategoría',
    'predicted_combined_score': 'Puntaje'
}).style.format({"Puntaje": "{:.2f}"}).set_properties(**{'text-align': 'center'}))

# Selección de subcategoría para recomendar ubicaciones
st.subheader("🔸 Selecciona una Subcategoría")
selected_subcategory = st.selectbox("Subcategoría", top_2_subcategories['categories'])

# 19. Predecir la primera ubicación basada en la subcategoría seleccionada
def recommend_top_1_location_for_subcategory(subcategory, df_filtered_categories, scaler, best_model):
    subcategory_data = df_filtered_categories[df_filtered_categories['categories'] == subcategory]
    
    if subcategory_data.empty:
        st.warning(f"No se encontraron ubicaciones para la subcategoría '{subcategory}'.")
        return pd.DataFrame()  # Devolver un DataFrame vacío si no hay datos
    
    # Seleccionar las características necesarias para la predicción
    X_subcategory = subcategory_data[['latitude', 'longitude', 'stars', 'avg_sentiment', 'postal_code']]
    
    # Escalar las características de la misma manera que durante el entrenamiento
    X_subcategory_scaled = scaler.transform(X_subcategory)
    
    # Predecir el combined_score usando el modelo entrenado
    subcategory_data['predicted_combined_score'] = best_model.predict(X_subcategory_scaled)
    
    # Seleccionar la mejor ubicación basada en la predicción del modelo
    top_1_location = subcategory_data.nlargest(1, 'predicted_combined_score')
    return top_1_location

# 20. Predecir la segunda ubicación basada en análisis de subcategorías similares usando el modelo entrenado y Haversine
def recommend_top_1_location_from_similar_categories(subcategory, top_1_lat, top_1_lon, df_filtered_categories, scaler, best_model):
    similar_categories_data = df_filtered_categories[
        (df_filtered_categories['categories'] != subcategory) & 
        (abs(df_filtered_categories['avg_sentiment'] - df_filtered_categories[df_filtered_categories['categories'] == subcategory]['avg_sentiment'].mean()) < 0.1)
    ]
    
    if similar_categories_data.empty:
        st.warning(f"No se encontraron ubicaciones similares para la subcategoría '{subcategory}'.")
        return pd.DataFrame()  # Devolver un DataFrame vacío si no hay datos
    
    # Seleccionar las características necesarias para la predicción
    X_similar_categories = similar_categories_data[['latitude', 'longitude', 'stars', 'avg_sentiment', 'postal_code']]
    X_similar_categories_scaled = scaler.transform(X_similar_categories)
    
    # Predecir el combined_score usando el modelo entrenado
    similar_categories_data['predicted_combined_score'] = best_model.predict(X_similar_categories_scaled)
    
    # Calcular la distancia desde la primera ubicación
    similar_categories_data['distance'] = similar_categories_data.apply(
        lambda row: haversine_distance(top_1_lat, top_1_lon, row['latitude'], row['longitude']), axis=1
    )
    
    # Seleccionar la mejor ubicación que esté a más de 1 km de la primera
    top_1_other_location = similar_categories_data[similar_categories_data['distance'] > 1000].nlargest(1, 'predicted_combined_score')
    
    return top_1_other_location

# Diccionario de códigos postales y coordenadas
postal_code_coords = {
    19027: [40.076470, -75.127484], 19093: [39.950000, -75.180000], 19099: [39.950000, -75.160000],
    19100: [39.952584, -75.165222], 19101: [39.952272, -75.162518], 19102: [39.949424, -75.165884],
    19103: [39.951199, -75.170658], 19104: [39.957007, -75.197148], 19106: [39.948937, -75.153921],
    19107: [39.944402, -75.158027], 19108: [39.948043, -75.154380], 19109: [39.949823, -75.163825],
    19110: [39.950372, -75.164532], 19111: [40.061417, -75.083345], 19112: [39.895943, -75.173964],
    19113: [39.873100, -75.274796], 19114: [40.071728, -74.983126], 19115: [40.108637, -75.048397],
    19116: [40.114249, -75.002230], 19118: [40.067600, -75.197800], 19119: [40.056644, -75.189574],
    19120: [40.016709, -75.115846], 19121: [39.971941, -75.159771], 19122: [39.979333, -75.153033],
    19123: [39.958902, -75.143597], 19124: [40.030360, -75.104880], 19125: [39.964337, -75.133546],
    19126: [40.067164, -75.146497], 19127: [40.023455, -75.219510], 19128: [40.044357, -75.231860],
    19129: [40.009489, -75.193298], 19130: [39.963091, -75.173804], 19131: [39.998275, -75.231635],
    19132: [40.004688, -75.178820], 19133: [39.998528, -75.129278], 19134: [39.994155, -75.096265],
    19135: [40.025481, -75.035853], 19136: [40.048358, -75.012134], 19137: [39.999399, -75.090452],
    19138: [40.058689, -75.164659], 19139: [39.958979, -75.241256], 19140: [40.005424, -75.150720],
    19141: [40.036820, -75.143506], 19142: [39.922691, -75.230404], 19143: [39.948547, -75.216113],
    19144: [40.035351, -75.173668], 19145: [39.915672, -75.185781], 19146: [39.939440, -75.166805],
    19147: [39.929918, -75.143519], 19148: [39.913127, -75.155426], 19149: [40.047876, -75.058251],
    19150: [40.070028, -75.157582], 19151: [39.975985, -75.253740], 19152: [40.051333, -75.058631],
    19153: [39.900151, -75.229263], 19154: [40.083871, -74.960973]
}


# Mostrar el mapa con Pydeck
def show_postal_codes_map(postal_codes, postal_code_coords):
    map_data = pd.DataFrame([{'postal_code': pc, 'lat': postal_code_coords[pc][0], 'lon': postal_code_coords[pc][1]} 
                             for pc in postal_codes if pc in postal_code_coords])
    
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v10',
        initial_view_state=pdk.ViewState(
            latitude=39.9526,  # Centrar el mapa en Filadelfia
            longitude=-75.1652,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=map_data,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',  # Color de los puntos
                get_radius=500,
            ),
        ],
    ))

# Botón para recomendar ubicaciones y mostrar el mapa
if st.button("🔍 Recomendar ubicaciones y mostrar mapa"):
    top_1_location_subcategory = recommend_top_1_location_for_subcategory(selected_subcategory, df_filtered_categories, scaler, best_model)

    if not top_1_location_subcategory.empty:
        top_1_lat = top_1_location_subcategory['latitude'].iloc[0]
        top_1_lon = top_1_location_subcategory['longitude'].iloc[0]

        top_1_location_similar_category = recommend_top_1_location_from_similar_categories(selected_subcategory, top_1_lat, top_1_lon, df_filtered_categories, scaler, best_model)

        if not top_1_location_similar_category.empty:
            top_1_location_similar_category = top_1_location_similar_category.reset_index(drop=True)
            top_2_locations = pd.concat([top_1_location_subcategory, top_1_location_similar_category], ignore_index=True)

            # Ajustar los índices para que comiencen desde 1
            top_2_locations = top_2_locations.reset_index(drop=True)
            top_2_locations.index = top_2_locations.index + 1

            # Mostrar la tabla con los códigos postales
            st.subheader(f"Posibles Ubicaciones para la subcategoría '{selected_subcategory}'")
            st.dataframe(top_2_locations[['postal_code']].rename(columns={'postal_code': 'Código Postal'}))

            # Mostrar el mapa con las ubicaciones
            show_postal_codes_map(top_2_locations['postal_code'].tolist(), postal_code_coords)
        else:
            st.warning("No se encontraron ubicaciones similares.")
    else:
        st.warning("No se encontraron ubicaciones para la subcategoría seleccionada.")
