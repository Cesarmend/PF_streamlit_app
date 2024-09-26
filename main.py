import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pydeck as pdk  # Para visualización en mapas

# Función de distancia Haversine
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 1000  # Convertir a metros

# Función que filtra ubicaciones por distancia mínima usando Haversine
def filter_by_minimum_distance(locations, min_distance_km=1):
    valid_locations = []
    for i, loc1 in locations.iterrows():
        for j, loc2 in locations.iterrows():
            if i != j:  # Comparar diferentes ubicaciones
                distance = haversine_distance(loc1['latitude'], loc1['longitude'], loc2['latitude'], loc2['longitude'])
                #print(f"Distancia entre {loc1['postal_code']} y {loc2['postal_code']}: {distance} metros")  # Agregar este print
                if distance >= min_distance_km * 1000:  # Distancia mínima en metros
                    valid_locations.append(loc2)
                    if len(valid_locations) == 2:  # Limitar a las 2 mejores ubicaciones
                        return valid_locations
    return valid_locations




# ---- Caching Data ----
@st.cache_data
def load_data():
    # Cargar los datos desde el archivo parquet sin 'avg_sentiment'
    df = pd.read_parquet('stream2.parquet')
    return df

# ---- Cargar el modelo ----
@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')

# ---- Agrupar los datos y calcular puntajes ----
@st.cache_data
def process_data(df):
    df['categories'] = df['categories'].str.split(',\s*')
    df_exploded = df.explode('categories')
    df_exploded['avg_sentiment'] = df_exploded[['reviews_sentiment', 'tips_sentiment']].mean(axis=1).fillna(0)
    X_geo = df_exploded[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_exploded['cluster'] = kmeans.fit_predict(X_geo)
    df_exploded['combined_score'] = df_exploded['stars'] + df_exploded['avg_sentiment'] * 7.5
    min_combined_score = df_exploded['combined_score'].min()
    max_combined_score = df_exploded['combined_score'].max()
    df_exploded['combined_score'] = 1 + (df_exploded['combined_score'] - min_combined_score) * (10 - 1) / (max_combined_score - min_combined_score)
    df_grouped = df_exploded.groupby(['main category', 'categories', 'cluster']).agg({
        'combined_score': 'mean',
        'stars': 'mean',
        'avg_sentiment': 'mean',
        'business_id': 'count',
        'latitude': 'mean',
        'longitude': 'mean',
        'postal_code': 'first'
    }).reset_index()
    df_grouped['combined_score'] = df_grouped['combined_score'].round(2)
    return df_grouped

def show_postal_codes_map(postal_codes, postal_code_coords):
    # Filtrar códigos postales con coordenadas disponibles
    postal_codes_with_coords = [pc for pc in postal_codes if pc in postal_code_coords]
    
    if not postal_codes_with_coords:
        st.write("No se encontraron coordenadas para los códigos postales seleccionados.")
        return

    # Convertir códigos postales a coordenadas
    locations = pd.DataFrame([postal_code_coords[pc] for pc in postal_codes_with_coords], columns=['latitude', 'longitude'])

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
                data=locations,
                get_position='[longitude, latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=500,
            ),
        ],
    ))

# ---- Interfaz de Streamlit ----
def main():
    st.title("🏢 Comercios de interés en Filadelfia")

    # Cargar los datos y procesarlos solo una vez
    if 'df_grouped' not in st.session_state:
        df = load_data()
        st.session_state.df_grouped = process_data(df)

    df_grouped = st.session_state.df_grouped

    # Muestra las top 5 categorías
    top_5_categories = df_grouped.groupby('main category').agg({
        'combined_score': 'mean'
    }).reset_index().nlargest(5, 'combined_score')

    top_5_categories['combined_score'] = top_5_categories['combined_score'].round(2)

    top_5_categories = top_5_categories.rename(columns={
        'main category': 'Categoría Principal',
        'combined_score': 'Puntaje'
    })

    top_5_categories.index = np.arange(1, len(top_5_categories) + 1)

    st.subheader("📊 Top 5 Categorías")
    st.write(top_5_categories[['Categoría Principal', 'Puntaje']])

    selected_main_category = st.selectbox('Selecciona una categoría', top_5_categories['Categoría Principal'].tolist())

    df_filtered = df_grouped[df_grouped['main category'] == selected_main_category]

    top_2_subcategories = df_filtered.groupby('categories').agg({
        'combined_score': 'mean'
    }).reset_index().nlargest(2, 'combined_score')

    top_2_subcategories = top_2_subcategories.rename(columns={
        'categories': 'Subcategorías',
        'combined_score': 'Puntaje'
    })

    top_2_subcategories['Puntaje'] = top_2_subcategories['Puntaje'].round(2)

    top_2_subcategories.index = np.arange(1, len(top_2_subcategories) + 1)

    st.subheader(f"Top 2 subcategorías para '{selected_main_category}'")
    st.write(top_2_subcategories[['Subcategorías', 'Puntaje']])

    selected_subcategory = st.selectbox('Selecciona una subcategoría', top_2_subcategories['Subcategorías'].tolist())

    if st.button('🏢 Recomendar ubicaciones'):
        top_1_location = df_grouped[df_grouped['categories'] == selected_subcategory].nlargest(1, 'combined_score')
        
        similar_categories = df_grouped[
            (df_grouped['categories'] != selected_subcategory) & 
            (abs(df_grouped['avg_sentiment'] - top_1_location['avg_sentiment'].values[0]) < 0.1)
        ]
        if len(similar_categories) > 0:
            top_2_location = similar_categories.nlargest(1, 'combined_score')
        else:
            top_2_location = df_grouped[df_grouped['categories'] != selected_subcategory].nlargest(1, 'combined_score')

        top_2_locations = pd.concat([top_1_location, top_2_location], ignore_index=True)
        top_2_locations.index += 1

        st.write(f"🏠 Posibles Ubicaciones para '{selected_subcategory}':")
        st.write(top_2_locations[['postal_code']])

        # Aquí se pasa postal_code_coords (un diccionario con los códigos postales y sus coordenadas)
        postal_code_coords = {
    19027: [40.076470, -75.127484],
    19093: [39.950000, -75.180000],
    19099: [39.950000, -75.160000],
    19100: [39.952584, -75.165222],
    19101: [39.952272, -75.162518],
    19102: [39.949424, -75.165884],
    19103: [39.951199, -75.170658],
    19104: [39.957007, -75.197148],
    19106: [39.948937, -75.153921],
    19107: [39.944402, -75.158027],
    19108: [39.948043, -75.154380],
    19109: [39.949823, -75.163825],
    19110: [39.950372, -75.164532],
    19111: [40.061417, -75.083345],
    19112: [39.895943, -75.173964],
    19113: [39.873100, -75.274796],
    19114: [40.071728, -74.983126],
    19115: [40.108637, -75.048397],
    19116: [40.114249, -75.002230],
    19118: [40.067600, -75.197800],
    19119: [40.056644, -75.189574],
    19120: [40.016709, -75.115846],
    19121: [39.971941, -75.159771],
    19122: [39.979333, -75.153033],
    19123: [39.958902, -75.143597],
    19124: [40.030360, -75.104880],
    19125: [39.964337, -75.133546],
    19126: [40.067164, -75.146497],
    19127: [40.023455, -75.219510],  
    19128: [40.044357, -75.231860],
    19129: [40.009489, -75.193298],
    19130: [39.963091, -75.173804],
    19131: [39.998275, -75.231635],
    19132: [40.004688, -75.178820],
    19133: [39.998528, -75.129278],
    19134: [39.994155, -75.096265],
    19135: [40.025481, -75.035853],
    19136: [40.048358, -75.012134],
    19137: [39.999399, -75.090452],
    19138: [40.058689, -75.164659],
    19139: [39.958979, -75.241256],
    19140: [40.005424, -75.150720],
    19141: [40.036820, -75.143506],
    19142: [39.922691, -75.230404],
    19143: [39.948547, -75.216113],
    19144: [40.035351, -75.173668],
    19145: [39.915672, -75.185781],  
    19146: [39.939440, -75.166805],
    19147: [39.929918, -75.143519],
    19148: [39.913127, -75.155426],
    19149: [40.047876, -75.058251],
    19150: [40.070028, -75.157582],
    19151: [39.975985, -75.253740],
    19152: [40.051333, -75.058631],
    19153: [39.900151, -75.229263],
    19154: [40.083871, -74.960973],
    19155: [39.874557, -75.243750],
    19160: [40.009879, -75.149847],
    19176: [39.934212, -75.147424],
    19195: [39.953640, -75.166656]
}

        show_postal_codes_map(top_2_locations['postal_code'].tolist(), postal_code_coords)

if __name__ == "__main__":
    main()