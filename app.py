import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import ftfy

# 1. Configuración de la interfaz web
st.set_page_config(page_title="FIFA 23 - Buscador de Similitudes", layout="centered")
st.title("FIFA 23 – Buscador de similitudes entre jugadores")

# 2. Carga y limpieza adaptativa de la base de datos
@st.cache_data
def cargar_datos_fifa23():
    # Intenta cargar 'fifa23_data.csv' o 'data.csv'
    archivos = ["fifa23_data.csv", "data.csv"]
    df = None
    
    for archivo in archivos:
        try:
            df = pd.read_csv(archivo, encoding="utf-8-sig")
            break
        except (UnicodeDecodeError, TypeError):
            try:
                df = pd.read_csv(archivo, encoding="latin-1")
                break
            except FileNotFoundError:
                continue
        except FileNotFoundError:
            continue

    if df is None:
        st.error("⚠️ No se encontró el archivo CSV de FIFA 23. Asegúrate de que se llame 'data.csv' o 'fifa23_data.csv' y esté en la misma carpeta.")
        st.stop()
        
    # Estandarización de nombres de columnas comunes en FIFA 23
    renombres = {
        "KnownAs": "Name",
        "FullName": "Name",
        "ClubName": "Club",
        "BestPosition": "Position",
        "OverallRating": "Overall"
    }
    df = df.rename(columns={k: v for k, v in renombres.items() if k in df.columns and v not in df.columns})

    # Rellenar valores vacíos y corregir codificación
    if "Name" in df.columns:
        df["Name"] = df["Name"].fillna("").astype(str).apply(ftfy.fix_text)
    
    if "Club" in df.columns:
        df["Club"] = df["Club"].fillna("").astype(str).apply(ftfy.fix_text)
        
    # Identificar columnas esenciales para limpiar filas nulas
    columnas_esenciales = ['Name', 'Age', 'Overall', 'ShortPassing', 'Dribbling']
    columnas_presentes = [col for col in columnas_esenciales if col in df.columns]
    
    if columnas_presentes:
        df = df.dropna(subset=columnas_presentes)
        
    return df

# Inicialización de la base de datos
df = cargar_datos_fifa23()

# 3. Formulario de selección y filtros
with st.container():
    if "Name" not in df.columns:
        st.error("⚠️ Tu archivo CSV no contiene una columna identificadora de jugador ('Name').")
        st.stop()
        
    jugadores_disponibles = sorted(df["Name"].unique())
    jugador_seleccionado = st.selectbox("Seleccionar jugador:", jugadores_disponibles)
    
    # Rango de sliders ajustado a los valores estándar de FIFA 23
    edad_maxima = st.slider("Edad máxima:", min_value=16, max_value=45, value=25, step=1)
    calificacion_maxima = st.slider("Calificación máxima general:", min_value=45, max_value=99, value=75, step=1)
    
    n_resultados = st.number_input("Los N mejores resultados:", min_value=1, max_value=30, value=5)
    buscar = st.button("Encontrar jugadores similares")

# 4. Procesamiento e índice de similitud matemática
if buscar:
    fila_objetivo = df[df["Name"] == jugador_seleccionado].iloc[0]
    id_objetivo = fila_objetivo["ID"] if "ID" in df.columns else jugador_seleccionado
    
    # Aplicar filtros de restricciones
    if "Age" in df.columns and "Overall" in df.columns:
        df_filtrado = df[
            (df["Age"] <= edad_maxima) & 
            (df["Overall"] <= calificacion_maxima) & 
            (df["Name"] != jugador_seleccionado)
        ]
    else:
        st.warning("No se pudo filtrar por Edad u Overall debido a la falta de esas columnas.")
        df_filtrado = df[df["Name"] != jugador_seleccionado]
    
    if df_filtrado.empty:
        st.warning("No hay jugadores que cumplan simultáneamente con los filtros establecidos.")
    else:
        # Métricas ampliadas compatibles con datasets de FIFA 23
        columnas_metricas = [
            'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 
            'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 
            'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 
            'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
            'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
            'Composure', 'Marking', 'StandingTackle', 'SlidingTackle'
        ]
        
        columnas_metricas = [col for col in columnas_metricas if col in df.columns]
        
        if not columnas_metricas:
            st.error("⚠️ No se encontraron columnas de atributos técnicos en tu archivo CSV.")
            st.stop()
            
        columna_indice = "ID" if "ID" in df.columns else df.index
        
        # Normalización MinMaxScaler
        scaler = MinMaxScaler()
        df_metricas_norm = scaler.fit_transform(df[columnas_metricas])
        df_norm_completo = pd.DataFrame(df_metricas_norm, columns=columnas_metricas, index=df[columna_indice])
        
        # Vectores para la similitud de coseno
        vector_objetivo = df_norm_completo.loc[[id_objetivo]]
        vectores_filtrados = df_norm_completo.loc[df_filtrado[columna_indice]]
        
        similitudes = cosine_similarity(vectores_filtrados, vector_objetivo)
        
        df_filtrado = df_filtrado.copy()
        df_filtrado["Similitud (%)"] = (similitudes.flatten() * 100).round(2)
        
        resultados = df_filtrado.sort_values(by="Similitud (%)", ascending=False).head(int(n_resultados))
        
        st.subheader(f"Jugadores más similares a {jugador_seleccionado}:")
        
        columnas_visibles = ["Name"]
        if "Age" in df.columns: columnas_visibles.append("Age")
        if "Overall" in df.columns: columnas_visibles.append("Overall")
        if "Club" in df.columns: columnas_visibles.append("Club")
        if "Position" in df.columns: columnas_visibles.append("Position")
        columnas_visibles.append("Similitud (%)")
        
        st.dataframe(resultados[columnas_visibles], use_container_width=True)