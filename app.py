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
    archivos = ["data.csv", "fifa23_data.csv"]
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
        st.error("⚠️ No se encontró el archivo 'data.csv'. Asegúrate de que esté subido al repositorio.")
        st.stop()
        
    # Mapeo exhaustivo para detectar la columna del nombre en cualquier dataset de FIFA 23
    renombres = {
        "short_name": "Name",
        "long_name": "Name",
        "player_name": "Name",
        "KnownAs": "Name",
        "FullName": "Name",
        "Name": "Name",
        "club_name": "Club",
        "ClubName": "Club",
        "player_positions": "Position",
        "BestPosition": "Position",
        "overall": "Overall",
        "OverallRating": "Overall",
        "age": "Age"
    }
    
    # Renombrar columnas encontradas
    columnas_existentes = {k: v for k, v in renombres.items() if k in df.columns and v not in df.columns}
    df = df.rename(columns=columnas_existentes)

    # Si aún no existe 'Name', buscar la primera columna de texto que parezca contener nombres
    if "Name" not in df.columns:
        cols_texto = df.select_dtypes(include=['object']).columns
        if len(cols_texto) > 0:
            df = df.rename(columns={cols_texto[0]: "Name"})

    # Rellenar valores vacíos y limpiar codificación
    if "Name" in df.columns:
        df["Name"] = df["Name"].fillna("Sin Nombre").astype(str).apply(ftfy.fix_text)
    
    if "Club" in df.columns:
        df["Club"] = df["Club"].fillna("Sin Club").astype(str).apply(ftfy.fix_text)
        
    return df

# Inicialización de la base de datos
df = cargar_datos_fifa23()

# 3. Formulario de selección y filtros
with st.container():
    if "Name" not in df.columns:
        st.error("⚠️ No se pudo identificar la columna con el nombre de los jugadores en el CSV. Revisa el encabezado de tu archivo.")
        st.stop()
        
    jugadores_disponibles = sorted(df["Name"].unique())
    jugador_seleccionado = st.selectbox("Seleccionar jugador:", jugadores_disponibles)
    
    edad_maxima = st.slider("Edad máxima:", min_value=16, max_value=45, value=25, step=1)
    calificacion_maxima = st.slider("Calificación máxima general:", min_value=45, max_value=99, value=75, step=1)
    
    n_resultados = st.number_input("Los N mejores resultados:", min_value=1, max_value=30, value=5)
    buscar = st.button("Encontrar jugadores similares")

# 4. Procesamiento e índice de similitud matemática
if buscar:
    fila_objetivo = df[df["Name"] == jugador_seleccionado].iloc[0]
    id_objetivo = fila_objetivo["sofifa_id"] if "sofifa_id" in df.columns else (fila_objetivo["ID"] if "ID" in df.columns else jugador_seleccionado)
    
    # Aplicar filtros
    if "Age" in df.columns and "Overall" in df.columns:
        df_filtrado = df[
            (df["Age"] <= edad_maxima) & 
            (df["Overall"] <= calificacion_maxima) & 
            (df["Name"] != jugador_seleccionado)
        ]
    else:
        df_filtrado = df[df["Name"] != jugador_seleccionado]
    
    if df_filtrado.empty:
        st.warning("No hay jugadores que cumplan simultáneamente con los filtros establecidos.")
    else:
        # Detectar automáticamente columnas numéricas de atributos
        columnas_excluidas = ['Age', 'Overall', 'sofifa_id', 'ID', 'potential', 'value_eur', 'wage_eur']
        columnas_metricas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        columnas_metricas = [col for col in columnas_metricas if col not in columnas_excluidas]
        
        if not columnas_metricas:
            st.error("⚠️ No se encontraron columnas numéricas de atributos en el archivo CSV.")
            st.stop()
            
        columna_indice = "sofifa_id" if "sofifa_id" in df.columns else ("ID" if "ID" in df.columns else df.index)
        
        # Normalización y similitud de coseno
        scaler = MinMaxScaler()
        df_metricas_norm = scaler.fit_transform(df[columnas_metricas].fillna(0))
        df_norm_completo = pd.DataFrame(df_metricas_norm, columns=columnas_metricas, index=df[columna_indice])
        
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
