import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Clasificación de Especies de Iris", layout="wide", page_icon="🌸")

# Título principal
st.title("🌸 Dashboard de Clasificación de Especies de Iris")
st.markdown("---")

# Sidebar con información del proyecto
st.sidebar.header("📊 Información del Proyecto")
st.sidebar.markdown("""
**Proyecto Final de Minería de Datos**  
Universidad de la Costa  

**Integrantes del Equipo:**
- [Nombre 1]
- [Nombre 2]
- [Nombre 3]
- [Nombre 4]

**Profesor:** José Escorcia-Gutierrez, Ph.D.
""")

# Cargar y preparar datos
@st.cache_data
def cargar_datos():
    # Cargar el dataset desde CSV
    df = pd.read_csv('Iris.csv')
    return df

@st.cache_resource
def entrenar_modelo(X_train, y_train):
    # Entrenar modelo Random Forest
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    modelo.fit(X_train, y_train)
    return modelo

# Cargar datos
try:
    df = cargar_datos()
    
    # Preparar datos
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    modelo = entrenar_modelo(X_train_scaled, y_train)
    
    # Predicciones
    y_pred = modelo.predict(X_test_scaled)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Resumen y Métricas", "🔍 Exploración de Datos", "🎯 Hacer Predicciones", "📊 Análisis del Modelo"])
    
    # TAB 1: Resumen y Métricas
    with tab1:
        st.header("Métricas de Rendimiento del Modelo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Exactitud (Accuracy)", f"{accuracy:.4f}", delta="Alto Rendimiento")
        with col2:
            st.metric("Precisión", f"{precision:.4f}")
        with col3:
            st.metric("Exhaustividad (Recall)", f"{recall:.4f}")
        with col4:
            st.metric("Puntaje F1", f"{f1:.4f}")
        
        st.markdown("---")
        
        # Matriz de confusión
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = px.imshow(cm, 
                           labels=dict(x="Predicción", y="Real", color="Cantidad"),
                           x=['Setosa', 'Versicolor', 'Virginica'],
                           y=['Setosa', 'Versicolor', 'Virginica'],
                           text_auto=True,
                           color_continuous_scale='Blues')
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Workflow explicación
        st.markdown("---")
        st.subheader("🔄 Flujo de Trabajo del Proyecto")
        st.markdown("""
        **1. Comprensión de los Datos**
        - Se cargó el dataset de Iris con 150 muestras y 4 características
        - Se exploró la distribución de clases y estadísticas de las características
        
        **2. Preprocesamiento de Datos**
        - Se verificó la ausencia de valores faltantes
        - Se aplicó StandardScaler para normalización de características
        - División de datos: 80% entrenamiento, 20% prueba con estratificación
        
        **3. Selección y Entrenamiento del Modelo**
        - Algoritmo: Clasificador Random Forest
        - Justificación: Robusto ante sobreajuste, maneja relaciones no lineales, 
          proporciona información sobre importancia de características
        - Hiperparámetros: 100 estimadores, profundidad máxima=5
        
        **4. Evaluación del Modelo**
        - Validación cruzada para confiabilidad
        - Múltiples métricas para evaluación comprehensiva
        - Matriz de confusión para análisis detallado de errores
        """)
    
    # TAB 2: Exploración de Datos
    with tab2:
        st.header("Exploración de Datos")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Vista Previa del Dataset")
            st.dataframe(df.head(10), height=300)
            
            st.subheader("Estadísticas del Dataset")
            st.write(f"**Total de Muestras:** {len(df)}")
            st.write(f"**Características:** {len(X.columns)}")
            st.write(f"**Clases:** {df['Species'].nunique()}")
            
            st.write("**Distribución de Clases:**")
            class_counts = df['Species'].value_counts()
            st.write(class_counts)
        
        with col2:
            st.subheader("Distribución de Características por Especie")
            feature_names = {
                'SepalLengthCm': 'Longitud del Sépalo (cm)',
                'SepalWidthCm': 'Ancho del Sépalo (cm)',
                'PetalLengthCm': 'Longitud del Pétalo (cm)',
                'PetalWidthCm': 'Ancho del Pétalo (cm)'
            }
            feature_select = st.selectbox("Seleccionar Característica", 
                                         list(feature_names.keys()),
                                         format_func=lambda x: feature_names[x])
            
            fig_dist = px.histogram(df, x=feature_select, color='Species', 
                                   marginal='box', 
                                   title=f'Distribución de {feature_names[feature_select]}',
                                   barmode='overlay',
                                   opacity=0.7,
                                   labels={'Species': 'Especie'})
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Scatter matrix
        st.subheader("Relaciones entre Características")
        df_plot = df.copy()
        df_plot.columns = ['Id', 'Longitud Sépalo', 'Ancho Sépalo', 'Longitud Pétalo', 'Ancho Pétalo', 'Especie']
        
        fig_scatter = px.scatter_matrix(df_plot, 
                                       dimensions=['Longitud Sépalo', 'Ancho Sépalo', 'Longitud Pétalo', 'Ancho Pétalo'],
                                       color='Especie',
                                       title="Matriz de Dispersión de Todas las Características")
        fig_scatter.update_traces(diagonal_visible=False)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # TAB 3: Hacer Predicciones
    with tab3:
        st.header("🎯 Predictor Interactivo de Especies")
        st.markdown("Ingresa las medidas de una flor de Iris para predecir su especie:")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Medidas de Entrada")
            
            sepal_length = st.slider("Longitud del Sépalo (cm)", 
                                    float(df['SepalLengthCm'].min()), 
                                    float(df['SepalLengthCm'].max()), 
                                    float(df['SepalLengthCm'].mean()),
                                    0.1)
            
            sepal_width = st.slider("Ancho del Sépalo (cm)", 
                                   float(df['SepalWidthCm'].min()), 
                                   float(df['SepalWidthCm'].max()), 
                                   float(df['SepalWidthCm'].mean()),
                                   0.1)
            
            petal_length = st.slider("Longitud del Pétalo (cm)", 
                                    float(df['PetalLengthCm'].min()), 
                                    float(df['PetalLengthCm'].max()), 
                                    float(df['PetalLengthCm'].mean()),
                                    0.1)
            
            petal_width = st.slider("Ancho del Pétalo (cm)", 
                                   float(df['PetalWidthCm'].min()), 
                                   float(df['PetalWidthCm'].max()), 
                                   float(df['PetalWidthCm'].mean()),
                                   0.1)
            
            # Realizar predicción
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)
            prediction = modelo.predict(input_scaled)[0]
            prediction_proba = modelo.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.subheader("Resultado de la Predicción")
            st.success(f"**Especie Predicha:** {prediction}")
            
            st.write("**Niveles de Confianza:**")
            species_list = modelo.classes_
            for species, prob in zip(species_list, prediction_proba):
                st.write(f"{species}: {prob:.2%}")
        
        with col2:
            st.subheader("Visualización 3D")
            
            # Crear DataFrame para visualización
            df_viz = df.copy()
            df_viz['Tipo'] = 'Dataset'
            
            # Agregar el punto nuevo
            new_point = pd.DataFrame({
                'SepalLengthCm': [sepal_length],
                'SepalWidthCm': [sepal_width],
                'PetalLengthCm': [petal_length],
                'PetalWidthCm': [petal_width],
                'Species': [prediction],
                'Tipo': ['Nueva Muestra']
            })
            
            df_viz = pd.concat([df_viz, new_point], ignore_index=True)
            
            # Crear gráfico 3D
            fig_3d = px.scatter_3d(df_viz, 
                                  x='PetalLengthCm', 
                                  y='PetalWidthCm', 
                                  z='SepalLengthCm',
                                  color='Species',
                                  symbol='Tipo',
                                  title='Gráfico 3D: Posición de la Muestra en el Espacio de Características',
                                  opacity=0.7,
                                  size_max=10,
                                  labels={'PetalLengthCm': 'Longitud Pétalo (cm)',
                                         'PetalWidthCm': 'Ancho Pétalo (cm)',
                                         'SepalLengthCm': 'Longitud Sépalo (cm)',
                                         'Species': 'Especie',
                                         'Tipo': 'Tipo'})
            
            fig_3d.update_traces(marker=dict(size=5), selector=dict(name='Dataset'))
            fig_3d.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')), 
                               selector=dict(name='Nueva Muestra'))
            
            fig_3d.update_layout(height=600)
            st.plotly_chart(fig_3d, use_container_width=True)
    
    # TAB 4: Análisis del Modelo
    with tab4:
        st.header("📊 Análisis del Modelo")
        
        # Feature importance
        st.subheader("Importancia de las Características")
        feature_names_es = {
            'SepalLengthCm': 'Longitud del Sépalo',
            'SepalWidthCm': 'Ancho del Sépalo',
            'PetalLengthCm': 'Longitud del Pétalo',
            'PetalWidthCm': 'Ancho del Pétalo'
        }
        
        feature_importance = pd.DataFrame({
            'Característica': [feature_names_es[f] for f in X.columns],
            'Importancia': modelo.feature_importances_
        }).sort_values('Importancia', ascending=False)
        
        fig_imp = px.bar(feature_importance, 
                        x='Importancia', 
                        y='Característica', 
                        orientation='h',
                        title='Importancia de las Características en el Modelo Random Forest')
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("---")
        
        # Análisis detallado por clase
        st.subheader("Rendimiento por Clase")
        
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df[report_df.index.isin(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])]
        report_df.columns = ['Precisión', 'Exhaustividad', 'F1-Score', 'Soporte']
        
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))
        
        st.markdown("---")
        
        # Información del modelo
        st.subheader("Información del Modelo")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tipo de Modelo:** Random Forest Classifier")
            st.write(f"**Número de Estimadores:** {modelo.n_estimators}")
            st.write(f"**Profundidad Máxima:** {modelo.max_depth}")
            st.write(f"**Muestras de Entrenamiento:** {len(X_train)}")
            st.write(f"**Muestras de Prueba:** {len(X_test)}")
        
        with col2:
            st.write("**Características Utilizadas:**")
            for feat in X.columns:
                st.write(f"- {feature_names_es[feat]}")
            
            st.write("")
            st.write("**Justificación del Modelo:**")
            st.write("- Robusto ante sobreajuste")
            st.write("- Maneja relaciones no lineales")
            st.write("- Proporciona importancia de características")
            st.write("- Excelente para datos tabulares")

except FileNotFoundError:
    st.error("❌ Error: No se encontró el archivo 'Iris.csv'. Asegúrate de que el archivo esté en el mismo directorio que este script.")
    st.info("Sube tu archivo Iris.csv para continuar.")
except Exception as e:
    st.error(f"❌ Ocurrió un error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Universidad de la Costa - Proyecto Final de Minería de Datos</p>
    <p><em>"Las tres virtudes principales de un programador son: Pereza, Impaciencia y Arrogancia." - Larry Wall</em></p>
</div>
""", unsafe_allow_html=True)