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
    df = pd.read_csv('Iris.csv')
    return df

@st.cache_resource
def entrenar_modelo(X_train, y_train):
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
    
    # TABS REORGANIZADOS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 1. Visualización de Datos", 
        "📈 2. Comprensión y Flujo de Trabajo",
        "🎓 3. Entrenamiento del Modelo",
        "🎯 4. Predicciones Interactivas"
    ])
    
    # ============================================================================
    # TAB 1: VISUALIZACIÓN DE DATOS
    # ============================================================================
    with tab1:
        st.header("📊 Visualización de Datos")
        st.markdown("### Exploración visual del dataset de Iris")
        
        # Nombres en español para las características
        feature_names_es = {
            'SepalLengthCm': 'Longitud del Sépalo (cm)',
            'SepalWidthCm': 'Ancho del Sépalo (cm)',
            'PetalLengthCm': 'Longitud del Pétalo (cm)',
            'PetalWidthCm': 'Ancho del Pétalo (cm)'
        }
        
        st.markdown("---")
        
        # HISTOGRAMAS
        st.subheader("📊 Histogramas de Distribución por Característica")
        
        # Seleccionar característica
        feature_select = st.selectbox(
            "Selecciona una característica para visualizar:",
            list(feature_names_es.keys()),
            format_func=lambda x: feature_names_es[x]
        )
        
        fig_hist = px.histogram(
            df, 
            x=feature_select, 
            color='Species',
            marginal='box',
            nbins=20,
            title=f'Distribución de {feature_names_es[feature_select]} por Especie',
            labels={'Species': 'Especie', feature_select: feature_names_es[feature_select]},
            color_discrete_map={
                'Iris-setosa': '#FF6B6B',
                'Iris-versicolor': '#4ECDC4',
                'Iris-virginica': '#45B7D1'
            },
            opacity=0.7
        )
        fig_hist.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        # MAPA DE CALOR DE CORRELACIONES
        st.subheader("🔥 Mapa de Calor de Correlaciones")
        st.markdown("Muestra la relación entre las diferentes características numéricas")
        
        # Calcular matriz de correlación
        corr_matrix = X.corr()
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[feature_names_es[col] for col in corr_matrix.columns],
            y=[feature_names_es[col] for col in corr_matrix.columns],
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlación")
        ))
        
        fig_heatmap.update_layout(
            title='Matriz de Correlación entre Características',
            height=500,
            xaxis_title='',
            yaxis_title=''
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        
        # BOXPLOTS POR ESPECIE
        st.subheader("📦 Boxplots por Especie")
        st.markdown("Visualización de la distribución y valores atípicos para cada especie")
        
        # Crear 4 boxplots (uno por característica)
        col1, col2 = st.columns(2)
        
        features_list = list(feature_names_es.keys())
        
        with col1:
            # Boxplot 1
            fig_box1 = px.box(
                df, 
                x='Species', 
                y=features_list[0],
                color='Species',
                title=feature_names_es[features_list[0]],
                labels={'Species': 'Especie', features_list[0]: feature_names_es[features_list[0]]},
                color_discrete_map={
                    'Iris-setosa': '#FF6B6B',
                    'Iris-versicolor': '#4ECDC4',
                    'Iris-virginica': '#45B7D1'
                }
            )
            fig_box1.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_box1, use_container_width=True)
            
            # Boxplot 3
            fig_box3 = px.box(
                df, 
                x='Species', 
                y=features_list[2],
                color='Species',
                title=feature_names_es[features_list[2]],
                labels={'Species': 'Especie', features_list[2]: feature_names_es[features_list[2]]},
                color_discrete_map={
                    'Iris-setosa': '#FF6B6B',
                    'Iris-versicolor': '#4ECDC4',
                    'Iris-virginica': '#45B7D1'
                }
            )
            fig_box3.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_box3, use_container_width=True)
        
        with col2:
            # Boxplot 2
            fig_box2 = px.box(
                df, 
                x='Species', 
                y=features_list[1],
                color='Species',
                title=feature_names_es[features_list[1]],
                labels={'Species': 'Especie', features_list[1]: feature_names_es[features_list[1]]},
                color_discrete_map={
                    'Iris-setosa': '#FF6B6B',
                    'Iris-versicolor': '#4ECDC4',
                    'Iris-virginica': '#45B7D1'
                }
            )
            fig_box2.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_box2, use_container_width=True)
            
            # Boxplot 4
            fig_box4 = px.box(
                df, 
                x='Species', 
                y=features_list[3],
                color='Species',
                title=feature_names_es[features_list[3]],
                labels={'Species': 'Especie', features_list[3]: feature_names_es[features_list[3]]},
                color_discrete_map={
                    'Iris-setosa': '#FF6B6B',
                    'Iris-versicolor': '#4ECDC4',
                    'Iris-virginica': '#45B7D1'
                }
            )
            fig_box4.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_box4, use_container_width=True)
    
    # ============================================================================
    # TAB 2: COMPRENSIÓN Y FLUJO DE TRABAJO
    # ============================================================================
    with tab2:
        st.header("📈 Comprensión de los Datos y Flujo de Trabajo")
        
        # ESTADÍSTICAS DEL DATASET
        st.subheader("📊 Estadísticas Descriptivas del Dataset")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Muestras", len(df))
            st.metric("Número de Características", len(X.columns))
        
        with col2:
            st.metric("Número de Especies", df['Species'].nunique())
            st.metric("Muestras por Especie", "50 cada una")
        
        with col3:
            st.metric("Valores Faltantes", "0")
            st.metric("Tipo de Problema", "Clasificación")
        
        st.markdown("---")
        
        # Distribución de clases
        st.subheader("📊 Distribución de Clases")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Conteo por Especie:**")
            class_counts = df['Species'].value_counts()
            st.dataframe(class_counts, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title='Distribución Balanceada de Especies',
                color=class_counts.index,
                color_discrete_map={
                    'Iris-setosa': '#FF6B6B',
                    'Iris-versicolor': '#4ECDC4',
                    'Iris-virginica': '#45B7D1'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        
        # Estadísticas descriptivas
        st.subheader("📈 Estadísticas Detalladas por Característica")
        
        # Renombrar columnas para la tabla
        stats_df = X.describe().T
        stats_df.index = [feature_names_es[col] for col in stats_df.index]
        stats_df = stats_df.round(3)
        
        st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("---")
        
        # FLUJO DE TRABAJO
        st.subheader("🔄 Flujo de Trabajo del Proyecto")
        
        st.markdown("""
        ### Metodología Aplicada
        
        #### **1. Comprensión de los Datos** 📊
        - **Dataset:** Iris flower dataset con 150 muestras
        - **Características:** 4 variables numéricas continuas
            - Longitud del Sépalo (cm)
            - Ancho del Sépalo (cm)
            - Longitud del Pétalo (cm)
            - Ancho del Pétalo (cm)
        - **Variable Objetivo:** Especie (3 clases balanceadas)
        - **Calidad:** Sin valores faltantes, datos limpios
        
        #### **2. Análisis Exploratorio** 🔍
        - Visualización de distribuciones por característica
        - Análisis de correlaciones entre variables
        - Identificación de patrones y separabilidad entre clases
        - Detección de valores atípicos mediante boxplots
        
        #### **3. Preprocesamiento** ⚙️
        - **Normalización:** StandardScaler para estandarizar características
            - Media = 0, Desviación estándar = 1
            - Mejora el rendimiento del modelo
        - **División de Datos:**
            - 80% Entrenamiento (120 muestras)
            - 20% Prueba (30 muestras)
            - Estratificación para mantener proporciones de clases
        
        #### **4. Selección del Modelo** 🤖
        - **Algoritmo:** Random Forest Classifier
        - **Justificación:**
            - ✅ Robusto ante sobreajuste
            - ✅ Maneja relaciones no lineales
            - ✅ Proporciona importancia de características
            - ✅ Excelente para datos tabulares
            - ✅ Ensemble learning: combina múltiples árboles
        - **Hiperparámetros:**
            - 100 árboles de decisión (n_estimators=100)
            - Profundidad máxima de 5 niveles
            - Random state=42 (reproducibilidad)
        
        #### **5. Entrenamiento y Evaluación** 📈
        - Entrenamiento con datos normalizados
        - Validación con conjunto de prueba
        - Métricas múltiples: Accuracy, Precision, Recall, F1-Score
        - Análisis de matriz de confusión
        - Evaluación de importancia de características
        
        #### **6. Implementación** 🚀
        - Dashboard interactivo en Streamlit
        - Visualizaciones en tiempo real
        - Sistema de predicción interactivo
        - Documentación completa
        """)
        
        st.markdown("---")
        
        # Vista previa de los datos
        st.subheader("👀 Vista Previa del Dataset")
        st.dataframe(df.head(10), use_container_width=True)
    
    # ============================================================================
    # TAB 3: ENTRENAMIENTO DEL MODELO
    # ============================================================================
    with tab3:
        st.header("🎓 Entrenamiento y Evaluación del Modelo")
        
        # Información del entrenamiento
        st.subheader("⚙️ Configuración del Entrenamiento")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Algoritmo**")
            st.write("Random Forest")
            st.write(f"**Estimadores:** {modelo.n_estimators}")
            st.write(f"**Profundidad Máx:** {modelo.max_depth}")
        
        with col2:
            st.info("**Datos de Entrenamiento**")
            st.write(f"**Muestras Train:** {len(X_train)}")
            st.write(f"**Muestras Test:** {len(X_test)}")
            st.write(f"**Proporción:** 80/20")
        
        with col3:
            st.info("**Preprocesamiento**")
            st.write("**Normalización:** StandardScaler")
            st.write("**Estratificación:** Sí")
            st.write("**Random State:** 42")
        
        st.markdown("---")
        
        # MÉTRICAS DE RENDIMIENTO
        st.subheader("📊 Métricas de Rendimiento del Modelo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Exactitud (Accuracy)", 
                f"{accuracy:.4f}",
                delta=f"{(accuracy-0.5)*100:.1f}% sobre azar",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Precisión (Precision)", 
                f"{precision:.4f}",
                help="Proporción de predicciones positivas correctas"
            )
        
        with col3:
            st.metric(
                "Exhaustividad (Recall)", 
                f"{recall:.4f}",
                help="Proporción de positivos reales identificados"
            )
        
        with col4:
            st.metric(
                "Puntaje F1 (F1-Score)", 
                f"{f1:.4f}",
                help="Media armónica de precisión y exhaustividad"
            )
        
        st.markdown("---")
        
        # FEATURE IMPORTANCE
        st.subheader("🎯 Importancia de las Características (Feature Importance)")
        st.markdown("Muestra qué características tienen mayor influencia en las predicciones del modelo")
        
        # Crear DataFrame de importancia
        feature_importance_df = pd.DataFrame({
            'Característica': [feature_names_es[col] for col in X.columns],
            'Importancia': modelo.feature_importances_,
            'Porcentaje': modelo.feature_importances_ * 100
        }).sort_values('Importancia', ascending=True)
        
        # Gráfico de barras horizontal
        fig_importance = px.bar(
            feature_importance_df,
            x='Importancia',
            y='Característica',
            orientation='h',
            title='Importancia de Características en el Modelo Random Forest',
            labels={'Importancia': 'Importancia Relativa', 'Característica': ''},
            color='Importancia',
            color_continuous_scale='Viridis',
            text=feature_importance_df['Porcentaje'].round(2).astype(str) + '%'
        )
        fig_importance.update_traces(textposition='outside')
        fig_importance.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Interpretación
        st.info(f"""
        **Interpretación:** La característica más importante es **{feature_importance_df.iloc[-1]['Característica']}** 
        con un {feature_importance_df.iloc[-1]['Porcentaje']:.1f}% de importancia, lo que significa que esta característica 
        tiene el mayor poder discriminativo para clasificar las especies de Iris.
        """)
        
        st.markdown("---")
        
        # MATRIZ DE CONFUSIÓN
        st.subheader("📊 Matriz de Confusión")
        st.markdown("Muestra el rendimiento detallado del modelo para cada clase")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            cm = confusion_matrix(y_test, y_pred)
            
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicción", y="Valor Real", color="Cantidad"),
                x=['Setosa', 'Versicolor', 'Virginica'],
                y=['Setosa', 'Versicolor', 'Virginica'],
                text_auto=True,
                color_continuous_scale='Blues',
                aspect='auto'
            )
            fig_cm.update_layout(height=400, title="Matriz de Confusión")
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.write("**Interpretación:**")
            st.write("- Diagonal: Predicciones correctas")
            st.write("- Fuera diagonal: Errores")
            st.write("")
            
            # Calcular accuracy por clase
            st.write("**Accuracy por Especie:**")
            for i, species in enumerate(['Setosa', 'Versicolor', 'Virginica']):
                class_accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                st.write(f"- {species}: {class_accuracy:.2%}")
        
        st.markdown("---")
        
        # REPORTE DE CLASIFICACIÓN
        st.subheader("📋 Reporte Detallado de Clasificación")
        
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df[report_df.index.isin(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])]
        report_df.columns = ['Precisión', 'Exhaustividad', 'F1-Score', 'Soporte']
        report_df['Soporte'] = report_df['Soporte'].astype(int)
        
        # Formatear y resaltar
        st.dataframe(
            report_df.style.format({
                'Precisión': '{:.4f}',
                'Exhaustividad': '{:.4f}',
                'F1-Score': '{:.4f}',
                'Soporte': '{:.0f}'
            }).background_gradient(subset=['Precisión', 'Exhaustividad', 'F1-Score'], cmap='RdYlGn', vmin=0.8, vmax=1.0),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # EXPLICACIÓN DE RESULTADOS
        st.subheader("💡 Explicación de los Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### ✅ Fortalezas del Modelo
            - **Alta exactitud general:** El modelo clasifica correctamente la mayoría de las muestras
            - **Buena generalización:** Rendimiento consistente en datos no vistos
            - **Separabilidad clara:** Especialmente para Iris-setosa
            - **Balance entre métricas:** Precision y Recall equilibrados
            """)
        
        with col2:
            st.markdown("""
            #### 🎯 Observaciones Clave
            - **Características más relevantes:** Las medidas del pétalo son más discriminativas
            - **Confusión mínima:** Principalmente entre Versicolor y Virginica
            - **Modelo robusto:** Random Forest reduce el riesgo de sobreajuste
            - **Dataset balanceado:** Facilita el entrenamiento equitativo
            """)
    
    # ============================================================================
    # TAB 4: PREDICCIONES INTERACTIVAS
    # ============================================================================
    with tab4:
        st.header("🎯 Sistema de Predicción Interactivo")
        st.markdown("Ingresa las medidas de una flor de Iris para predecir su especie")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📝 Ingresar Medidas")
            
            sepal_length = st.slider(
                "Longitud del Sépalo (cm)", 
                float(df['SepalLengthCm'].min()), 
                float(df['SepalLengthCm'].max()), 
                float(df['SepalLengthCm'].mean()),
                0.1,
                help="Desliza para ajustar la longitud del sépalo"
            )
            
            sepal_width = st.slider(
                "Ancho del Sépalo (cm)", 
                float(df['SepalWidthCm'].min()), 
                float(df['SepalWidthCm'].max()), 
                float(df['SepalWidthCm'].mean()),
                0.1,
                help="Desliza para ajustar el ancho del sépalo"
            )
            
            petal_length = st.slider(
                "Longitud del Pétalo (cm)", 
                float(df['PetalLengthCm'].min()), 
                float(df['PetalLengthCm'].max()), 
                float(df['PetalLengthCm'].mean()),
                0.1,
                help="Desliza para ajustar la longitud del pétalo"
            )
            
            petal_width = st.slider(
                "Ancho del Pétalo (cm)", 
                float(df['PetalWidthCm'].min()), 
                float(df['PetalWidthCm'].max()), 
                float(df['PetalWidthCm'].mean()),
                0.1,
                help="Desliza para ajustar el ancho del pétalo"
            )
            
            st.markdown("---")
            
            # Botón para predecir
            if st.button("🔮 Predecir Especie", type="primary", use_container_width=True):
                # Realizar predicción
                input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                input_scaled = scaler.transform(input_data)
                prediction = modelo.predict(input_scaled)[0]
                prediction_proba = modelo.predict_proba(input_scaled)[0]
                
                st.balloons()
                
                st.success(f"### 🌸 Especie Predicha: **{prediction}**")
                
                st.markdown("#### 📊 Niveles de Confianza:")
                species_list = modelo.classes_
                
                for species, prob in zip(species_list, prediction_proba):
                    st.progress(prob, text=f"{species}: {prob:.1%}")
                
                # Guardar predicción en session_state para gráficos
                st.session_state.prediction = prediction
                st.session_state.sepal_length = sepal_length
                st.session_state.sepal_width = sepal_width
                st.session_state.petal_length = petal_length
                st.session_state.petal_width = petal_width
        
        with col2:
            st.subheader("📊 Visualización de la Predicción")
            
            # Verificar si hay una predicción
            if 'prediction' in st.session_state:
                # Crear DataFrame para visualización
                df_viz = df.copy()
                df_viz['Tipo'] = 'Dataset'
                df_viz['Tamaño'] = 5
                
                # Agregar el punto nuevo
                new_point = pd.DataFrame({
                    'SepalLengthCm': [st.session_state.sepal_length],
                    'SepalWidthCm': [st.session_state.sepal_width],
                    'PetalLengthCm': [st.session_state.petal_length],
                    'PetalWidthCm': [st.session_state.petal_width],
                    'Species': [st.session_state.prediction],
                    'Tipo': ['⭐ Nueva Muestra'],
                    'Tamaño': [20]
                })
                
                df_viz = pd.concat([df_viz, new_point], ignore_index=True)
                
                # Gráfico 3D
                st.markdown("##### 🎲 Visualización 3D")
                fig_3d = px.scatter_3d(
                    df_viz, 
                    x='PetalLengthCm', 
                    y='PetalWidthCm', 
                    z='SepalLengthCm',
                    color='Species',
                    symbol='Tipo',
                    size='Tamaño',
                    title='Dispersión 3D: Posición de la Nueva Muestra',
                    labels={
                        'PetalLengthCm': 'Longitud Pétalo (cm)',
                        'PetalWidthCm': 'Ancho Pétalo (cm)',
                        'SepalLengthCm': 'Longitud Sépalo (cm)',
                        'Species': 'Especie'
                    },
                    color_discrete_map={
                        'Iris-setosa': '#FF6B6B',
                        'Iris-versicolor': '#4ECDC4',
                        'Iris-virginica': '#45B7D1'
                    },
                    opacity=0.7
                )
                
                fig_2d.update_layout(height=400)
                st.plotly_chart(fig_2d, use_container_width=True)
                
                # Segundo gráfico 2D - Características del Sépalo
                st.markdown("##### 📈 Visualización 2D - Longitud vs Ancho del Sépalo")
                fig_2d_sepal = px.scatter(
                    df_viz,
                    x='SepalLengthCm',
                    y='SepalWidthCm',
                    color='Species',
                    symbol='Tipo',
                    size='Tamaño',
                    title='Dispersión 2D: Características del Sépalo',
                    labels={
                        'SepalLengthCm': 'Longitud del Sépalo (cm)',
                        'SepalWidthCm': 'Ancho del Sépalo (cm)',
                        'Species': 'Especie'
                    },
                    color_discrete_map={
                        'Iris-setosa': '#FF6B6B',
                        'Iris-versicolor': '#4ECDC4',
                        'Iris-virginica': '#45B7D1'
                    },
                    opacity=0.7
                )
                
                fig_2d_sepal.update_layout(height=400)
                st.plotly_chart(fig_2d_sepal, use_container_width=True)
                
            else:
                st.info("👈 Ajusta los valores de las características y presiona el botón 'Predecir Especie' para visualizar los resultados")
                
                # Mostrar gráfico 2D del dataset completo mientras tanto
                st.markdown("##### 📊 Vista del Dataset Completo")
                fig_dataset = px.scatter(
                    df,
                    x='PetalLengthCm',
                    y='PetalWidthCm',
                    color='Species',
                    title='Distribución del Dataset: Longitud vs Ancho del Pétalo',
                    labels={
                        'PetalLengthCm': 'Longitud del Pétalo (cm)',
                        'PetalWidthCm': 'Ancho del Pétalo (cm)',
                        'Species': 'Especie'
                    },
                    color_discrete_map={
                        'Iris-setosa': '#FF6B6B',
                        'Iris-versicolor': '#4ECDC4',
                        'Iris-virginica': '#45B7D1'
                    },
                    opacity=0.7
                )
                st.plotly_chart(fig_dataset, use_container_width=True)

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
""", unsafe_allow_html=True)_map={
                        'Iris-setosa': '#FF6B6B',
                        'Iris-versicolor': '#4ECDC4',
                        'Iris-virginica': '#45B7D1'
                    },
                    opacity=0.6
                )
                
                fig_3d.update_layout(height=400)
                st.plotly_chart(fig_3d, use_container_width=True)
                
                st.markdown("---")
                
                # Gráfico 2D
                st.markdown("##### 📈 Visualización 2D - Longitud vs Ancho del Pétalo")
                fig_2d = px.scatter(
                    df_viz,
                    x='PetalLengthCm',
                    y='PetalWidthCm',
                    color='Species',
                    symbol='Tipo',
                    size='Tamaño',
                    title='Dispersión 2D: Características del Pétalo',
                    labels={
                        'PetalLengthCm': 'Longitud del Pétalo (cm)',
                        'PetalWidthCm': 'Ancho del Pétalo (cm)',
                        'Species': 'Especie'
                    },
                    color_discrete