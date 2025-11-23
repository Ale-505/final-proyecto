# 🌸 Proyecto de Clasificación de Especies de Iris

**Proyecto Final de Minería de Datos**  
Universidad de la Costa  
Profesor: José Escorcia-Gutierrez, Ph.D.

## 👥 Integrantes del Equipo
- [Nombre Estudiante 1]
- [Nombre Estudiante 2]
- [Nombre Estudiante 3]
- [Nombre Estudiante 4]

## 📝 Descripción del Proyecto

Este proyecto implementa un pipeline completo de machine learning para clasificar especies de flores Iris utilizando el famoso dataset de Iris. El proyecto incluye exploración de datos, preprocesamiento, entrenamiento del modelo (Random Forest), evaluación y un dashboard interactivo construido con Streamlit.

### Dataset
El dataset de Iris contiene 150 muestras de flores iris con las siguientes características:
- Longitud del Sépalo (cm)
- Ancho del Sépalo (cm)
- Longitud del Pétalo (cm)
- Ancho del Pétalo (cm)

Variable objetivo: Especie (Iris-setosa, Iris-versicolor, Iris-virginica)

## 🎯 Objetivos del Proyecto

1. Diseñar e implementar un flujo de trabajo completo de minería de datos
2. Entrenar un modelo de clasificación para predecir especies de iris
3. Desarrollar un dashboard interactivo para visualización y predicción
4. Evaluar el rendimiento del modelo usando múltiples métricas

## 🔄 Metodología

### 1. Comprensión de los Datos
- Se cargó y exploró el dataset de Iris
- Se analizaron las distribuciones de características y correlaciones
- Se verificó la calidad de los datos (sin valores faltantes)

### 2. Preprocesamiento de Datos
- Se aplicó StandardScaler para normalización de características
- División de datos: 80% entrenamiento, 20% prueba
- Se usó muestreo estratificado para mantener el balance de clases

### 3. Selección del Modelo
**Algoritmo:** Random Forest Classifier

**Justificación:**
- Robusto ante sobreajuste con ajuste apropiado de hiperparámetros
- Maneja relaciones no lineales entre características
- Proporciona información sobre la importancia de características
- Excelente rendimiento en datos tabulares
- No requiere supuestos sobre la distribución de los datos

**Hiperparámetros:**
- Número de estimadores: 100
- Profundidad máxima: 5
- Estado aleatorio: 42 (para reproducibilidad)

### 4. Evaluación del Modelo
Se utilizaron múltiples métricas para una evaluación comprehensiva:
- **Exactitud (Accuracy)**: Corrección general
- **Precisión**: Calidad de las predicciones positivas
- **Exhaustividad (Recall)**: Cobertura de los positivos reales
- **Puntaje F1**: Media armónica de precisión y exhaustividad
- **Matriz de Confusión**: Análisis detallado de errores

## 🚀 Instalación y Configuración

### Requisitos Previos
- Python 3.8 o superior
- Gestor de paquetes pip

### Pasos de Instalación

1. Clonar este repositorio:
```bash
git clone [url-de-tu-repositorio]
cd proyecto-clasificacion-iris
```

2. Instalar los paquetes requeridos:
```bash
pip install -r requirements.txt
```

3. Asegurarse de que el archivo `Iris.csv` esté en el directorio del proyecto

4. Ejecutar la aplicación de Streamlit:
```bash
streamlit run Proyecto.py
```

El dashboard se abrirá automáticamente en tu navegador predeterminado en `http://localhost:8501`

## 📊 Características del Dashboard

### 1. Pestaña Resumen y Métricas
- Métricas de rendimiento del modelo (Exactitud, Precisión, Exhaustividad, F1-Score)
- Visualización de la matriz de confusión
- Explicación detallada del flujo de trabajo

### 2. Pestaña Exploración de Datos
- Vista previa y estadísticas del dataset
- Histogramas de distribución de características por especie
- Matriz de dispersión mostrando relaciones entre pares de características

### 3. Pestaña Hacer Predicciones
- Controles deslizantes interactivos para medidas de flores
- Predicción de especies en tiempo real con niveles de confianza
- Gráfico de dispersión 3D mostrando la posición de la nueva muestra relativa al dataset

### 4. Pestaña Análisis del Modelo
- Visualización de importancia de características
- Métricas de rendimiento por clase
- Detalles de configuración del modelo

## 📁 Estructura del Proyecto

```
proyecto-clasificacion-iris/
│
├── Proyecto.py           # Aplicación principal de Streamlit
├── Iris.csv             # Archivo del dataset
├── requirements.txt     # Dependencias de Python
├── README.md           # Documentación del proyecto
└── .gitignore          # Archivos a ignorar en Git
```

## 🎥 Presentación en Video

[El enlace a la presentación en video se añadirá aquí]

## 📈 Resultados

El modelo Random Forest logra un excelente rendimiento en la tarea de clasificación de Iris:
- Alta exactitud en las tres especies
- Separación clara de Iris-setosa de las otras especies
- Buena discriminación entre Iris-versicolor e Iris-virginica
- Rendimiento consistente en todas las métricas de evaluación

## 🛠️ Tecnologías Utilizadas

- **Python**: Lenguaje de programación
- **Streamlit**: Framework para dashboard interactivo
- **Scikit-learn**: Biblioteca de machine learning
- **Pandas**: Manipulación de datos
- **Plotly**: Visualizaciones interactivas
- **NumPy**: Computación numérica

## 📚 Referencias

1. Fisher, R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936)
2. Documentación de Scikit-learn: https://scikit-learn.org/
3. Documentación de Streamlit: https://docs.streamlit.io/

## 📄 Licencia

Este proyecto es parte de una tarea académica para el curso de Minería de Datos en la Universidad de la Costa.

## 🙏 Agradecimientos

- Profesor José Escorcia-Gutierrez por la orientación e instrucción
- Universidad de la Costa, Departamento de Ciencias de la Computación y Electrónica
- R.A. Fisher por el dataset original de Iris

---

*"Las tres virtudes principales de un programador son: Pereza, Impaciencia y Arrogancia." - Larry Wall*