# Statistical Learning 2

## Descripción

Este repositorio contiene los proyectos, tareas y actividades desarrolladas en el curso de **Statistical Learning II**. El curso se enfoca en técnicas avanzadas de aprendizaje automático y estadístico, incluyendo redes neuronales, series temporales, clustering y aprendizaje por refuerzo.

**Autor:** Francisco González  
**Carnet:** 24002914

## Contenido del Repositorio

### 📊 Proyectos

#### Proyecto 1: Series Temporales y Forecasting
- **Archivo:** `Proyecto_1_Statistical_Learning_2.ipynb`
- **Descripción:** Implementación de modelos de series temporales utilizando TensorFlow, Prophet y NeuralProphet
- **Técnicas utilizadas:**
  - Redes neuronales para forecasting
  - Prophet para análisis de series temporales
  - TensorFlow con soporte GPU (CUDA 12.1)
  - Preprocesamiento con MinMaxScaler y StandardScaler

#### Proyecto 2: Clustering y Segmentación de Clientes
- **Archivo:** `Proyecto_2_Statistical_Learning_2.ipynb`
- **Descripción:** Análisis de datos de retail y segmentación de clientes
- **Técnicas utilizadas:**
  - K-Means clustering
  - K-Medoids (usando pyclustering)
  - Análisis RFM (Recency, Frequency, Monetary)
  - Reducción de dimensionalidad (PCA, t-SNE)
  - Métricas de evaluación (Silhouette Score, Davies-Bouldin)

### 📝 Tareas

#### Tarea 1: Clasificación Binaria con Redes Neuronales
- **Archivo:** `Tarea_1_Statistical_Learning_2.ipynb`
- **Descripción:** Construcción y entrenamiento de un modelo de clasificación binaria
- **Técnicas utilizadas:**
  - TensorFlow/Keras
  - Redes neuronales secuenciales
  - Dropout para regularización
  - StandardScaler para normalización
  - Métricas de evaluación: accuracy, confusion matrix, classification report

#### Tarea 2: Forecasting con RNN, LSTM y GRU
- **Archivo:** `Tarea_2_Statistical_Learning_2.ipynb`
- **Descripción:** Implementación y comparación de diferentes arquitecturas de redes recurrentes para forecasting
- **Contenido:**
  - Investigación teórica sobre RNN, LSTM y GRU
  - Implementación práctica con datos de consumo energético (KwhConsumptionBlower78)
  - Comparación de rendimiento entre SimpleRNN, LSTM y GRU
  - Análisis de series temporales

### 🎯 Actividades

#### Actividad: Comparación K-Means vs K-Medoids
- **Archivo:** `Actividad.ipynb`
- **Descripción:** Ejercicio práctico comparando algoritmos de clustering
- **Técnicas utilizadas:**
  - K-Means (sklearn)
  - K-Medoids (pyclustering)
  - Generación de datos sintéticos con outliers
  - Visualización de clusters

#### Clase 8: Aprendizaje por Refuerzo
- **Archivo:** `Clase8_SL2_Ej.ipynb`
- **Descripción:** Ejercicio "El Aventurero del Tesoro" - Introducción al aprendizaje por refuerzo
- **Conceptos explorados:**
  - Agente y Entorno
  - Estados, Acciones y Recompensas
  - Políticas de decisión
  - Ecuación de Bellman
  - Valor a largo plazo vs recompensa inmediata

## Tecnologías y Bibliotecas

### Frameworks de Deep Learning
- **TensorFlow/Keras** - Construcción y entrenamiento de redes neuronales
- **PyTorch** - Soporte para CUDA/GPU

### Análisis de Series Temporales
- **Prophet** - Forecasting de series temporales
- **NeuralProphet** - Forecasting con redes neuronales

### Machine Learning y Análisis de Datos
- **scikit-learn** - Algoritmos de ML, preprocesamiento y métricas
- **pandas** - Manipulación y análisis de datos
- **numpy** - Operaciones numéricas

### Clustering
- **pyclustering** - Implementación de K-Medoids

### Visualización
- **matplotlib** - Gráficos y visualizaciones
- **seaborn** - Visualizaciones estadísticas
- **plotly** - Gráficos interactivos

### Utilidades
- **tqdm** - Barras de progreso
- **rich** - Output formateado en consola

## Estructura del Repositorio

```
statistical-learning-2/
│
├── Proyecto_1_Statistical_Learning_2.ipynb    # Series temporales y forecasting
├── Proyecto_2_Statistical_Learning_2.ipynb    # Clustering y segmentación
├── Tarea_1_Statistical_Learning_2.ipynb       # Clasificación binaria
├── Tarea_2_Statistical_Learning_2.ipynb       # RNN, LSTM, GRU
├── Actividad.ipynb                            # K-Means vs K-Medoids
├── Clase8_SL2_Ej.ipynb                        # Aprendizaje por refuerzo
├── LICENSE                                    # Licencia CC0 1.0
└── README.md                                  # Este archivo
```

## Cómo Usar este Repositorio

### Opción 1: Google Colab (Recomendado)
Cada notebook incluye un botón "Open in Colab" en la parte superior. Simplemente haz clic en él para abrir el notebook directamente en Google Colab.

### Opción 2: Entorno Local
1. Clona el repositorio:
   ```bash
   git clone https://github.com/franciscogonzalez-gal/statistical-learning-2.git
   cd statistical-learning-2
   ```

2. Instala las dependencias necesarias (se recomienda usar un entorno virtual):
   ```bash
   pip install tensorflow torch prophet neuralprophet
   pip install scikit-learn pandas numpy matplotlib seaborn plotly
   pip install pyclustering tqdm rich
   ```

3. Inicia Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

### Requisitos
- Python 3.10 o superior
- Para aprovechar GPU: CUDA 12.1 (opcional pero recomendado para los proyectos de deep learning)
- Jupyter Notebook o Google Colab

## Notas Importantes

- **GPU:** Algunos notebooks están optimizados para ejecutarse con GPU. Si usas Google Colab, asegúrate de habilitar GPU en: Runtime → Change runtime type → Hardware accelerator → GPU
- **Datos:** Los proyectos que requieren datos externos (como el Proyecto 2 y la Tarea 2) asumen que los datos están disponibles en Google Drive
- **Instalación:** Cada notebook incluye celdas de instalación de dependencias al inicio

## Licencia

Este proyecto está bajo la licencia [CC0 1.0 Universal](LICENSE) - es de dominio público y puede ser usado libremente sin restricciones.

## Contacto

**Francisco González**  
Carnet: 24002914

---

*Repositorio desarrollado como parte del curso Statistical Learning II*
