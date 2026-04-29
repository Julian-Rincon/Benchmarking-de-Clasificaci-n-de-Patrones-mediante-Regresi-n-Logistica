# Benchmarking de Clasificacion de Patrones mediante Regresion Logistica

Proyecto final de la asignatura **Redes Neuronales** de la Universidad Sergio Arboleda. El objetivo es evaluar un clasificador de regresion logistica binaria implementado en C++ con Eigen, usando un subconjunto de MNIST para distinguir los digitos 3 y 8.

## Integrantes

- Julian Esteban Rincon R.
- Miguel Flechas
- Andres Castro
- Juan Hurtado

## Descripcion

El proyecto implementa el flujo completo de entrenamiento y evaluacion de una regresion logistica:

- carga de datos desde CSV;
- filtrado binario para las clases 3 y 8;
- division estratificada de entrenamiento y prueba;
- normalizacion tipo StandardScaler;
- entrenamiento con descenso por gradiente y regularizacion L2;
- calculo de accuracy, precision, recall y F1-score.

Tambien se incluye un notebook de apoyo y un informe tecnico con la documentacion del experimento.

## Requisitos

| Herramienta | Version minima |
| --- | --- |
| Qt Creator | 6.x |
| GCC, MinGW o MSVC | C++17 |
| Eigen | 3.4+ |
| Python | 3.10+ |

Dependencias de Python para generar el CSV:

```bash
pip install scikit-learn numpy pandas
```

## Instalacion de Eigen

### Windows

1. Descargar Eigen 3.4 o superior desde <https://eigen.tuxfamily.org>.
2. Extraerlo en `C:\eigen-3.4.0`.
3. Verificar que `logistic_regression.pro` tenga configurado ese path en `INCLUDEPATH`.

### Linux / macOS

```bash
sudo apt install libeigen3-dev
brew install eigen
```

## Ejecucion

### 1. Generar el dataset CSV

Desde la raiz del proyecto:

```bash
python export_csv.py
```

Este comando descarga MNIST desde OpenML y genera:

```text
mnist_binary_3_8.csv
```

Si se ejecuta desde Qt Creator, copiar el CSV al directorio de build donde queda el ejecutable.

### 2. Compilar con Qt Creator

1. Abrir `logistic_regression.pro`.
2. Seleccionar un kit compatible con C++17.
3. Compilar y ejecutar el proyecto.

### 3. Compilar por linea de comandos

```bash
qmake logistic_regression.pro
make -j4
./logistic_regression_eigen mnist_binary_3_8.csv
```

En Windows, el binario generado puede quedar como `logistic_regression_eigen.exe`.

## Salida esperada

```text
[1/5] Cargando dataset...
      Muestras: 13782 | Caracteristicas: 784
[2/5] Dividiendo datos (80/20 estratificado)...
[3/5] Normalizando datos (StandardScaler)...
[4/5] Entrenando Regresion Logistica con Eigen...
      Hiperparametros: lr=0.1, epochs=500, lambda=0.001
  Epoch    0 | Loss: 0.693147
  Epoch   50 | Loss: 0.198342
  ...
  Epoch  499 | Loss: 0.089201

====================================================
  METRICAS - C++ (Eigen) Logistic Regression
====================================================
  Accuracy   : 96.81 %
  Precision  : 97.02 %
  Recall     : 96.44 %
  F1-Score   : 96.73 %
```

## Archivos generados

| Archivo | Contenido |
| --- | --- |
| `mnist_binary_3_8.csv` | Dataset MNIST filtrado para digitos 3 y 8 |
| `resultados_cpp.txt` | Metricas del experimento |
| `pesos_cpp.csv` | Pesos aprendidos por el modelo |

## Estructura del repositorio

```text
.
├── README.md
├── logistic_regression.pro
├── main.cpp
├── export_csv.py
├── Proyecto_Redes_Neuronales_MNIST.ipynb
└── Informe_Tecnico_Redes_Neuronales.docx
```

## Componentes principales

```text
LogisticRegressionEigen
├── fit()          - Gradiente descendente con Eigen
├── sigmoid()      - Funcion sigmoide vectorizada
├── computeLoss()  - Binary Cross-Entropy + L2
├── predict()      - Clasificacion con umbral 0.5
└── predictProba() - Probabilidades del modelo

DataLoader
├── loadCSV()          - Carga del CSV
├── standardScale()    - Normalizacion columnar
└── trainTestSplit()   - Division estratificada

Evaluator
├── evaluate()     - Calculo de TP, TN, FP y FN
└── printReport()  - Reporte de metricas

Benchmarker
├── start()              - Marca de tiempo inicial
└── elapsedSeconds()     - Duracion en segundos
```
