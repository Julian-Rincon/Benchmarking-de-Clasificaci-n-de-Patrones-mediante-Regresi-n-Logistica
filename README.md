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

Este repositorio tambien incluye soporte CMake y una copia local de Eigen en
`third_party/eigen` para compilar sin configurar rutas globales.

Dependencias de Python para generar el CSV:

```bash
pip install -r requirements.txt
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
python scripts/export_csv.py
```

Este comando descarga MNIST desde OpenML y genera:

```text
data/mnist_binary_3_8.csv
```

Si se ejecuta desde Qt Creator, copiar el CSV al directorio de build donde queda el ejecutable.

### 2. Compilar con Qt Creator

1. Abrir `logistic_regression.pro`.
2. Seleccionar un kit compatible con C++17.
3. Compilar y ejecutar el proyecto.

### 3. Compilar por linea de comandos

En Windows, la forma mas simple es:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_and_run_cpp.ps1
```

El binario generado con CMake se enlaza estaticamente con el runtime de MinGW,
por lo que no requiere copiar `libstdc++-6.dll` al directorio de ejecucion.

Con CMake y MinGW/GCC:

```bash
cmake -S . -B build
cmake --build build --config Release
./build/logistic_regression_eigen data/mnist_binary_3_8.csv
```

Con qmake:

```bash
qmake logistic_regression.pro
make -j4
./logistic_regression_eigen data/mnist_binary_3_8.csv
```

En Windows, el binario generado puede quedar como `logistic_regression_eigen.exe`.

## Salida esperada

Los valores exactos pueden cambiar ligeramente segun compilador, CPU y version de las
dependencias. El notebook ya valida el baseline Python y lee automaticamente
`results/resultados_cpp.txt` cuando el binario C++ lo genera.

```text
[1/5] Cargando dataset...
      Muestras: 13966 | Caracteristicas: 784
[2/5] Dividiendo datos (80/20 estratificado)...
      Train: 11173 | Test: 2793

[3/5] Normalizando datos (StandardScaler)...
[4/5] Entrenando Regresion Logistica con Eigen...
      Hiperparametros: lr=0.1, epochs=500, lambda=0.001
  Epoch    0 | Loss: 0.756172
  Epoch   50 | Loss: 0.116411
  ...
  Epoch  499 | Loss: 0.085635

====================================================
  METRICAS - C++ (Eigen) Logistic Regression
====================================================
  Accuracy   : 96.60 %
  Precision  : 96.01 %
  Recall     : 97.07 %
  F1-Score   : 96.54 %
  Tiempo entrenamiento: 1.8262 segundos
```

Baseline Python verificado con `data/mnist_binary_3_8.csv`:

```text
Accuracy  : 96.24 %
Precision : 96.46 %
Recall    : 95.82 %
F1-Score  : 96.14 %
```

## Archivos generados

| Archivo | Contenido |
| --- | --- |
| `data/mnist_binary_3_8.csv` | Dataset MNIST filtrado para digitos 3 y 8 |
| `results/resultados_cpp.txt` | Metricas del experimento |
| `results/pesos_cpp.csv` | Pesos aprendidos por el modelo |
| `results/figures/*.png` | Graficas generadas por el notebook |

## Estructura del repositorio

```text
.
├── README.md
├── CMakeLists.txt
├── logistic_regression.pro
├── src/
│   └── main.cpp
├── scripts/
│   ├── export_csv.py
│   └── build_and_run_cpp.ps1
├── notebooks/
│   └── Proyecto_Redes_Neuronales_MNIST.ipynb
├── docs/
│   ├── Informe_Tecnico_Redes_Neuronales.docx
│   └── metricas_finales.md
├── data/
│   └── mnist_binary_3_8.csv
├── results/
│   ├── resultados_cpp.txt
│   ├── pesos_cpp.csv
│   └── figures/
└── third_party/
    └── eigen/
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
