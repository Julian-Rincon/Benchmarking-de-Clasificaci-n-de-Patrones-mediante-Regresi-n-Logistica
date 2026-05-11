# Metricas finales y verificacion

Este archivo resume el estado reproducible de la entrega frente al enunciado del
proyecto final.

## Requisitos del enunciado

- Dataset de imagenes para clasificacion binaria: MNIST, digitos 3 vs 8.
- EDA y baseline en Python con Scikit-Learn.
- Implementacion C++ de regresion logistica con Eigen.
- Comparativa de precision, recall, F1-score, tiempo de entrenamiento y memoria.
- Informe tecnico y sustentacion.

## Dataset generado

Dependencias Python:

```bash
pip install -r requirements.txt
```

Comando:

```bash
python scripts/export_csv.py
```

Resultado verificado:

```text
Filas: 13,966
Columnas: 785
Digito 3: 7,141
Digito 8: 6,825
```

## Baseline Python verificado

Sobre `data/mnist_binary_3_8.csv`, con split estratificado 80/20 y `random_state=42`:

```text
Train: 11,172
Test: 2,794
Iteraciones: 90
Tiempo entrenamiento promedio: ~0.54 s
Accuracy: 96.24 %
Precision: 96.46 %
Recall: 95.82 %
F1-Score: 96.14 %
Matriz confusion: [[1381, 48], [57, 1308]]
```

## Verificacion del notebook

Comando usado:

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/Proyecto_Redes_Neuronales_MNIST.ipynb --output Proyecto_Redes_Neuronales_MNIST.executed.ipynb --ExecutePreprocessor.timeout=900
```

Estado: ejecuta completo. En Windows puede aparecer un warning de `joblib` al
cerrar procesos paralelos; no falla ninguna celda.

## Verificacion C++

Se instalo WinLibs/MinGW en modo usuario y Eigen 3.4.0 en `third_party/eigen`.
El proyecto compila por CMake y tambien conserva el archivo `.pro` para Qt
Creator.

Comando directo en Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_and_run_cpp.ps1
```

El ejecutable CMake se enlaza estaticamente para evitar errores de DLL como
`libstdc++-6.dll` no encontrada.

Pasos esperados:

```bash
python scripts/export_csv.py
qmake logistic_regression.pro
make
./logistic_regression_eigen data/mnist_binary_3_8.csv
```

Resultado C++ verificado con `data/mnist_binary_3_8.csv`:

```text
Train: 11,173
Test: 2,793
Tiempo entrenamiento: ~1.83 s
Accuracy: 96.60 %
Precision: 96.01 %
Recall: 97.07 %
F1-Score: 96.54 %
Matriz confusion: [[1373, 55], [40, 1325]]
```

Memoria pico medida con `WorkingSet64` en Windows:

```text
Python: 395.54 MB
C++: 88.36 MB
Factor aproximado: 4.48x menos memoria en C++
```

Archivos generados:

```text
results/resultados_cpp.txt
results/pesos_cpp.csv
```

En Windows con Qt Creator:

1. Abrir `logistic_regression.pro`.
2. Confirmar que Eigen este instalado y que `INCLUDEPATH` apunte a la carpeta correcta.
3. Ejecutar el proyecto con `data/mnist_binary_3_8.csv`.
4. Confirmar que se generen `results/resultados_cpp.txt` y `results/pesos_cpp.csv`.
5. Reejecutar el notebook para que la comparativa use `results/resultados_cpp.txt` real.

## Correcciones aplicadas

- `scripts/export_csv.py` usa salida ASCII para evitar errores de codificacion en PowerShell.
- `scripts/export_csv.py` guarda la cache de OpenML en `data/.sklearn_data` dentro del proyecto.
- `src/main.cpp` ahora hace split estratificado real por clase.
- El notebook usa `data/.sklearn_data` y lee `results/resultados_cpp.txt` si existe.
- README e informe fueron alineados con el total real de MNIST 3 vs 8: `13,966`.
- Se agrego `CMakeLists.txt` para compilar sin depender de Qt Creator.
