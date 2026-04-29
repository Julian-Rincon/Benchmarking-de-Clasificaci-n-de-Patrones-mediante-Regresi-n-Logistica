######################################################################
# Proyecto Final — Redes Neuronales
# Benchmarking Regresión Logística: C++ con Eigen
# Universidad Sergio Arboleda
# Configuración para Qt Creator
######################################################################

QT       -= gui
QT       -= core

TARGET    = logistic_regression_eigen
TEMPLATE  = app
CONFIG   += console c++17
CONFIG   -= app_bundle

# ---------------------------------------------------------------
# Eigen: Header-only — solo añadir el path de include
# Descarga: https://eigen.tuxfamily.org (Eigen 3.4+)
# Descomprimir en: C:/eigen-3.4.0  (Windows) o /usr/include/eigen3 (Linux)
# ---------------------------------------------------------------

# --- Windows (ajustar el path según instalación) ---
win32: INCLUDEPATH += C:/eigen-3.4.0

# --- Linux / macOS ---
unix:  INCLUDEPATH += /usr/include/eigen3

# ---------------------------------------------------------------
# Flags de optimización
# ---------------------------------------------------------------
QMAKE_CXXFLAGS += -O2 -march=native -ffast-math

# ---------------------------------------------------------------
# Fuentes del proyecto
# ---------------------------------------------------------------
SOURCES += \
    main.cpp

HEADERS +=

# ---------------------------------------------------------------
# Archivos de datos (no compilados, solo referenciados)
# ---------------------------------------------------------------
DISTFILES += \
    README.md \
    mnist_binary_3_8.csv
