# ============================================================
# export_csv.py
# Exportar MNIST binario (3 vs 8) como CSV para consumo en C++
# Ejecutar ANTES de compilar/correr el proyecto C++
# ============================================================

from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd

print("⏳ Descargando MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

X_full = mnist.data.astype(np.float32)
y_full = mnist.target.astype(int)

mask = (y_full == 3) | (y_full == 8)
X = X_full[mask]
y = y_full[mask]

# Crear DataFrame con columna label primero
cols = ['label'] + [f'pixel{i}' for i in range(784)]
df = pd.DataFrame(np.column_stack([y, X]), columns=cols)
df['label'] = df['label'].astype(int)

output_path = 'mnist_binary_3_8.csv'
df.to_csv(output_path, index=False)

print(f"✅ CSV exportado: {output_path}")
print(f"   Filas: {len(df):,}  |  Columnas: {len(df.columns)}")
print(f"   Dígito 3: {(df['label']==3).sum():,}  |  Dígito 8: {(df['label']==8).sum():,}")
print(f"\n👉 Copiar '{output_path}' al directorio de build de Qt Creator")
print(f"   (donde queda el ejecutable .exe / binario)")
