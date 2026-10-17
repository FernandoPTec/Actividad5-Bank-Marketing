# Actividad 5 - Bank Marketing

## Descripción
Predicción de contratación de depósitos bancarios
mediante regresión logística y random forest.

Proyecto académico de Machine Learning y MLOps.

## Dataset
Fuente: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/222/bank+marketing

Archivo original: bank-full.csv
Registros: 45,211
Variables originales: 17

## Preparación de datos
- Sin valores nulos de Pandas.
- Sin duplicados exactos originales.
- Exclusión de duration para evitar fuga de información.
- Conservación de unknown y pdays = -1.
- Dataset preparado: 45,211 registros y 16 columnas.

## Estructura del proyecto
- datos/datos_ini/: dataset original.
- datos/datos_limp/: dataset preparado.
- fuentes/datos_prep.py: limpieza reproducible.
- fuentes/train.py: entrenamiento y MLflow.
- fuentes/entrena.ipynb: notebook de Colab.
- evidencias/: gráficas y resultados.
- CHANGELOG.md: historial de modificaciones.

## Algoritmos
1. Regresión logística.
2. Random forest.

## Metodología
- División estratificada 80/20.
- StratifiedKFold de 5 particiones.
- GridSearchCV con F1 como métrica de selección.
- Evaluación con accuracy, precision, recall,
  F1 y ROC-AUC.
- Seguimiento de experimentos mediante MLflow.

## Requisitos
Python 3.13 y las dependencias de requirements.txt.

## Reproducción
Desde la carpeta raíz del proyecto:

1. Instalar dependencias:
   python -m pip install -r requirements.txt

2. Preparar datos:
   python fuentes/datos_prep.py

3. Entrenar y registrar experimentos:
   python fuentes/train.py

El entrenamiento puede tardar varios minutos.

## MLflow
El script genera una base de datos SQLite denominada
mlflow.db y registra los experimentos en:

Actividad5_Bank_Marketing

Para abrir el panel desde la carpeta raíz:

mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

Después abrir:
http://127.0.0.1:5000

El respaldo histórico de MLflow se entrega
por separado en el ZIP final.

## Resultados de prueba

| Modelo | F1 | ROC-AUC |
|---|---|---|
| Regresión logística | 0.3732 | 0.7722 |
| Random forest | 0.4810 | 0.8031 |

Random forest obtuvo el mejor F1 y ROC-AUC.

## Limitaciones
- Dataset histórico de campañas bancarias portuguesas.
- Clases desbalanceadas.
- No se garantiza generalización a otros bancos.
- Los modelos requieren validación adicional
  antes de cualquier uso operativo.
- Los artefactos serializados deben cargarse
  únicamente desde fuentes confiables.

## Repositorio
https://github.com/FernandoPTec/Actividad5-Bank-Marketing
