# Actividad 5 - Machine Learning y MLOps

## Descripción
Predicción de contratación de depósitos bancarios mediante
regresión logística y random forest.

## Dataset
Nombre: Bank Marketing
Fuente: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/dataset/222/bank+marketing

## Preparación de datos
- 45,211 registros originales.
- 17 columnas originales.
- Sin valores nulos.
- Sin duplicados exactos originales.
- Exclusión de la variable duration.
- Conservación de valores unknown y pdays = -1.
- Dataset preparado: 45,211 registros y 16 columnas.

## Análisis exploratorio
- Contratación positiva: 11.7%.
- Contratación negativa: 88.3%.
- Valores atípicos de balance: 4,729 (10.46%).
- Mayor tasa de contratación por edad: mayores de 60 años.

## Modelos
1. Regresión logística.
2. Random forest.

## Metodología
- Preparación mediante pipelines de scikit-learn.
- Validación cruzada de 5 particiones.
- Optimización mediante GridSearchCV.
- Evaluación con accuracy, precision, recall, F1 y ROC-AUC.
- Registro de experimentos mediante MLflow.

## Estructura
- datos/datos_ini: datos originales.
- datos/datos_limp: datos preparados.
- fuentes: notebook y scripts Python.
- evidencias: gráficas y capturas.
- CHANGELOG.md: historial de cambios.

## Reproducción
Los pasos de instalación y ejecución se completarán
cuando finalice la implementación de los modelos.

## Consideraciones
Proyecto académico basado en datos públicos.
No utiliza información de clientes de BBVA.
