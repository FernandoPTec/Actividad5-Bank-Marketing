# Historial de cambios - Actividad 5

## Versión 0.1.0 - Preparación inicial

### Incorporado
- Dataset original Bank Marketing.
- Estructura de carpetas del proyecto.
- Diagnóstico de calidad de datos.
- Archivo de datos preparados.
- Análisis exploratorio con tres visualizaciones.
- Configuración inicial de transformadores.

### Resultados
- Registros originales: 45,211.
- Duplicados exactos originales: 0.
- Valores nulos: 0.
- Registros finales: 45,211.
- Columnas finales: 16.
- Coincidencias tras excluir duration: 16.

### Decisiones técnicas
- Se excluyó duration para evitar fuga de información.
- Se conservaron los valores categóricos unknown.
- Se mantuvo pdays = -1 por su significado.
- No se eliminaron automáticamente valores atípicos.
- Se conservaron coincidencias posteriores a la exclusión
  de duration por no demostrar duplicación real.

## Pendiente
- Entrenamiento de modelos.
- Ajuste de hiperparámetros.
- Registro de experimentos en MLflow.
- Evaluación comparativa.

## Verificación de Git
- Repositorio clonado y conexión remota verificada.
- Script de limpieza ejecutado correctamente.

## Verificación de Git
- Repositorio clonado y conexión remota verificada.
- Script de limpieza ejecutado correctamente.


## Versión 1.0.0 - Entrenamiento y evaluación final

### Incorporado
- Entrenamiento reproducible mediante train.py.
- Regresión logística y random forest.
- Optimización con GridSearchCV y cinco folds.
- Evaluación final con cinco métricas.
- Comparación estadística mediante McNemar.
- Registro de experimentos y modelos en MLflow.
- Almacenamiento de experimentos mediante SQLite.

### Resultados
- Regresión logística: F1 = 0.3732.
- Random forest: F1 = 0.4810.
- Random forest: ROC-AUC = 0.8031.
- Modelo seleccionado: random forest.

### Verificación
- Script train.py ejecutado independientemente.
- Código de salida: 0.
- Resultados reproducidos correctamente.
