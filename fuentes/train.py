# Actividad 5 - Universidad BBVA
# Entrenamiento y registro reproducible en MLflow

import argparse
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parents[1]
DATOS = BASE / "datos" / "datos_limp" / "bank_limpio.csv"
EVIDENCIAS = BASE / "evidencias"


def ejecutar():
    EVIDENCIAS.mkdir(parents=True, exist_ok=True)

    # 1. Cargar y validar datos
    df = pd.read_csv(DATOS)
    assert "y" in df.columns
    assert "duration" not in df.columns
    assert df["y"].isin(["yes", "no"]).all()

    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})

    # 2. División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20,
        stratify=y, random_state=42
    )

    numericas = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categoricas = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    # 3. Pipelines
    prep_log = ColumnTransformer([
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)
    ])

    prep_rf = ColumnTransformer([
        ("num", "passthrough", numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)
    ])

    modelos = {
        "Regresion_Logistica_Optimizada": (
            Pipeline([
                ("preprocesamiento", prep_log),
                ("clasificador", LogisticRegression(
                    max_iter=1000, random_state=42
                ))
            ]),
            {
                "clasificador__C": [0.1, 1, 10],
                "clasificador__class_weight": [None, "balanced"]
            }
        ),
        "Random_Forest_Optimizado": (
            Pipeline([
                ("preprocesamiento", prep_rf),
                ("clasificador", RandomForestClassifier(
                    random_state=42, n_jobs=-1
                ))
            ]),
            {
                "clasificador__n_estimators": [100, 200],
                "clasificador__max_depth": [10, 20],
                "clasificador__min_samples_leaf": [1, 5],
                "clasificador__class_weight": [None, "balanced"]
            }
        )
    }

    # 4. MLflow con SQLite
    db = BASE / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db}")

    nombre_exp = "Actividad5_Bank_Marketing"
    existente = mlflow.get_experiment_by_name(nombre_exp)

    if existente is None:
        mlflow.create_experiment(
            nombre_exp,
            artifact_location=(BASE / "mlartifacts").resolve().as_uri()
        )

    mlflow.set_experiment(nombre_exp)

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    resultados = []

    # 5. Entrenamiento, evaluación y registro
    for nombre, (pipeline, parametros) in modelos.items():
        print(f"\nEntrenando {nombre}", flush=True)

        grid = GridSearchCV(
            pipeline,
            parametros,
            scoring="f1",
            cv=cv,
            n_jobs=2,
            refit=True,
            error_score="raise"
        )

        grid.fit(X_train, y_train)

        modelo = grid.best_estimator_
        pred = modelo.predict(X_test)
        prob = modelo.predict_proba(X_test)[:, 1]

        metricas = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, pred, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, prob)
        }

        resultados.append({
            "modelo": nombre,
            "f1_cv": grid.best_score_,
            **metricas
        })

        ejemplo = X_train.head(5).copy()
        firma = infer_signature(ejemplo, modelo.predict(ejemplo))

        with mlflow.start_run(run_name=nombre):
            mlflow.set_tags({
                "proyecto": "Actividad 5",
                "dataset": "Bank Marketing UCI",
                "algoritmo": nombre
            })

            mlflow.log_params(grid.best_params_)
            mlflow.log_params({
                "cv_folds": 5,
                "test_size": 0.20,
                "random_state": 42,
                "scoring_grid": "f1"
            })

            mlflow.log_metric("f1_gridsearch_cv", grid.best_score_)

            for metrica, valor in metricas.items():
                mlflow.log_metric(f"test_{metrica}", valor)

            mlflow.sklearn.log_model(
                sk_model=modelo,
                name="modelo",
                input_example=ejemplo,
                signature=firma,
                skops_trusted_types=[
                    "sklearn.compose._column_transformer._RemainderColsList",
                    "sklearn.tree._tree.Tree"
                ]
            )

        print("Mejores parámetros:", grid.best_params_)
        print("F1 de prueba:", round(metricas["f1"], 4))

    # 6. Guardar comparación
    tabla = pd.DataFrame(resultados)
    tabla.to_csv(
        EVIDENCIAS / "metricas_train_script.csv",
        index=False
    )

    print("\n=== RESULTADOS FINALES ===")
    print(tabla.to_string(index=False))
    print("\nEntrenamiento y registro completados.")


if __name__ == "__main__":
    ejecutar()
