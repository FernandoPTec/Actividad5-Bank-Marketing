import pandas as pd
from pathlib import Path


def preparar_datos(ruta_entrada, ruta_salida):
    """
    Limpia el dataset Bank Marketing y guarda
    una versión preparada para entrenamiento.
    """

    # Cargar dataset original
    df = pd.read_csv(ruta_entrada, sep=";")

    registros_originales = len(df)

    # Identificar y eliminar duplicados exactos
    duplicados = df.duplicated().sum()
    df = df.drop_duplicates().copy()

    # Excluir duration para evitar fuga de información
    df = df.drop(columns=["duration"])

    # Conservar valores unknown y pdays = -1
    # por su significado en el dataset

    # Validar datos
    assert df["y"].isin(["yes", "no"]).all()
    assert df["age"].between(18, 100).all()

    # Crear carpeta de salida
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Guardar dataset preparado
    df.to_csv(ruta_salida, index=False)

    print("=== PREPARACIÓN COMPLETADA ===")
    print("Registros originales:", registros_originales)
    print("Duplicados eliminados:", duplicados)
    print("Registros finales:", len(df))
    print("Columnas finales:", len(df.columns))
    print("Valores nulos:", df.isna().sum().sum())

    return df


if __name__ == "__main__":
    preparar_datos(
        "Actividad5/datos/datos_ini/bank-full.csv",
        "Actividad5/datos/datos_limp/bank_limpio.csv"
    )
