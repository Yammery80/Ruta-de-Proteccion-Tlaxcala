"""Pronóstico mensual de casos registrados. Entrada: mes, casos_registrados.

La tabla puede tener varias filas por mes (por ejemplo, una por municipio).
Use el historial estatal completo, con meses cerrados y sin duplicados.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _estimar(historial):
    """Entrena una regresión con los conteos de los tres meses previos."""
    ventanas = np.lib.stride_tricks.sliding_window_view(historial, 4)
    modelo = Ridge(alpha=1.0).fit(ventanas[:, :3], ventanas[:, 3])
    estimacion = modelo.predict(historial[-3:].reshape(1, -1))[0]
    return max(0, round(float(estimacion)))


def pronosticar(tabla):
    """Evalúa Ridge y el último mes en los mismos seis meses recientes."""
    requeridas = {"mes", "casos_registrados"}
    if not requeridas.issubset(tabla.columns):
        raise ValueError("Se requieren las columnas mes y casos_registrados.")

    meses = pd.to_datetime(tabla["mes"], errors="coerce").dt.to_period("M")
    casos = pd.to_numeric(tabla["casos_registrados"], errors="coerce")
    if meses.isna().any() or not np.isfinite(casos).all():
        raise ValueError("Hay meses o conteos inválidos.")
    if (casos < 0).any() or (casos % 1 != 0).any():
        raise ValueError("Los conteos deben ser enteros no negativos.")

    serie = casos.groupby(meses).sum().sort_index()
    if len(serie) < 18:
        return {"estado": "historial_insuficiente", "meses_disponibles": len(serie),
                "meses_minimos": 18}

    esperados = pd.period_range(serie.index[0], serie.index[-1], freq="M")
    faltantes = esperados.difference(serie.index)
    if len(faltantes):
        return {"estado": "meses_faltantes", "meses_faltantes": list(map(str, faltantes))}

    valores = serie.to_numpy(dtype=float)
    inicio = len(valores) - 6
    predicciones = [_estimar(valores[:i]) for i in range(inicio, len(valores))]
    reales = valores[inicio:]
    mae_modelo = float(np.mean(np.abs(reales - predicciones)))
    mae_referencia = float(np.mean(np.abs(reales - valores[inicio - 1:-1])))
    gana_modelo = mae_modelo < mae_referencia

    return {
        "estado": "ok",
        "mes_pronosticado": str(serie.index[-1] + 1),
        "casos_estimados": _estimar(valores) if gana_modelo else int(valores[-1]),
        "metodo": "Ridge con tres rezagos" if gana_modelo else "Último mes observado",
        "mae_modelo": round(mae_modelo, 2),
        "mae_referencia": round(mae_referencia, 2),
        "meses_evaluados": list(map(str, serie.index[inicio:])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    entrada = parser.add_mutually_exclusive_group(required=True)
    entrada.add_argument("--demo", action="store_true", help="Usa datos ficticios.")
    entrada.add_argument("--csv", help="Ruta a la tabla mensual limpia.")
    args = parser.parse_args()

    if args.demo:
        meses = pd.period_range("2023-01", periods=30, freq="M")
        conteos = [65 + 2 * i + (0, 6, 12)[i % 3] for i in range(len(meses))]
        tabla = pd.DataFrame({"mes": meses.astype(str), "casos_registrados": conteos})
    else:
        tabla = pd.read_csv(args.csv)

    try:
        resultado = pronosticar(tabla)
    except ValueError as error:
        parser.exit(2, f"Error: {error}\n")
    if args.demo:
        resultado["datos"] = "ficticios"
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
