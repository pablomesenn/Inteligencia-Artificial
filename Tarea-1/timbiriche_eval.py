#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
timbiriche_eval.py (con --depth)
1) Prueba mínima IA vs IA
2) Benchmark -> bench/dot_bench.csv
3) Gráficas -> bench/dot_time_hist.png, bench/dot_winrate_bar.png

Uso:
  conda install -y matplotlib
  python timbiriche_eval.py --runs 20 --n 4 --depth 3
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_dot_boxes.py"

OUTDIR = ROOT / "bench"
CSV_PATH = OUTDIR / "dot_bench.csv"
PLOT_TIME = OUTDIR / "dot_time_hist.png"
PLOT_WIN = OUTDIR / "dot_winrate_bar.png"

# Patrones del output
SCORE_LINE_RE_A = re.compile(r"J0\s*=\s*(\d+).+?J1\s*=\s*(\d+)", re.IGNORECASE)             # "🟦 J0=0  🟥 J1=0   | Turno..."
SCORE_LINE_RE_B = re.compile(r"Marcador\s+J0\s*=\s*(\d+)\s+J1\s*=\s*(\d+)", re.IGNORECASE)  # "Turno: J1 | Marcador J0=.. J1=.."
MOVE_NO_RE      = re.compile(r"(?:Jugada|Move)\s*#\s*(\d+)", re.IGNORECASE)
SIZE_LINE_RE    = re.compile(r"TIMBIRICHE\s+N\s*=\s*(\d+)", re.IGNORECASE)
TOTAL_BOXES_RE  = re.compile(r"(?:total\s+cuadros|boxes)\s*[:=]\s*(\d+)", re.IGNORECASE)

def _ensure_runner_exists() -> None:
    """
    Verifica que el script 'run_dot_boxes.py' exista en el mismo directorio.
    Si no se encuentra, el programa termina con un mensaje de error.
    """
    if not RUNNER.exists():
        sys.exit(f"[ERROR] No encontré {RUNNER.name}. Coloca este script junto a run_dot_boxes.py")

def _run_one_game(n: int, depth: int, timeout_sec: int = 600) -> Dict[str, Any]:
    """
    Ejecuta una partida IA vs IA con profundidad limitada.

    Args:
        n (int): Tamaño del tablero (n x n puntos).
        depth (int): Profundidad de búsqueda para el algoritmo minimax.
        timeout_sec (int): Tiempo máximo permitido para la ejecución de la partida.

    Returns:
        Dict[str, Any]: Diccionario con los resultados de la partida, incluyendo
                        el tiempo de ejecución, puntajes, ganador y número de movimientos.
    """
    cmd = [
        sys.executable, str(RUNNER),
        "-n", str(n),
        "--ai-vs-ai",
        "--style", "clean",
        "--depth", str(depth),
        "--delay", "0",
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
            env=env,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {"stdout": "", "n": n, "time_sec": float(timeout_sec), "score0": None, "score1": None, "winner": None, "moves": None}

    dt = time.perf_counter() - t0
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

    score0, score1, last_move = None, None, None
    n_detected, total_boxes = None, None

    for line in out.splitlines():
        s = line.strip()

        m = SIZE_LINE_RE.search(s)
        if m:
            try: n_detected = int(m.group(1))
            except: pass

        mA = SCORE_LINE_RE_A.search(s)
        if mA:
            try:
                score0 = int(mA.group(1)); score1 = int(mA.group(2))
            except: pass

        mB = SCORE_LINE_RE_B.search(s)
        if mB:
            try:
                score0 = int(mB.group(1)); score1 = int(mB.group(2))
            except: pass

        mM = MOVE_NO_RE.search(s)
        if mM:
            try:
                mv = int(mM.group(1))
                last_move = mv if last_move is None else max(last_move, mv)
            except: pass

        mT = TOTAL_BOXES_RE.search(s)
        if mT:
            try: total_boxes = int(mT.group(1))
            except: pass

    if total_boxes is None:
        nn = n_detected if n_detected else n
        total_boxes = (nn - 1) * (nn - 1)

    winner = None
    if isinstance(score0, int) and isinstance(score1, int):
        if score0 + score1 > total_boxes:
            # incoherente -> invalida
            score0 = score1 = winner = None
        else:
            winner = 0 if score0 > score1 else (1 if score1 > score0 else -1)

    return {
        "stdout": out,
        "n": n_detected or n,
        "time_sec": dt,
        "score0": score0,
        "score1": score1,
        "winner": winner,
        "moves": last_move,
    }

def _sanity_test(n: int, depth: int) -> None:
    """
    Realiza una prueba mínima de una partida IA vs IA para verificar la coherencia de los resultados.

    Args:
        n (int): Tamaño del tablero (n x n puntos).
        depth (int): Profundidad de búsqueda para el algoritmo minimax.
    """
    print("[*] Prueba mínima (IA vs IA) ...")
    res = _run_one_game(n=n, depth=depth, timeout_sec=600)

    if res["score0"] is None or res["score1"] is None:
        print(res["stdout"])
        sys.exit("[ERROR] No pude inferir los puntajes finales (J0/J1). "
                 "Prueba con un depth más bajo (ej. --depth 3) o revisa heurística.")

    total_boxes = (res["n"] - 1) * (res["n"] - 1)
    if res["score0"] + res["score1"] != total_boxes:
        print(res["stdout"])
        sys.exit(f"[ERROR] Puntajes incoherentes: J0={res['score0']} J1={res['score1']} "
                 f"(esperado total={total_boxes}).")

    if res["winner"] not in (0, 1, -1):
        print(res["stdout"])
        sys.exit("[ERROR] No pude inferir el ganador.")

    print(f"[OK] Partida completada. N={res['n']}, tiempo={res['time_sec']:.3f}s, "
          f"J0={res['score0']}, J1={res['score1']}, ganador={res['winner']}")

def _run_benchmark(n: int, depth: int, runs: int) -> None:
    """
    Ejecuta un benchmark de múltiples partidas IA vs IA y guarda los resultados en un archivo CSV.

    Args:
        n (int): Tamaño del tablero (n x n puntos).
        depth (int): Profundidad de búsqueda para el algoritmo minimax.
        runs (int): Número de partidas a ejecutar para el benchmark.
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"[*] Benchmark: {runs} partidas IA vs IA (N={n}, depth={depth}) ...")
    for i in range(runs):
        res = _run_one_game(n=n, depth=depth, timeout_sec=600)
        rows.append({
            "run": i,
            "n": res["n"],
            "depth": depth,
            "time_sec": round(float(res["time_sec"]), 6),
            "moves": res["moves"] if res["moves"] is not None else "",
            "score0": res["score0"] if res["score0"] is not None else "",
            "score1": res["score1"] if res["score1"] is not None else "",
            "winner": res["winner"] if res["winner"] is not None else "",
        })
        print(f"  run {i+1}/{runs}: time={rows[-1]['time_sec']}s "
              f"J0={rows[-1]['score0']} J1={rows[-1]['score1']} winner={rows[-1]['winner']}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] CSV -> {CSV_PATH}")

def _make_plots() -> None:
    """
    Genera gráficas a partir de los resultados del benchmark guardados en el archivo CSV.
    Crea un histograma de tiempos y una gráfica de barras de la tasa de resultados.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib no está instalado. Instálalo con: conda install -y matplotlib")
        return

    if not CSV_PATH.exists():
        print(f"[WARN] No existe {CSV_PATH}. Corre primero el benchmark.")
        return

    # Carga ligera de CSV
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["time_sec"] = float(row.get("time_sec", "nan"))
            except Exception:
                row["time_sec"] = float("nan")
            try:
                row["winner"] = int(row["winner"]) if row.get("winner", "") != "" else None
            except Exception:
                row["winner"] = None
            rows.append(row)

    # Histograma de tiempos
    times = [x["time_sec"] for x in rows if x["time_sec"] == x["time_sec"]]
    if times:
        plt.figure()
        plt.hist(times, bins=10)
        plt.title("Timbiriche Minimax: Tiempo por partida")
        plt.xlabel("segundos")
        plt.ylabel("frecuencia")
        plt.tight_layout()
        OUTDIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_TIME)
        plt.close()
        print(f"[OK] {PLOT_TIME.name}")
    else:
        print("[WARN] No hay datos de tiempo para graficar.")

    # Winrate
    total = len(rows)
    w0 = sum(1 for x in rows if x["winner"] == 0)
    w1 = sum(1 for x in rows if x["winner"] == 1)
    wd = sum(1 for x in rows if x["winner"] == -1)
    if total > 0:
        p0 = 100.0 * w0 / total
        p1 = 100.0 * w1 / total
        pd = 100.0 * wd / total

        plt.figure()
        plt.bar(["J0", "J1", "Empate"], [p0, p1, pd])
        plt.title("Timbiriche Minimax: Tasa de resultado")
        plt.ylabel("%")
        plt.tight_layout()
        plt.savefig(PLOT_WIN)
        plt.close()
        print(f"[OK] {PLOT_WIN.name}")
    else:
        print("[WARN] CSV vacío, no se puede graficar tasa de resultado.")

def main():
    """
    Función principal que coordina la ejecución de pruebas, benchmarks y generación de gráficas.
    """
    _ensure_runner_exists()
    ap = argparse.ArgumentParser(description="Evaluación Timbiriche (Minimax): prueba + benchmark + gráficas")
    ap.add_argument("--n", type=int, default=4, help="Tamaño del tablero (puntos por lado)")
    ap.add_argument("--depth", type=int, default=3, help="Profundidad de búsqueda minimax (menor = más rápido)")
    ap.add_argument("--runs", type=int, default=20, help="Número de partidas IA vs IA para el benchmark")
    args = ap.parse_args()

    _sanity_test(n=args.n, depth=args.depth)
    _run_benchmark(n=args.n, depth=args.depth, runs=args.runs)

    print("[*] Generando gráficas...")
    _make_plots()
    print("[OK] Todo listo. Revisa la carpeta bench/")

if __name__ == "__main__":
    main()
