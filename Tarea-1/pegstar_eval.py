#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pegstar_eval.py (versión parcheada)
Un solo script que:
  1) Prueba Peg Solitaire (tablero clásico) y verifica optimalidad (31 movimientos)
  2) Ejecuta benchmarks repetidos y guarda CSV con tiempos/éxito
  3) Genera gráficas (matplotlib) a partir del CSV

No modifica tu código. Invoca a run_peg_astar.py
Uso:
  conda install -y matplotlib
  python pegstar_eval.py --runs 30 --time-limit 15
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_peg_astar.py"
OUTDIR = ROOT / "bench"
CSV_PATH = OUTDIR / "peg_bench.csv"
PLOT_TIME = OUTDIR / "peg_time_hist.png"
PLOT_SUCCESS = OUTDIR / "peg_success_bar.png"

# Formatos posibles en la salida del runner:
# (A) "Métricas: {...}"
METRICS_DICT_RE = re.compile(r"^M[ée]tricas:\s*(\{.*\})", re.IGNORECASE)

# (B) Bloque "=== Resultados ===" con pares clave: valor
KV_STATUS_RE   = re.compile(r"^status\s*:\s*(\w+)", re.IGNORECASE)
KV_MOVES_RE    = re.compile(r"^moves\s*:\s*(\d+)", re.IGNORECASE)
KV_EXPANDED_RE = re.compile(r"^expanded\s*:\s*(\d+)", re.IGNORECASE)
KV_ELAPSED_RE  = re.compile(r"^elapsed_sec\s*:\s*([\d.]+)", re.IGNORECASE)

# Mensaje alterno
ALT_SOL_RE     = re.compile(r"Soluci[oó]n encontrada con\s+(\d+)\s+mov", re.IGNORECASE)

def _ensure_runner_exists() -> None:
    """
    Verifica que el script 'run_peg_astar.py' exista en el mismo directorio.
    Si no existe, termina el programa con un mensaje de error.
    """
    if not RUNNER.exists():
        sys.exit(f"[ERROR] No encontré {RUNNER.name}. Coloca este script en la misma carpeta que run_peg_astar.py")

def _run_solver_cli(time_limit: int, show_path: bool = False) -> Dict[str, Any]:
    """
    Ejecuta el script 'run_peg_astar.py' con un límite de tiempo especificado y opcionalmente muestra el camino.
    
    Args:
        time_limit (int): Límite de tiempo para la ejecución del solver.
        show_path (bool): Indica si se debe mostrar el camino de la solución.

    Returns:
        Dict[str, Any]: Un diccionario con la salida estándar, métricas, longitud del camino, 
                        si se resolvió, nodos expandidos y tiempo transcurrido.
    """
    cmd = [sys.executable, str(RUNNER), "--time-limit", str(time_limit)]
    if show_path:
        cmd.append("--show-path")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # fuerza UTF-8 en stdout/stderr del runner

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
            env=env,
            timeout=time_limit + 5,
        )
    except FileNotFoundError:
        sys.exit("[ERROR] No pude ejecutar Python. Verifica tu entorno.")
    except subprocess.TimeoutExpired:
        return {'stdout': '', 'metrics': None, 'path_len': None, 'solved': False, 'expanded': None, 'elapsed_sec': None}

    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

    metrics = None
    path_len = None
    solved = None
    expanded = None
    elapsed = None

    # (A) ¿Dict en una sola línea?
    for line in out.splitlines():
        m = METRICS_DICT_RE.search(line.strip())
        if m:
            try:
                metrics = ast.literal_eval(m.group(1))
            except Exception:
                metrics = None
            break

    # (B) Si no hubo dict, parsea pares clave:valor y/o mensajes alternos
    if metrics is None:
        status_val = None
        moves_val = None

        for line in out.splitlines():
            s = line.strip()

            m = KV_STATUS_RE.match(s)
            if m:
                status_val = m.group(1).lower()

            m = KV_MOVES_RE.match(s)
            if m:
                try:
                    moves_val = int(m.group(1))
                except Exception:
                    pass

            m = KV_EXPANDED_RE.match(s)
            if m:
                try:
                    expanded = int(m.group(1))
                except Exception:
                    pass

            m = KV_ELAPSED_RE.match(s)
            if m:
                try:
                    elapsed = float(m.group(1))
                except Exception:
                    pass

        if moves_val is None:
            # intenta con el mensaje "Solución encontrada con X movimientos."
            m = ALT_SOL_RE.search(out)
            if m:
                try:
                    moves_val = int(m.group(1))
                except Exception:
                    pass

        # Infierelos
        if moves_val is not None:
            path_len = moves_val
        if status_val:
            solved = status_val in ("success", "ok", "solved", "resuelto")

    # Si hubo dict, intenta extraer de ahí
    if metrics and isinstance(metrics, dict):
        if path_len is None:
            # distintos posibles nombres
            for k in ("path_len", "moves", "movimientos", "length", "longitud"):
                if k in metrics and isinstance(metrics[k], int):
                    path_len = metrics[k]
                    break
        if solved is None:
            st = (metrics.get('status') or metrics.get('estado') or '')
            if isinstance(st, str):
                s = st.lower()
                solved = any(t in s for t in ("success", "ok", "solved", "resuelto"))
        if expanded is None and isinstance(metrics.get('expansions'), (int, float)):
            expanded = int(metrics['expansions'])
        if elapsed is None and isinstance(metrics.get('time_sec'), (int, float)):
            elapsed = float(metrics['time_sec'])

    # Si aún no se pudo, heurística trivial
    if solved is None and path_len is not None:
        solved = path_len >= 1
    if solved is None:
        low = out.lower()
        if "no se encontró solución" in low or "no solution" in low:
            solved = False
        elif "solución" in low or "solution" in low or "objetivo" in low or "goal" in low:
            solved = True

    return {
        'stdout': out,
        'metrics': metrics,
        'path_len': path_len,
        'solved': solved,
        'expanded': expanded,
        'elapsed_sec': elapsed
    }

def test_optimal(time_limit: int = 20) -> None:
    """
    Verifica que el solver encuentre la solución óptima de 31 movimientos en el tablero clásico.
    
    Args:
        time_limit (int): Límite de tiempo para la prueba de optimalidad.
    """
    print("[*] Prueba de optimalidad (clásico, espera 31 movimientos)...")
    res = _run_solver_cli(time_limit=time_limit, show_path=True)

    if not res['solved']:
        print(res['stdout'])
        sys.exit("[ERROR] El solver no encontró solución en el tablero clásico.")

    if res['path_len'] is None:
        print(res['stdout'])
        sys.exit("[ERROR] No pude inferir la longitud de la solución desde la salida (moves/path_len).")

    if res['path_len'] != 31:
        sys.exit(f"[ERROR] La solución no es óptima: {res['path_len']} movimientos (esperado: 31)")

    print(f"[OK] Óptimo verificado: 31 movimientos. (expanded={res['expanded']}, elapsed={res['elapsed_sec']})")

def run_benchmark(runs: int, time_limit: int) -> None:
    """
    Ejecuta el solver varias veces y guarda los resultados en un archivo CSV.
    
    Args:
        runs (int): Número de ejecuciones del solver.
        time_limit (int): Límite de tiempo para cada ejecución.
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(runs):
        t0 = time.perf_counter()
        res = _run_solver_cli(time_limit=time_limit, show_path=False)
        dt = time.perf_counter() - t0

        rows.append({
            "run": i,
            "hole": "center",
            "solved": int(bool(res['solved'])),
            "time_sec": round(dt, 6) if res['elapsed_sec'] is None else round(float(res['elapsed_sec']), 6),
            "expansions": res['expanded'] if res['expanded'] is not None else "",
            "path_len": res['path_len'] if res['path_len'] is not None else "",
        })
        print(f"  run {i+1}/{runs}: solved={rows[-1]['solved']} time={rows[-1]['time_sec']}s path_len={rows[-1]['path_len']}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] CSV -> {CSV_PATH}")

def make_plots() -> None:
    """
    Genera gráficos PNG a partir de los datos del CSV usando matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib no está instalado. Instálalo con: conda install -y matplotlib")
        return

    if not CSV_PATH.exists():
        print(f"[WARN] No existe {CSV_PATH}. Corre primero el benchmark.")
        return

    # Carga ligera de CSV sin pandas
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row['solved'] = int(row.get('solved', 0))
            except Exception:
                row['solved'] = 0
            try:
                row['time_sec'] = float(row.get('time_sec', 'nan'))
            except Exception:
                row['time_sec'] = float('nan')
            rows.append(row)

    # Histograma de tiempos (soluciones)
    times = [r['time_sec'] for r in rows if r['solved'] == 1 and (r['time_sec'] == r['time_sec'])]
    if times:
        plt.figure()
        plt.hist(times, bins=10)
        plt.title("Peg A*: Tiempo de ejecución (soluciones)")
        plt.xlabel("segundos")
        plt.ylabel("frecuencia")
        plt.tight_layout()
        PLOT_TIME.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_TIME)
        plt.close()
        print(f"[OK] {PLOT_TIME.name}")
    else:
        print("[WARN] No hay soluciones para graficar tiempos.")

    # Tasa de éxito (clásico)
    if rows:
        tot = len(rows)
        ok = sum(1 for r in rows if r['solved'] == 1)
        rate = 100.0 * ok / tot if tot else 0.0

        plt.figure()
        plt.bar(['clásico'], [rate])
        plt.title("Peg A*: Tasa de éxito")
        plt.ylabel("% éxito")
        plt.tight_layout()
        plt.savefig(PLOT_SUCCESS)
        plt.close()
        print(f"[OK] {PLOT_SUCCESS.name}")

def main():
    """
    Función principal que coordina la ejecución de pruebas, benchmarks y generación de gráficos.
    """
    _ensure_runner_exists()
    ap = argparse.ArgumentParser(description="Pruebas + Benchmarks + Gráficas para Peg Solitaire (A*)")
    ap.add_argument("--runs", type=int, default=20, help="Número de corridas para el benchmark")
    ap.add_argument("--time-limit", type=int, default=15, help="Límite de tiempo por corrida (seg)")
    args = ap.parse_args()

    # 1) prueba de optimalidad (31)
    test_optimal(time_limit=args.time_limit)

    # 2) benchmark
    print(f"[*] Benchmark: {args.runs} corridas...")
    run_benchmark(runs=args.runs, time_limit=args.time_limit)

    # 3) gráficas
    print("[*] Generando gráficas...")
    make_plots()
    print("[OK] Todo listo. Revisa la carpeta bench/")

if __name__ == "__main__":
    main()
