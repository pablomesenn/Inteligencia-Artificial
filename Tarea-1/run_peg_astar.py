#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python run_peg_astar.py --time-limit 180 --show-path --emoji
# python run_peg_astar.py --time-limit 20 --show-path 

from peg_solitaire_astar import (
    astar_solve, initial_state, pretty_board, format_move,
    bit_clear, bit_set, popcount, get_valid_moves, apply_move, heuristic, IDX_TO_POS
)
import argparse

def _is_valid_cell(r, c):
    """
    Check if a cell is valid within the Peg Solitaire board.

    Args:
        r (int): Row index of the cell.
        c (int): Column index of the cell.

    Returns:
        bool: True if the cell is valid, False otherwise.
    """
    if not (0 <= r < 7 and 0 <= c < 7):
        return False
    if (r < 2 or r > 4) and (c < 2 or c > 4):
        return False
    return True

def _mid(a, b):
    """
    Calculate the midpoint between two coordinates.

    Args:
        a (tuple): First coordinate (row, column).
        b (tuple): Second coordinate (row, column).

    Returns:
        tuple: Midpoint coordinate.
    """
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)

def print_move_overlay(src, dst, title=None):
    """
    Print a visual overlay of a move on the Peg Solitaire board.

    Args:
        src (tuple): Source coordinate (row, column).
        dst (tuple): Destination coordinate (row, column).
        title (str, optional): Title to display above the overlay.
    """
    mid = _mid(src, dst)
    grid = []
    for r in range(7):
        row = []
        for c in range(7):
            row.append('·' if _is_valid_cell(r, c) else ' ')
        grid.append(row)
    if _is_valid_cell(*src): grid[src[0]][src[1]] = '🟢'
    if _is_valid_cell(*mid): grid[mid[0]][mid[1]] = '🍽️'
    if _is_valid_cell(*dst): grid[dst[0]][dst[1]] = '🎯'

    if title:
        print(title)
    print("   Leyenda: 🟢 Origen   🍽️ Comida   🎯 Destino")
    for r in range(7):
        print("   " + " ".join(grid[r]))
    print(f"   🟢 {src}  →  🍽️ {mid}  →  🎯 {dst}")

def main():
    """
    Main function to execute the Peg Solitaire A* solver.

    Parses command-line arguments, initializes the game state, and runs the A* search algorithm.
    Displays the results and optionally shows the solution path with overlays.
    """
    parser = argparse.ArgumentParser(description="Runner para A* de Peg Solitaire.")
    parser.add_argument("--time-limit", type=float, default=90.0, help="Tiempo máximo en segundos (default: 90)")
    parser.add_argument("--no-greedy-bias", action="store_true", help="Desactiva el desempate por menos peones.")
    parser.add_argument("--no-sym", action="store_true", help="Desactiva la poda por simetrías.")
    parser.add_argument("--show-path", action="store_true", help="Muestra el camino paso a paso (si hay solución).")
    parser.add_argument("--emoji", action="store_true", help="Muestra overlay con emojis por cada jugada")
    args = parser.parse_args()

    print("Estado inicial: (● = pieza, · = vacío)\n")
    s0 = initial_state()
    print(pretty_board(s0))

    path, metrics = astar_solve(
        time_limit_sec=args.time_limit,
        greedy_bias=not args.no_greedy_bias,
        use_symmetry=not args.no_sym,
    )

    print("\n=== Resultados ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    if path is None:
        print("\nNo se encontró solución en el tiempo dado. Sube el --time-limit o deja activado el pruning por simetrías.")
        return

    print(f"\nSolución encontrada con {len(path)} movimientos.")
    if args.show_path:
        state = s0
        step = 0
        print("\nCamino:")
        for mv in path:
            step += 1
            print(f"\nPaso {step}: {format_move(mv)}")
            src_idx, over_idx, dst_idx = mv

            # Update the state using indices
            state = bit_clear(state, src_idx)
            state = bit_clear(state, over_idx)
            state = bit_set(state, dst_idx)

            # Print the board
            print(pretty_board(state))

            # Convert to coordinates for overlay
            if args.emoji:
                src_pos = IDX_TO_POS[src_idx]
                dst_pos = IDX_TO_POS[dst_idx]
                print_move_overlay(src_pos, dst_pos)

if __name__ == "__main__":
    main()
