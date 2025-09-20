#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python run_dot_boxes.py --ai-vs-ai --depth 5 --no-heuristic --delay 0.5 --style clean
# python run_dot_boxes.py -n 4 --ai-vs-ai --style clean --delay 0.2
# python run_dot_boxes.py -n 4 --ai-plays-0 --depth 6
# python run_dot_boxes.py -n 4 --ai-vs-ai --delay 0.75
# python run_dot_boxes.py -n 4 --ai-vs-ai --style clean --depth 3 --delay 0.2



import argparse
import sys
import time

from timbiriche_minimax import (
    new_game, render_board, legal_moves, apply_move, current_player, score,
    move_to_str, parse_move, is_terminal, minimax_decision
)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Timbiriche game.

    Returns:
        argparse.Namespace: Parsed arguments including board size, depth, AI options, heuristic usage, delay, and style.
    """
    p = argparse.ArgumentParser(description="Timbiriche (Dots & Boxes) - Minimax")
    p.add_argument("-n", type=int, default=4, help="Tamaño del tablero (n x n puntos). Default: 4")
    p.add_argument("--depth", type=int, default=5, help="Profundidad de búsqueda. Default: 5")
    p.add_argument("--ai-vs-ai", action="store_true", help="Juega IA vs IA")
    p.add_argument("--ai-plays-0", action="store_true", help="La IA juega como J0")
    p.add_argument("--ai-plays-1", action="store_true", help="La IA juega como J1")
    p.add_argument("--no-heuristic", action="store_true", help="Desactiva heurística en hojas no terminales")
    p.add_argument("--delay", type=float, default=0.0, help="Pausa (s) entre jugadas para visualizar")
    p.add_argument("--style", default="clean", help="Etiqueta estética para impresión (no funcional). Default: clean")
    return p.parse_args()

def banner(n: int, style: str) -> None:
    """
    Print the game banner with board size and style.

    Args:
        n (int): Board size.
        style (str): Style tag for display.
    """
    print(f"=== TIMBIRICHE N={n} ===  (estilo: {style})\n")

def print_turn(state) -> None:
    """
    Display the current turn and scores of the players.

    Args:
        state: Current game state.
    """
    j = current_player(state)
    j0, j1 = score(state)
    print(f"🟦 J0={j0}  🟥 J1={j1}   |  Turno: J{j}")

def ask_human_move(state) -> tuple[str, int, int]:
    """
    Prompt the human player for a move and validate it.

    Args:
        state: Current game state.

    Returns:
        tuple[str, int, int]: The validated move from the human player.
    """
    print("Jugadas legales (ejemplo formato: 'H 2 1' ó 'V 0 3'):")
    ms = legal_moves(state)
    preview = "  ".join(move_to_str(m) for m in ms[:12])
    print(f"  {preview}{' ...' if len(ms) > 12 else ''}")
    while True:
        try:
            raw = input("Tu jugada> ").strip()
            m = parse_move(raw)
            if m in ms:
                return m
            print("Movimiento inválido. Intenta de nuevo.")
        except Exception:
            print("Formato inválido. Usa: H r c  ó  V r c")

def main() -> None:
    """
    Main function to run the Timbiriche game, handling game setup, turns, and endgame.
    """
    args = parse_args()
    state = new_game(args.n)
    ai0 = args.ai_vs_ai or args.ai_plays_0
    ai1 = args.ai_vs_ai or args.ai_plays_1
    use_heuristic = not args.no_heuristic

    banner(args.n, args.style)
    print_turn(state)
    print(render_board(state))
    print()

    move_no = 0
    while not is_terminal(state):
        j = current_player(state)
        is_ai = (j == 0 and ai0) or (j == 1 and ai1)

        if is_ai:
            # AI decision-making process
            t0 = time.time()
            mv = minimax_decision(state, depth=args.depth, use_heuristic=use_heuristic)
            dt = time.time() - t0
            move_no += 1
            print(f"\n IA jugó {move_to_str(mv)} en {dt:.3f}s.")
        else:
            # Human player's turn
            print("\n Tu turno.")
            mv = ask_human_move(state)
            move_no += 1
            print(f"Jugada #{move_no}: {move_to_str(mv)}")

        # Apply move and display the board
        state = apply_move(state, mv)
        print()
        print_turn(state)
        print()
        print(render_board(state))
        if args.delay > 0:
            time.sleep(args.delay)

    # End of game
    j0, j1 = score(state)
    print("\n=== PARTIDA FINALIZADA ===")
    print(f"Marcador final: 🟦 J0={j0}  🟥 J1={j1}")
    if j0 > j1:
        print("Ganó J0 🎉")
    elif j1 > j0:
        print("Ganó J1 🎉")
    else:
        print("Empate 🤝")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        sys.exit(130)
