# -*- coding: utf-8 -*-
"""
timbiriche_minimax: separación mínima en dos módulos
- game.py: lógica del juego (estado, movimientos, aplicar, puntuar, imprimir)
- ai.py: terminal/evaluación y Minimax con poda alfa-beta

Mantiene nombres comunes para no romper el runner.
"""

from .game import (
    GameState,  # dataclass del estado
    new_game,   # construir estado inicial
    legal_moves, apply_move, completed_boxes,
    current_player, score, is_full, render_board,
    move_to_str, parse_move
)

from .ai import (
    is_terminal, utility, heuristic_eval,
    minimax_decision,  # devuelve mejor movimiento para el jugador actual
)
