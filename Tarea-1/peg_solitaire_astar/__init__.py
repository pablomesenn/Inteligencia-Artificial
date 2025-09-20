# -*- coding: utf-8 -*-
"""
Reexporta la misma API que usa tu runner, separando en módulos:
- board.py: tablero, movimientos, helpers y pretty/format
- search.py: simetrías, heurística y A*
"""

from .board import (
    N, CENTER, CENTER_IDX, IDX_TO_POS, POS_TO_IDX,
    is_valid_cell, pretty_board, format_move,
    bit_clear, bit_set, popcount,
    get_valid_moves_fast as get_valid_moves,
    apply_move_fast as apply_move,
    initial_state, is_goal,
)

from .search import (
    apply_symmetry, canonicalize,
    manhattan_to_center, connected_components,
    heuristic_advanced as heuristic,
    astar_solve,  # firma compatible con tu runner (incluye greedy_bias)
)
