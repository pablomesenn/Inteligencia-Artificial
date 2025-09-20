# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple

# =========================
# Configuración del tablero
# =========================
N = 7
CENTER = (3, 3)

def is_valid_cell(r: int, c: int) -> bool:
    """
    Verifica si una celda en la posición (r, c) es válida dentro del tablero.
    """
    if not (0 <= r < N and 0 <= c < N):
        return False
    if r < 2 and (c < 2 or c > 4):
        return False
    if r > 4 and (c < 2 or c > 4):
        return False
    return True

# Mapeo de posiciones
# Este bloque de código crea dos estructuras de datos para mapear las posiciones válidas del tablero.
# IDX_TO_POS es una lista que almacena las coordenadas (r, c) de cada celda válida en el tablero.
# POS_TO_IDX es un diccionario que mapea cada coordenada válida (r, c) a un índice único, que es su posición en IDX_TO_POS.
IDX_TO_POS: List[Tuple[int, int]] = []
POS_TO_IDX: Dict[Tuple[int, int], int] = {}
for r in range(N):
    for c in range(N):
        if is_valid_cell(r, c):
            # Si la celda es válida, se añade a POS_TO_IDX con su índice actual en IDX_TO_POS.
            POS_TO_IDX[(r, c)] = len(IDX_TO_POS)
            # Se añade la posición (r, c) a IDX_TO_POS.
            IDX_TO_POS.append((r, c))

# Se obtiene el índice del centro del tablero usando el diccionario POS_TO_IDX.
CENTER_IDX = POS_TO_IDX[CENTER]

# =========================
# Adyacencias y movimientos
# =========================
# ADJ es una lista de listas que almacena las adyacencias de cada celda válida.
# Cada celda tiene una lista de índices que representan sus celdas adyacentes.
ADJ: List[List[int]] = [[] for _ in range(len(IDX_TO_POS))]
for i, (r, c) in enumerate(IDX_TO_POS):
    # Se consideran las cuatro direcciones cardinales para encontrar celdas adyacentes.
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if is_valid_cell(nr, nc):
            # Si la celda adyacente es válida, se añade su índice a la lista de adyacencias de la celda actual.
            j = POS_TO_IDX[(nr, nc)]
            ADJ[i].append(j)

# VALID_MOVES es una lista que almacena todos los movimientos válidos posibles en el tablero.
# Cada movimiento se representa como una tupla (src_idx, over_idx, dst_idx).
VALID_MOVES: List[Tuple[int, int, int]] = []
for src_idx, (src_r, src_c) in enumerate(IDX_TO_POS):
    # Se consideran las cuatro direcciones cardinales para encontrar movimientos válidos.
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        over_r, over_c = src_r + dr, src_c + dc
        dst_r, dst_c = src_r + 2 * dr, src_c + 2 * dc
        # Un movimiento es válido si la celda sobre la que se salta y la celda de destino son válidas.
        if is_valid_cell(over_r, over_c) and is_valid_cell(dst_r, dst_c):
            over_idx = POS_TO_IDX[(over_r, over_c)]
            dst_idx = POS_TO_IDX[(dst_r, dst_c)]
            # Se añade el movimiento a la lista de movimientos válidos.
            VALID_MOVES.append((src_idx, over_idx, dst_idx))

# =========================
# Estado y operaciones
# =========================
def initial_state() -> int:
    """
    Devuelve el estado inicial del tablero con todas las posiciones ocupadas excepto el centro.
    """
    state = (1 << len(IDX_TO_POS)) - 1
    state &= ~(1 << CENTER_IDX)
    return state

def is_goal(state: int) -> bool:
    """
    Verifica si el estado actual es el estado objetivo, es decir, solo queda una ficha en el centro.
    """
    return state == (1 << CENTER_IDX)

def popcount(state: int) -> int:
    """
    Cuenta el número de bits establecidos en el estado, es decir, el número de fichas en el tablero.
    """
    return state.bit_count()

def get_valid_moves_fast(state: int) -> List[Tuple[int, int, int]]:
    """
    Devuelve una lista de movimientos válidos para el estado dado.
    """
    moves = []
    for src_idx, over_idx, dst_idx in VALID_MOVES:
        if (state & (1 << src_idx)) and (state & (1 << over_idx)) and not (state & (1 << dst_idx)):
            moves.append((src_idx, over_idx, dst_idx))
    return moves

def apply_move_fast(state: int, move: Tuple[int, int, int]) -> int:
    """
    Aplica un movimiento al estado actual y devuelve el nuevo estado.
    """
    src_idx, over_idx, dst_idx = move
    return (state & ~(1 << src_idx) & ~(1 << over_idx)) | (1 << dst_idx)

# =========================
# Helpers y visualización
# =========================
def bit_clear(mask: int, bit: int) -> int:
    """
    Limpia (pone a 0) el bit especificado en la máscara.
    """
    return mask & ~(1 << bit)

def bit_set(mask: int, bit: int) -> int:
    """
    Establece (pone a 1) el bit especificado en la máscara.
    """
    return mask | (1 << bit)

def format_move(move: Tuple[int, int, int]) -> str:
    """
    Formatea un movimiento en una cadena legible que muestra la posición de origen, destino y la posición saltada.
    """
    src_idx, over_idx, dst_idx = move
    src_pos = IDX_TO_POS[src_idx]
    over_pos = IDX_TO_POS[over_idx]
    dst_pos = IDX_TO_POS[dst_idx]
    return f"{src_pos} -> {dst_pos} (salta {over_pos})"

def pretty_board(state: int) -> str:
    """
    Devuelve una representación visual del tablero en el estado actual.
    """
    lines = []
    for r in range(N):
        line = []
        for c in range(N):
            if not is_valid_cell(r, c):
                line.append("  ")
            else:
                idx = POS_TO_IDX[(r, c)]
                line.append("● " if state & (1 << idx) else "· ")
        lines.append("".join(line).rstrip())
    return "\n".join(lines)
