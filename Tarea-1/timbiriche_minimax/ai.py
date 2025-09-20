# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Optional
from .game import GameState, legal_moves, apply_move, score, is_full, current_player, completed_boxes

# -------- funciones de evaluación --------
def is_terminal(s: GameState) -> bool:
    """
    Verifica si el estado del juego es terminal, es decir, si el tablero está lleno.

    Args:
        s (GameState): El estado actual del juego.

    Returns:
        bool: True si el estado es terminal, False en caso contrario.
    """
    return is_full(s)

def utility(s: GameState) -> int:
    """
    Calcula la utilidad del estado del juego desde la perspectiva del jugador J0.

    Args:
        s (GameState): El estado actual del juego.

    Returns:
        int: La diferencia de puntuación entre J0 y J1.
    """
    j0, j1 = score(s)
    return j0 - j1  # utilidad desde la perspectiva de J0 (MAX)

def heuristic_eval(s: GameState) -> int:
    """
    Evalúa heurísticamente el estado del juego considerando la diferencia de cajas y las posibilidades de cierre.

    Args:
        s (GameState): El estado actual del juego.

    Returns:
        int: Valor heurístico calculado.
    """
    j0, j1 = score(s)
    base = j0 - j1

    # bonus/malus por “edge-of-box”: aristas que faltan 1 para cerrar
    near0 = 0
    near1 = 0
    # Aproximación barata: contar cajas a un movimiento (mirando completed_boxes
    # de cada posible movimiento del jugador actual)
    for m in legal_moves(s):
        closed = completed_boxes(s, m)
        if closed > 0:
            # Si el turno actual es J0, esas cajas “potenciales” favorecen a J0
            if current_player(s) == 0:
                near0 += closed
            else:
                near1 += closed
    return base + (near0 - near1)

# -------- Minimax con poda alfa-beta --------
def _minimax(s: GameState, depth: int, alpha: int, beta: int, use_heuristic: bool) -> Tuple[int, Optional[Tuple[str,int,int]]]:
    """
    Implementa el algoritmo Minimax con poda alfa-beta para determinar el mejor movimiento.

    Args:
        s (GameState): El estado actual del juego.
        depth (int): La profundidad de búsqueda restante.
        alpha (int): El valor alfa para la poda.
        beta (int): El valor beta para la poda.
        use_heuristic (bool): Si se debe usar evaluación heurística.

    Returns:
        Tuple[int, Optional[Tuple[str,int,int]]]: El valor del estado y el mejor movimiento.
    """
    if is_terminal(s) or depth == 0:
        # Si el estado es terminal o se ha alcanzado la profundidad máxima, evalúa el estado.
        val = utility(s) if is_terminal(s) else heuristic_eval(s) if use_heuristic else 0
        return val, None

    moves = legal_moves(s)
    if not moves:
        # Si no hay movimientos legales, evalúa el estado.
        val = utility(s) if is_terminal(s) else heuristic_eval(s) if use_heuristic else 0
        return val, None

    player = current_player(s)
    best_move = None

    if player == 0:  # MAX
        value = -10**9
        for m in moves:
            s2 = apply_move(s, m)
            # Si cerró caja, NO reduce profundidad (jugador repite)
            next_depth = depth if current_player(s2) == player else depth - 1
            child_val, _ = _minimax(s2, next_depth, alpha, beta, use_heuristic)
            if child_val > value:
                value = child_val
                best_move = m
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value, best_move
    else:  # MIN
        value = +10**9
        for m in moves:
            s2 = apply_move(s, m)
            next_depth = depth if current_player(s2) == player else depth - 1
            child_val, _ = _minimax(s2, next_depth, alpha, beta, use_heuristic)
            if child_val < value:
                value = child_val
                best_move = m
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_move

def minimax_decision(s: GameState, depth: int = 5, use_heuristic: bool = True) -> Tuple[str,int,int]:
    """
    Devuelve el mejor movimiento para el jugador actual utilizando el algoritmo Minimax.

    Args:
        s (GameState): El estado actual del juego.
        depth (int): Profundidad de búsqueda (efectiva, pues si cierras caja no decrece).
        use_heuristic (bool): Si se debe usar evaluación heurística.

    Returns:
        Tuple[str,int,int]: El mejor movimiento encontrado.
    """
    _, best = _minimax(s, depth, -10**9, 10**9, use_heuristic)
    # Si no hay mejor movimiento (estado terminal), devuelve un movimiento neutro.
    return best if best is not None else ("H", 0, 0)
