# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict

# Define un movimiento como una tupla que consiste en un tipo ("H" o "V"), fila y columna.
Move = Tuple[str, int, int]   
# Define un jugador como un entero, ya sea 0 o 1.
Player = int                  

@dataclass(frozen=True)
class GameState:
    """
    Representa el estado del juego en cualquier momento dado.
    
    Atributos:
        n (int): El tamaño del tablero (n×n puntos) que resulta en (n-1)×(n-1) cajas.
        h_edges (frozenset): El conjunto de aristas horizontales ocupadas representadas como ("H", fila, col).
        v_edges (frozenset): El conjunto de aristas verticales ocupadas representadas como ("V", fila, col).
        turn (Player): El jugador actual, ya sea 0 (MAX) o 1 (MIN).
        scores (Tuple[int, int]): Las puntuaciones de los jugadores, representadas como (puntos J0, puntos J1).
    """
    n: int
    h_edges: frozenset
    v_edges: frozenset
    turn: Player
    scores: Tuple[int, int]

# ----- Ayudantes de Rango -----
def _valid_h(n: int, r: int, c: int) -> bool:
    """
    Verifica si un movimiento horizontal está dentro del rango válido.
    
    Args:
        n (int): El tamaño del tablero.
        r (int): El índice de la fila.
        c (int): El índice de la columna.
    
    Returns:
        bool: True si el movimiento es válido, False en caso contrario.
    """
    return 0 <= r < n and 0 <= c < n - 1

def _valid_v(n: int, r: int, c: int) -> bool:
    """
    Verifica si un movimiento vertical está dentro del rango válido.
    
    Args:
        n (int): El tamaño del tablero.
        r (int): El índice de la fila.
        c (int): El índice de la columna.
    
    Returns:
        bool: True si el movimiento es válido, False en caso contrario.
    """
    return 0 <= r < n - 1 and 0 <= c < n

# ----- Construcción del Juego -----
def new_game(n: int = 4) -> GameState:
    """
    Inicializa un nuevo estado de juego con un tablero vacío.
    
    Args:
        n (int): El tamaño del tablero. El valor predeterminado es 4.
    
    Returns:
        GameState: El estado inicial del juego.
    """
    return GameState(
        n=n,
        h_edges=frozenset(),
        v_edges=frozenset(),
        turn=0,                # El jugador MAX comienza
        scores=(0, 0),
    )

# ----- Lista de Movimientos Legales -----
def legal_moves(s: GameState) -> List[Move]:
    """
    Genera una lista de todos los movimientos legales para el estado actual del juego.
    
    Args:
        s (GameState): El estado actual del juego.
    
    Returns:
        List[Move]: Una lista de todos los movimientos legales posibles.
    """
    moves: List[Move] = []
    # Verificar movimientos horizontales
    for r in range(s.n):
        for c in range(s.n - 1):
            m = ("H", r, c)
            if m not in s.h_edges:
                moves.append(m)
    # Verificar movimientos verticales
    for r in range(s.n - 1):
        for c in range(s.n):
            m = ("V", r, c)
            if m not in s.v_edges:
                moves.append(m)
    return moves

# ----- Detección de Cajas Completadas por una Arista -----
def _boxes_closed_by(n: int, move: Move, h_edges: Set[Move], v_edges: Set[Move]) -> int:
    """
    Determina el número de cajas completadas por un movimiento dado.
    
    Args:
        n (int): El tamaño del tablero.
        move (Move): El movimiento que se está evaluando.
        h_edges (Set[Move]): El conjunto de aristas horizontales.
        v_edges (Set[Move]): El conjunto de aristas verticales.
    
    Returns:
        int: El número de cajas completadas por el movimiento.
    """
    t, r, c = move
    closed = 0

    def box_complete(i: int, j: int) -> bool:
        """
        Verifica si una caja con la esquina superior izquierda en (i, j) está completa.
        
        Args:
            i (int): El índice de la fila de la esquina superior izquierda.
            j (int): El índice de la columna de la esquina superior izquierda.
        
        Returns:
            bool: True si la caja está completa, False en caso contrario.
        """
        return (
            ("H", i, j) in h_edges and
            ("H", i + 1, j) in h_edges and
            ("V", i, j) in v_edges and
            ("V", i, j + 1) in v_edges
        )

    if t == "H":
        # Verificar caja arriba
        if r > 0 and box_complete(r - 1, c):
            closed += 1
        # Verificar caja abajo
        if r < n - 1 and box_complete(r, c):
            closed += 1
    else:  # "V"
        # Verificar caja a la izquierda
        if c > 0 and box_complete(r, c - 1):
            closed += 1
        # Verificar caja a la derecha
        if c < n - 1 and box_complete(r, c):
            closed += 1
    return closed

def completed_boxes(s: GameState, move: Move) -> int:
    """
    Calcula el número de cajas completadas al aplicar un movimiento.
    
    Args:
        s (GameState): El estado actual del juego.
        move (Move): El movimiento a aplicar.
    
    Returns:
        int: El número de cajas completadas por el movimiento.
    """
    # Usar conjuntos mutables temporales para verificar el cierre con el movimiento aplicado
    h = set(s.h_edges)
    v = set(s.v_edges)
    t, r, c = move
    if t == "H":
        h.add(move)
    else:
        v.add(move)
    return _boxes_closed_by(s.n, move, h, v)

# ----- Aplicar Movimiento -----
def apply_move(s: GameState, move: Move) -> GameState:
    """
    Aplica un movimiento al estado actual del juego y devuelve el nuevo estado.
    
    Args:
        s (GameState): El estado actual del juego.
        move (Move): El movimiento a aplicar.
    
    Returns:
        GameState: El nuevo estado del juego después del movimiento.
    """
    t, r, c = move
    # Si el movimiento ya está ocupado, devuelve el mismo estado (o lanza un error)
    if (t == "H" and move in s.h_edges) or (t == "V" and move in s.v_edges):
        return s

    h = set(s.h_edges)
    v = set(s.v_edges)
    if t == "H":
        if not _valid_h(s.n, r, c):  # Verificación de seguridad
            return s
        h.add(move)
    else:
        if not _valid_v(s.n, r, c):
            return s
        v.add(move)

    # Determinar cuántas cajas se cerraron
    closed = _boxes_closed_by(s.n, move, h, v)

    j0, j1 = s.scores
    if closed > 0:
        # Puntos para el jugador que jugó; mantiene el turno
        if s.turn == 0:
            j0 += closed
            next_turn = 0
        else:
            j1 += closed
            next_turn = 1
    else:
        # Cambia el turno
        next_turn = 1 - s.turn

    return GameState(
        n=s.n,
        h_edges=frozenset(h),
        v_edges=frozenset(v),
        turn=next_turn,
        scores=(j0, j1),
    )

# ----- Estado -----
def current_player(s: GameState) -> Player:
    """
    Devuelve el jugador actual.
    
    Args:
        s (GameState): El estado actual del juego.
    
    Returns:
        Player: El jugador actual.
    """
    return s.turn

def score(s: GameState) -> Tuple[int, int]:
    """
    Devuelve las puntuaciones actuales de los jugadores.
    
    Args:
        s (GameState): El estado actual del juego.
    
    Returns:
        Tuple[int, int]: Las puntuaciones de los jugadores.
    """
    return s.scores

def is_full(s: GameState) -> bool:
    """
    Verifica si el tablero está lleno.
    
    Args:
        s (GameState): El estado actual del juego.
    
    Returns:
        bool: True si el tablero está lleno, False en caso contrario.
    """
    total_h = s.n * (s.n - 1)
    total_v = (s.n - 1) * s.n
    return len(s.h_edges) + len(s.v_edges) == total_h + total_v

# ----- Representación -----
def render_board(s: GameState) -> str:
    """
    Representa el estado actual del tablero como una cadena de texto.
    
    Args:
        s (GameState): El estado actual del juego.
    
    Returns:
        str: Una representación en cadena del tablero.
    """
    # Representación simple:
    # Puntos: "•"
    # Arista horizontal: "──"
    # Arista vertical: "|"
    n = s.n
    lines: List[str] = []
    for r in range(n - 1):
        # Fila de puntos + aristas horizontales
        row = []
        for c in range(n - 1):
            row.append("•")
            row.append("──" if ("H", r, c) in s.h_edges else "  ")
        row.append("•")
        lines.append("".join(row))
        # Fila de aristas verticales + espacio/caja
        row = []
        for c in range(n - 1):
            row.append("|" if ("V", r, c) in s.v_edges else " ")
            # Interior de la caja: marcador de propietario simple
            up = ("H", r, c) in s.h_edges
            down = ("H", r + 1, c) in s.h_edges
            left = ("V", r, c) in s.v_edges
            right = ("V", r, c + 1) in s.v_edges
            if up and down and left and right:
                # Caja cerrada, no almacenamos dueño aquí (para simplificar vista)
                row.append("[]")
            else:
                row.append("  ")
        row.append("|" if ("V", r, n - 1) in s.v_edges else " ")
        lines.append("".join(row))
    # Última fila de puntos + aristas horizontales
    row = []
    for c in range(n - 1):
        row.append("•")
        row.append("──" if ("H", n - 1, c) in s.h_edges else "  ")
    row.append("•")
    lines.append("".join(row))
    # Marcador de turno y puntaje
    j0, j1 = s.scores
    lines.append(f"\nTurno: J{s.turn} | Marcador J0={j0}  J1={j1}")
    return "\n".join(lines)

# Utilidades de texto (para logs/runner)
def move_to_str(m: Move) -> str:
    """
    Convierte un movimiento a una representación en cadena.
    
    Args:
        m (Move): El movimiento a convertir.
    
    Returns:
        str: La representación en cadena del movimiento.
    """
    t, r, c = m
    return f"{t} {r} {c}"

def parse_move(s: str) -> Move:
    """
    Analiza una cadena para crear un movimiento.
    
    Args:
        s (str): La representación en cadena del movimiento.
    
    Returns:
        Move: El movimiento representado por la cadena.
    """
    # Ejemplo: "H 2 1" o "V 3 0"
    t, r, c = s.strip().split()
    return (t.upper(), int(r), int(c))
