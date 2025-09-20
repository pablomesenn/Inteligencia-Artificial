# -*- coding: utf-8 -*-
from __future__ import annotations
import heapq, time, collections
from typing import Dict, List, Tuple, Optional

from .board import (
    N, CENTER, IDX_TO_POS, POS_TO_IDX,
    is_valid_cell, initial_state, is_goal,
    get_valid_moves_fast, apply_move_fast, popcount,
)

# =========================
# Simetrías D4
# =========================
def apply_symmetry(state: int, sym: int) -> int:
    """
    Aplica una simetría D4 al estado del tablero.
    
    Args:
        state (int): El estado actual del tablero.
        sym (int): El índice de la simetría a aplicar (0 a 7).

    Returns:
        int: El nuevo estado del tablero después de aplicar la simetría.
    """
    # Primero, se crea una lista `pieces` que almacenará las posiciones de las piezas en el tablero.
    # Se recorre cada índice en `IDX_TO_POS` y se verifica si hay una pieza en esa posición
    # usando una operación bit a bit. Si hay una pieza, se añade su posición a `pieces`.
    pieces = []
    for i in range(len(IDX_TO_POS)):
        if state & (1 << i):
            pieces.append(IDX_TO_POS[i])

    # Luego, se crea una nueva lista `new_pieces` para almacenar las posiciones de las piezas
    # después de aplicar la simetría especificada por `sym`.
    new_pieces = []
    for r, c in pieces:
        # Dependiendo del valor de `sym`, se calcula la nueva posición `(nr, nc)`
        # aplicando una de las ocho posibles simetrías del grupo D4.
        if sym == 0:    nr, nc = r, c
        elif sym == 1:  nr, nc = c, N - 1 - r
        elif sym == 2:  nr, nc = N - 1 - r, N - 1 - c
        elif sym == 3:  nr, nc = N - 1 - c, r
        elif sym == 4:  nr, nc = r, N - 1 - c
        elif sym == 5:  nr, nc = N - 1 - r, c
        elif sym == 6:  nr, nc = c, r
        else:           nr, nc = N - 1 - c, N - 1 - r
        # Solo se añaden las nuevas posiciones que son válidas en el tablero.
        if is_valid_cell(nr, nc):
            new_pieces.append((nr, nc))

    # Finalmente, se construye un nuevo estado `new_state` a partir de las posiciones
    # transformadas en `new_pieces`. Se utiliza una operación bit a bit para establecer
    # los bits correspondientes a las posiciones de las piezas en el nuevo estado.
    new_state = 0
    for nr, nc in new_pieces:
        if (nr, nc) in POS_TO_IDX:
            new_state |= (1 << POS_TO_IDX[(nr, nc)])
    return new_state

def canonicalize(state: int) -> int:
    """
    Encuentra la representación canónica del estado del tablero aplicando simetrías.

    Args:
        state (int): El estado actual del tablero.

    Returns:
        int: El estado canónico más pequeño.
    """
    canonical = state
    for sym in range(8):
        candidate = apply_symmetry(state, sym)
        if candidate < canonical:
            canonical = candidate
    return canonical

# =========================
# Heurística avanzada
# =========================
def manhattan_to_center(idx: int) -> int:
    """
    Calcula la distancia de Manhattan desde una posición al centro del tablero.

    Args:
        idx (int): Índice de la posición en el tablero.

    Returns:
        int: Distancia de Manhattan al centro.
    """
    r, c = IDX_TO_POS[idx]
    return abs(r - CENTER[0]) + abs(c - CENTER[1])

def connected_components(state: int) -> int:
    """
    Calcula el número de componentes conectados en el estado del tablero.

    Args:
        state (int): El estado actual del tablero.

    Returns:
        int: Número de componentes conectados.
    """
    # Inicializa una lista para rastrear las posiciones visitadas en el tablero.
    visited = [False] * len(IDX_TO_POS)
    # Inicializa el contador de componentes conectados.
    components = 0

    # Construye una lista de adyacencias ortogonales para cada celda válida en el tablero.
    # ADJ almacena listas de índices de celdas adyacentes para cada celda válida.
    ADJ: List[List[int]] = [[] for _ in range(len(IDX_TO_POS))]
    for i, (r, c) in enumerate(IDX_TO_POS):
        # Considera las cuatro direcciones cardinales para encontrar celdas adyacentes.
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            # Si la celda adyacente es válida, añade su índice a la lista de adyacencias.
            if is_valid_cell(nr, nc):
                ADJ[i].append(POS_TO_IDX[(nr, nc)])

    # Recorre cada celda válida en el tablero.
    for i in range(len(IDX_TO_POS)):
        # Si la celda contiene una pieza y no ha sido visitada, inicia una búsqueda de componentes.
        if (state & (1 << i)) and not visited[i]:
            # Incrementa el contador de componentes conectados.
            components += 1
            # Utiliza una cola para realizar una búsqueda en anchura (BFS) desde la celda actual.
            queue = collections.deque([i])
            visited[i] = True
            while queue:
                current = queue.popleft()
                # Explora las celdas adyacentes no visitadas que contienen piezas.
                for neighbor in ADJ[current]:
                    if (state & (1 << neighbor)) and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
    # Devuelve el número total de componentes conectados encontrados.
    return components

def heuristic_advanced(state: int) -> int:
    """
    Calcula una heurística avanzada para el estado del tablero.

    Args:
        state (int): El estado actual del tablero.

    Returns:
        int: Valor heurístico calculado.
    """
    num_pieces = popcount(state)
    if num_pieces <= 1:
        return 0 if is_goal(state) else float('inf')

    # H1: cada movimiento elimina 1 pieza
    h1 = num_pieces - 1

    # H2: mínima distancia Manhattan al centro, aprox saltos mínimos
    min_dist = float('inf')
    for i in range(len(IDX_TO_POS)):
        if state & (1 << i):
            min_dist = min(min_dist, manhattan_to_center(i))
    h2 = (min_dist + 1) // 2 if min_dist != float('inf') else 0

    # H3: componentes - 1
    comps = connected_components(state)
    h3 = max(0, comps - 1)

    # H4: penalización por piezas (casi) aisladas
    isolated_penalty = 0
    # Adyacencia local (otra vez, ligera y local aquí)
    for i, (r, c) in enumerate(IDX_TO_POS):
        if state & (1 << i):
            neighbors = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if is_valid_cell(nr, nc) and state & (1 << POS_TO_IDX[(nr, nc)]):
                    neighbors += 1
            if neighbors <= 1:
                isolated_penalty += 1
    h4 = isolated_penalty // 3

    return max(h1, h2, h3, h4)

# =========================
# Búsqueda A*
# =========================
def astar_solve(
    time_limit_sec: float = 300.0,
    greedy_bias: bool = True,      # se mantiene en la firma por compatibilidad con tu runner
    use_symmetry: bool = True,
    max_nodes: int = 2_000_000
) -> Tuple[Optional[List[Tuple[int, int, int]]], Dict]:
    """
    Realiza una búsqueda A* optimizada para resolver el problema del solitario.

    Args:
        time_limit_sec (float): Límite de tiempo en segundos para la búsqueda.
        greedy_bias (bool): Sesgo para preferir menos piezas y menor heurística.
        use_symmetry (bool): Si se debe usar simetría para optimizar la búsqueda.
        max_nodes (int): Máximo número de nodos a expandir.

    Returns:
        Tuple[Optional[List[Tuple[int, int, int]]], Dict]: La secuencia de movimientos y estadísticas de la búsqueda.
    """
    # Inicia el temporizador para medir el tiempo de ejecución del algoritmo.
    start_time = time.time()
    # Obtiene el estado inicial del tablero.
    start_state = initial_state()
    # Canonicaliza el estado inicial si se utiliza simetría, de lo contrario, usa el estado inicial tal cual.
    start_canonical = canonicalize(start_state) if use_symmetry else start_state

    # Cola de prioridad que almacena tuplas con la forma: (f, tiebreak, g, canonical_state, original_state)
    # f: costo total estimado (g + h), tiebreak: valor para desempate, g: costo desde el inicio,
    # canonical_state: estado canónico, original_state: estado original.
    open_set = [(heuristic_advanced(start_state), 0, 0, start_canonical, start_state)]
    heapq.heapify(open_set)

    # Diccionario que almacena el costo más bajo conocido para llegar a cada estado canónico.
    g_score: Dict[int, int] = {start_canonical: 0}
    # Diccionario que almacena de dónde vino cada estado y el movimiento que se usó para llegar allí.
    came_from: Dict[int, Tuple[int, Optional[Tuple[int, int, int]]]] = {}

    # Contadores para nodos expandidos y generados, y un valor para desempate.
    nodes_expanded = 0
    nodes_generated = 1
    tie_breaker = 0

    # Almacena el menor número de piezas encontradas en el camino.
    best_num_pieces = popcount(start_state)

    # Bucle principal de la búsqueda A*.
    while open_set:
        # Verifica si se ha excedido el límite de tiempo.
        now = time.time()
        if now - start_time > time_limit_sec:
            return None, {
                "status": "timeout",
                "expanded": nodes_expanded,
                "generated": nodes_generated,
                "elapsed_sec": now - start_time,
                "best_pieces": best_num_pieces
            }
        # Verifica si se ha excedido el número máximo de nodos.
        if nodes_expanded > max_nodes:
            return None, {
                "status": "max_nodes_exceeded",
                "expanded": nodes_expanded,
                "generated": nodes_generated,
                "elapsed_sec": now - start_time,
                "best_pieces": best_num_pieces
            }

        # Extrae el nodo con el menor costo estimado de la cola de prioridad.
        current_f, _, current_g, canonical_state, original_state = heapq.heappop(open_set)

        # Si el costo actual es mayor que el mejor conocido, ignora este nodo.
        if current_g > g_score.get(canonical_state, float('inf')):
            continue

        # Incrementa el contador de nodos expandidos.
        nodes_expanded += 1

        # Si se ha alcanzado el estado objetivo, reconstruye el camino y devuelve el resultado.
        if is_goal(original_state):
            path = []
            cur = original_state
            while cur in came_from:
                cur, mv = came_from[cur]
                if mv is not None:
                    path.append(mv)
            path.reverse()
            return path, {
                "status": "success",
                "expanded": nodes_expanded,
                "generated": nodes_generated,
                "elapsed_sec": now - start_time,
                "moves": len(path),
                "final_pieces": popcount(original_state),
            }

        # Actualiza el mejor número de piezas si se encuentra un estado con menos piezas.
        cur_pieces = popcount(original_state)
        if cur_pieces < best_num_pieces:
            best_num_pieces = cur_pieces
            # (si quieres, imprime progreso aquí)

        # Genera sucesores del estado actual.
        for move in get_valid_moves_fast(original_state):
            neighbor = apply_move_fast(original_state, move)
            neighbor_canonical = canonicalize(neighbor) if use_symmetry else neighbor
            tentative_g = current_g + 1

            # Si se encuentra un mejor camino hacia el vecino, actualiza la información.
            if tentative_g < g_score.get(neighbor_canonical, float('inf')):
                came_from[neighbor] = (original_state, move)
                g_score[neighbor_canonical] = tentative_g

                # Calcula la heurística y el costo total estimado.
                h = heuristic_advanced(neighbor)
                f = tentative_g + h

                # Calcula el valor de desempate para preferir menos piezas y menor heurística.
                tie_breaker += 1
                tie_value = (popcount(neighbor) << 16) + h + (tie_breaker & 0xFFFF)

                # Añade el vecino a la cola de prioridad.
                heapq.heappush(open_set, (f, tie_value, tentative_g, neighbor_canonical, neighbor))
                nodes_generated += 1

        # Opción para realizar logging periódico cada 50,000 nodos expandidos.
        if nodes_expanded % 50_000 == 0:
            pass  # opcional: logging periódico

    # Si se agotan los nodos sin encontrar una solución, devuelve el estado de agotamiento.
    return None, {
        "status": "exhausted",
        "expanded": nodes_expanded,
        "generated": nodes_generated,
        "elapsed_sec": time.time() - start_time,
        "best_pieces": best_num_pieces
    }
