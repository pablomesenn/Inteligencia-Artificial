# Tarea 2 - Algoritmo Genético para CartPole-v1

Este proyecto implementa un algoritmo genético para resolver el problema de control CartPole-v1 de la librería de entornos Gymnasium. El algoritmo evoluciona una población de redes neuronales o cromosomas simples para mantener el equilibrio de un péndulo invertido sobre un carro.

## Dependencias Necesarias

El proyecto requiere las siguientes librerías de Python:

- `gymnasium` - Entorno de simulación CartPole
- `numpy` - Operaciones numéricas y manejo de arrays
- `matplotlib` - Visualización de gráficos de progreso
- `random` - Generación de números aleatorios (incluida en Python estándar)

## Instalación

### 1. Crear un entorno virtual

```bash
# Crear el entorno virtual
python -m venv cartpole_genetic_env

# Activar el entorno virtual
# En Windows:
cartpole_genetic_env\Scripts\activate
# En Linux/Mac:
source cartpole_genetic_env/bin/activate
```

### 2. Instalar las dependencias

```bash
pip install gymnasium numpy matplotlib
```

**Nota:** Si se tienen problemas con la visualización del entorno, se debe de instalar:
```bash
pip install gymnasium[classic_control]
```

### 3. Verificar la instalación

Se puede verificar que las dependencias estén correctamente instaladas ejecutando:
```bash
python -c "import gymnasium, numpy, matplotlib; print('Todas las dependencias instaladas correctamente')"
```

## Ejecución del Proyecto

Para ejecutar el algoritmo genético:

```bash
python main.py
```

El programa mostrará:
1. Una ventana con la simulación visual del CartPole durante el entrenamiento
2. Información de debug en la consola mostrando observaciones y recompensas
3. Al finalizar, un gráfico con la progresión del fitness a lo largo de las generaciones

## Configuraciones Disponibles

### Parámetros Principales

Puedes modificar los siguientes parámetros al inicio del código:

```python
POP_SIZE = 30    # Tamaño de la población
GENS = 50        # Número de generaciones
MUT_RATE = 0.1   # Tasa de mutación (0.1 = 10%)
```

#### Consideraciones Importantes:

- **Tamaño de Población Mínimo:** El tamaño de población debe ser de al menos 10 individuos, ya que el algoritmo:
  - Selecciona el 20% superior de la población como supervivientes
  - Necesita al menos 2 supervivientes para realizar crossover
  - Con poblaciones muy pequeñas (< 10) puede generar errores al intentar hacer `random.sample(survivors, 2)`

- **Generaciones:** Más generaciones generalmente mejoran el rendimiento, pero aumentan el tiempo de entrenamiento

- **Tasa de Mutación:** 
  - Valores bajos (0.01-0.05): Convergencia más lenta pero estable
  - Valores medios (0.1-0.2): Balance entre exploración y explotación
  - Valores altos (0.3+): Mayor diversidad pero posible inestabilidad

### Estrategias de Crossover

El proyecto incluye tres estrategias de crossover que puedes intercambiar:

```python
# En la línea del crossover, puedes cambiar entre:
child = uniform_crossover(p1, p2)      # Crossover uniforme (por defecto)
child = single_point_crossover(p1, p2)  # Crossover de un punto
child = two_point_crossover(p1, p2)     # Crossover de dos puntos
```

#### Descripción de las Estrategias:

1. **Single Point Crossover:** Corta en un punto aleatorio y combina las partes
2. **Two Point Crossover:** Corta en dos puntos y intercambia el segmento medio
3. **Uniform Crossover:** Cada gen se hereda aleatoriamente de cualquiera de los padres

### Ejemplos de Configuración

```python
# Configuración para entrenamiento rápido
POP_SIZE = 20
GENS = 30
MUT_RATE = 0.15

# Configuración para mejor rendimiento (más lenta)
POP_SIZE = 50
GENS = 100
MUT_RATE = 0.08

# Configuración experimental con alta mutación
POP_SIZE = 40
GENS = 75
MUT_RATE = 0.25
```

## Funcionamiento del Algoritmo

1. **Inicialización:** Crea una población de vectores de pesos aleatorios
2. **Evaluación:** Cada individuo controla el CartPole y se mide su fitness (tiempo de supervivencia)
3. **Selección:** Se mantiene el 20% superior de la población
4. **Reproducción:** Se crean nuevos individuos mediante crossover de los supervivientes
5. **Mutación:** Se aplican pequeñas variaciones aleatorias
6. **Repetición:** El proceso se repite por el número de generaciones especificado

## Salida del Programa

- **Durante la ejecución:** Ventana de simulación visual y logs de debug
- **Al finalizar:** 
  - Mensaje "Training complete."
  - Gráfico mostrando la evolución del fitness promedio y máximo por generación