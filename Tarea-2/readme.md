# CartPole Genetic Algorithm - Tarea

## Descripción

Este proyecto implementa un algoritmo genético para entrenar agentes en el entorno **CartPole-v1** de Gymnasium.

Se realizan múltiples experimentos con distintos métodos de crossover y se guardan los mejores modelos en archivos JSON, así como gráficos de resultados.

## Instrucciones de ejecución

1. Navegar a la carpeta `code`:
   ```bash
   cd code
   ```

2. Dar permisos de ejecución al script de instalación y prueba (solo para Linux/macOS):
   ```bash
   chmod +x install-and-test.sh
   ```

3. Ejecutar el script correspondiente según tu sistema:

   **Linux/macOS:**
   ```bash
   ./install-and-test.sh
   ```

   **Windows (PowerShell):**
   ```powershell
   .\install-and-test.ps1
   ```

**Nota:** Los scripts instalarán las dependencias necesarias y ejecutarán los experimentos para probar los diferentes modelos generados durante el entrenamiento. No se realiza entrenamiento adicional, solo se evalúan los modelos existentes.

## Estructura de archivos

- `code/`: Contiene los scripts de ejecución y entrenamiento del algoritmo genético.
- `models/`: Archivos JSON con los mejores individuos por experimento.
- `results/`: Gráficos PNG de la evolución del fitness y comparación de experimentos.
- `install-and-test.sh` / `install-and-test.ps1`: Scripts para instalar dependencias y ejecutar la prueba.