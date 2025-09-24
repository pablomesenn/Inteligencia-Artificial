import os
import json
import numpy as np
import gymnasium as gym
from datetime import datetime

# Carpeta de modelos (ajusta la ruta si tu script no está junto a "modelos")
MODELS_DIR = "modelos"

def listar_modelos():
    """Devuelve una lista de archivos .json en la carpeta modelos."""
    return sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".json")
    ])

def elegir_modelo(archivos):
    """Muestra los archivos y permite elegir uno."""
    print("\nModelos disponibles:\n")
    for i, nombre in enumerate(archivos, start=1):
        print(f"{i}. {nombre}")
    
    while True:
        try:
            idx = int(input("\nIngresa el número del modelo a ejecutar: "))
            if 1 <= idx <= len(archivos):
                return archivos[idx - 1]
            else:
                print("Número fuera de rango.")
        except ValueError:
            print("Por favor ingresa un número válido.")

def cargar_pesos(ruta):
    """Carga y devuelve el array de pesos desde el archivo JSON."""
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["weights"]), data

def mostrar_info_modelo(data):
    """Muestra información detallada del modelo cargado."""
    print("=" * 60)
    print("INFORMACIÓN DEL MODELO")
    print("=" * 60)
    
    # Información del experimento
    exp_info = data.get("experiment_info", {})
    print(f"Nombre del experimento: {exp_info.get('experiment_name', 'N/A')}")
    print(f"Método de crossover: {exp_info.get('crossover_method', 'N/A')}")
    print(f"Mejor fitness obtenido: {exp_info.get('best_fitness', 'N/A')}")
    print(f"Fitness del modelo: {data.get('fitness', 'N/A')}")
    
    print("\nParámetros del algoritmo genético:")
    print(f"  - Tamaño de población: {exp_info.get('pop_size', 'N/A')}")
    print(f"  - Número de generaciones: {exp_info.get('generations', 'N/A')}")
    print(f"  - Tasa de mutación: {exp_info.get('mutation_rate', 'N/A')}")
    
    print("\nTiempos de ejecución:")
    print(f"  - Tiempo total: {exp_info.get('total_time', 0):.3f} segundos")
    print(f"  - Tiempo promedio por generación: {exp_info.get('avg_generation_time', 0):.3f} segundos")
    print(f"  - Tiempo promedio de evaluación: {exp_info.get('avg_evaluation_time', 0):.3f} segundos")
    
    print(f"\nFecha de guardado: {data.get('saved_at', 'N/A')}")
    
    # Mostrar pesos
    weights = np.array(data["weights"])
    print(f"\nPesos del modelo:")
    print(f"  - Dimensiones: {weights.shape}")
    print(f"  - Valores: {weights}")
    print(f"  - Norma L2: {np.linalg.norm(weights):.4f}")
    print(f"  - Media: {np.mean(weights):.4f}")
    print(f"  - Desviación estándar: {np.std(weights):.4f}")
    
    print("=" * 60)

def policy(state, weights):
    """Acción 1 si la combinación lineal supera 0, de lo contrario 0."""
    return int(np.dot(state, weights) > 0)

def evaluar_modelo_detallado(weights, num_episodios=5):
    """Evalúa el modelo con estadísticas detalladas."""
    env = gym.make("CartPole-v1", render_mode="human")
    
    resultados = []
    print("\n" + "=" * 60)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 60)
    
    for ep in range(num_episodios):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Registrar estados y acciones para análisis
        estados = []
        acciones = []
        
        while not done:
            action = policy(state, weights)
            estados.append(state.copy())
            acciones.append(action)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        resultados.append({
            'episodio': ep + 1,
            'reward': total_reward,
            'steps': steps,
            'estados': estados,
            'acciones': acciones
        })
        
        print(f"Episodio {ep + 1}: Recompensa = {total_reward}, Pasos = {steps}")
    
    env.close()
    
    # Calcular estadísticas
    rewards = [r['reward'] for r in resultados]
    steps_list = [r['steps'] for r in resultados]
    
    print("\n" + "-" * 40)
    print("ESTADÍSTICAS DE EVALUACIÓN")
    print("-" * 40)
    print(f"Recompensa promedio: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Recompensa máxima: {np.max(rewards)}")
    print(f"Recompensa mínima: {np.min(rewards)}")
    print(f"Pasos promedio: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}")
    print(f"Pasos máximos: {np.max(steps_list)}")
    print(f"Pasos mínimos: {np.min(steps_list)}")
    
    # Análisis de acciones
    todas_acciones = []
    for r in resultados:
        todas_acciones.extend(r['acciones'])
    
    accion_0_count = todas_acciones.count(0)
    accion_1_count = todas_acciones.count(1)
    total_acciones = len(todas_acciones)
    
    print(f"\nDistribución de acciones:")
    print(f"  - Acción 0 (izquierda): {accion_0_count} ({accion_0_count/total_acciones*100:.1f}%)")
    print(f"  - Acción 1 (derecha): {accion_1_count} ({accion_1_count/total_acciones*100:.1f}%)")
    
    return resultados

def main():
    modelos = listar_modelos()
    if not modelos:
        print("No se encontraron archivos JSON en la carpeta 'modelos'.")
        return
    
    elegido = elegir_modelo(modelos)
    ruta_modelo = os.path.join(MODELS_DIR, elegido)
    
    print(f"\nCargando modelo desde: {ruta_modelo}")
    weights, data = cargar_pesos(ruta_modelo)
    
    # Mostrar información del modelo
    mostrar_info_modelo(data)
    
    # Preguntar cuántos episodios evaluar
    while True:
        try:
            num_eps = int(input(f"\n¿Cuántos episodios deseas evaluar? (por defecto 3): ") or "3")
            if num_eps > 0:
                break
            else:
                print("Por favor ingresa un número mayor a 0.")
        except ValueError:
            print("Por favor ingresa un número válido.")
    
    # Evaluar el modelo
    resultados = evaluar_modelo_detallado(weights, num_eps)
    

if __name__ == "__main__":
    main()