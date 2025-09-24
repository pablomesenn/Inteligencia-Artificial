import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import time
import logging
import json
from typing import List, Tuple
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('genetic_algorithm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GA parameters
POP_SIZE = 10
GENS = 30
MUT_RATE = 0.1
NUM_EPISODES = 3  # Múltiples episodios para evaluación más robusta
MAX_WORKERS = multiprocessing.cpu_count() - 1  # Usar todos los cores menos uno

logger.info(f"Configuración del algoritmo genético:")
logger.info(f"  - Tamaño de población: {POP_SIZE}")
logger.info(f"  - Generaciones: {GENS}")
logger.info(f"  - Tasa de mutación: {MUT_RATE}")
logger.info(f"  - Episodios por evaluación: {NUM_EPISODES}")
logger.info(f"  - Workers para paralelización: {MAX_WORKERS}")

class GeneticAlgorithmCartPole:
    def __init__(self, pop_size=30, generations=50, mutation_rate=0.1, num_episodes=3):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.num_episodes = num_episodes
        
        # Estadísticas
        self.avg_fitness_per_gen = []
        self.max_fitness_per_gen = []
        self.min_fitness_per_gen = []
        self.std_fitness_per_gen = []
        
        # Mejor individuo encontrado
        self.best_individual = None
        self.best_fitness = -1
        
        # Estadísticas de tiempo
        self.generation_times = []
        self.evaluation_times = []
        
        logger.info("Algoritmo genético inicializado correctamente")
    
    def policy(self, obs: np.ndarray, weights: np.ndarray) -> int:
        """
        Política simple: acción basada en el signo del producto punto
        obs: [cart position, cart velocity, pole angle, pole angular velocity]
        weights: vector de 4 pesos para cada observación
        """
        return 0 if np.dot(obs, weights) < 0 else 1
    
    def evaluate_individual(self, weights: np.ndarray) -> float:
        """
        Evalúa un individuo ejecutando múltiples episodios
        Retorna el fitness promedio de todos los episodios
        """
        total_rewards = []
        
        for episode in range(self.num_episodes):
            # Crear entorno sin visualización para evaluación rápida
            env = gym.make("CartPole-v1")
            total_reward = 0
            obs, _ = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 500:  # Límite de pasos para evitar loops infinitos
                action = self.policy(obs, weights)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                steps += 1
            
            total_rewards.append(total_reward)
            env.close()
        
        # Retornar el promedio de recompensas de todos los episodios
        avg_reward = np.mean(total_rewards)
        return avg_reward
    
    def evaluate_population_parallel(self, population: List[np.ndarray]) -> List[float]:
        """
        Evalúa toda la población en paralelo usando multiprocessing
        """
        logger.info(f"Evaluando población de {len(population)} individuos en paralelo...")
        start_time = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Mapear la función de evaluación a toda la población
            fitness_scores = list(executor.map(self.evaluate_individual, population))
        
        elapsed_time = time.time() - start_time
        self.evaluation_times.append(elapsed_time)
        logger.info(f"Evaluación completada en {elapsed_time:.2f} segundos")
        
        return fitness_scores
    
    def single_point_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Crossover de un punto"""
        point = np.random.randint(1, len(p1))
        child = np.concatenate([p1[:point], p2[point:]])
        return child
    
    def two_point_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Crossover de dos puntos"""
        point1 = np.random.randint(1, len(p1)-1)
        point2 = np.random.randint(point1+1, len(p1))
        child = np.concatenate([p1[:point1], p2[point1:point2], p1[point2:]])
        return child
    
    def uniform_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Crossover uniforme"""
        mask = np.random.randint(0, 2, size=len(p1))
        child = np.where(mask, p1, p2)
        return child
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Aplica mutación con ruido gaussiano
        """
        if np.random.rand() < self.mutation_rate:
            # Mutación gaussiana más suave
            mutation = np.random.normal(0, 0.2, size=individual.shape)
            individual = individual + mutation
            # Mantener los pesos en un rango razonable
            individual = np.clip(individual, -2, 2)
        return individual
    
    def run_experiment(self, crossover_method="uniform", experiment_name="default"):
        """
        Ejecuta un experimento completo del algoritmo genético
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"INICIANDO EXPERIMENTO: {experiment_name}")
        logger.info(f"Método de crossover: {crossover_method}")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        
        # Reiniciar estadísticas
        self.avg_fitness_per_gen = []
        self.max_fitness_per_gen = []
        self.min_fitness_per_gen = []
        self.std_fitness_per_gen = []
        self.best_individual = None
        self.best_fitness = -1
        self.generation_times = []
        self.evaluation_times = []
        
        # Inicializar población
        logger.info("Inicializando población...")
        population = [np.random.uniform(-1, 1, size=4) for _ in range(self.pop_size)]
        logger.info(f"Población inicial creada: {len(population)} individuos con 4 pesos cada uno")
        
        # Seleccionar método de crossover
        crossover_methods = {
            "single_point": self.single_point_crossover,
            "two_point": self.two_point_crossover,
            "uniform": self.uniform_crossover
        }
        crossover_func = crossover_methods.get(crossover_method, self.uniform_crossover)
        
        # Evolución
        for gen in range(self.generations):
            gen_start_time = time.time()
            logger.info(f"\n--- GENERACIÓN {gen + 1}/{self.generations} ---")
            
            # Evaluar fitness en paralelo
            fitness_scores = self.evaluate_population_parallel(population)
            
            # Calcular estadísticas
            avg_fitness = np.mean(fitness_scores)
            max_fitness = np.max(fitness_scores)
            min_fitness = np.min(fitness_scores)
            std_fitness = np.std(fitness_scores)
            
            # Guardar estadísticas
            self.avg_fitness_per_gen.append(avg_fitness)
            self.max_fitness_per_gen.append(max_fitness)
            self.min_fitness_per_gen.append(min_fitness)
            self.std_fitness_per_gen.append(std_fitness)
            
            # Actualizar mejor individuo
            best_idx = np.argmax(fitness_scores)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_individual = population[best_idx].copy()
                logger.info(f"¡NUEVO MEJOR INDIVIDUO! Fitness: {max_fitness:.2f}")
                logger.info(f"Pesos: {self.best_individual}")
            
            # Mostrar estadísticas de la generación
            logger.info(f"Estadísticas - Avg: {avg_fitness:.2f}, Max: {max_fitness:.2f}, "
                       f"Min: {min_fitness:.2f}, Std: {std_fitness:.2f}")
            
            # Selección: top 20%
            num_survivors = max(2, self.pop_size // 5)  # Al menos 2 supervivientes
            sorted_indices = np.argsort(fitness_scores)[::-1]
            survivors_indices = sorted_indices[:num_survivors]
            survivors = [population[i] for i in survivors_indices]
            
            logger.info(f"Supervivientes seleccionados: {len(survivors)} individuos")
            logger.info(f"Fitness de supervivientes: {[f'{fitness_scores[i]:.2f}' for i in survivors_indices[:5]]}")

            # Reproducción
            new_population = []
            mutations_applied = 0
            
            for i in range(self.pop_size):
                # Seleccionar padres aleatoriamente de los supervivientes
                p1, p2 = random.sample(survivors, 2)
                
                # Crossover
                child = crossover_func(p1, p2)
                
                # Mutación
                original_child = child.copy()
                child = self.mutate(child)
                
                if not np.array_equal(original_child, child):
                    mutations_applied += 1
                
                new_population.append(child)
            
            population = new_population
            
            gen_time = time.time() - gen_start_time
            self.generation_times.append(gen_time)
            logger.info(f"Mutaciones aplicadas: {mutations_applied}/{self.pop_size}")
            logger.info(f"Generación completada en {gen_time:.2f} segundos")
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*50}")
        logger.info(f"EXPERIMENTO COMPLETADO: {experiment_name}")
        logger.info(f"Tiempo total: {total_time:.2f} segundos")
        logger.info(f"Mejor fitness alcanzado: {self.best_fitness:.2f}")
        logger.info(f"Mejor individuo: {self.best_individual}")
        logger.info(f"{'='*50}")
        
        return {
            'experiment_name': experiment_name,
            'crossover_method': crossover_method,
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual,
            'avg_fitness_history': self.avg_fitness_per_gen,
            'max_fitness_history': self.max_fitness_per_gen,
            'min_fitness_history': self.min_fitness_per_gen,
            'std_fitness_history': self.std_fitness_per_gen,
            'total_time': total_time,
            'generation_times': self.generation_times,
            'evaluation_times': self.evaluation_times
        }
    
    def save_best_individual_json(self, result: dict, crossover_method: str):
        """
        Guarda el mejor individuo en formato JSON
        """
        # Calcular promedios de tiempo
        avg_generation_time = np.mean(result['generation_times']) if result['generation_times'] else 0
        avg_evaluation_time = np.mean(result['evaluation_times']) if result['evaluation_times'] else 0
        
        # Crear nombre de archivo según el formato solicitado
        filename = f"modelos/{self.generations}-gen-{self.pop_size}-pob-{crossover_method}.json"

        
        # Crear estructura JSON
        json_data = {
            "weights": result['best_individual'].tolist() if result['best_individual'] is not None else [],
            "experiment_info": {
                "experiment_name": result['experiment_name'],
                "crossover_method": result['crossover_method'],
                "best_fitness": result['best_fitness'],
                "total_time": result['total_time'],
                "pop_size": self.pop_size,
                "generations": self.generations,
                "mutation_rate": self.mutation_rate,
                "avg_generation_time": avg_generation_time,
                "avg_evaluation_time": avg_evaluation_time
            },
            "saved_at": datetime.now().isoformat(),
            "fitness": result['best_fitness']
        }
        
        # Guardar archivo JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultado guardado en: {filename}")
        return filename
    
    def demonstrate_best_individual(self):
        """
        Demuestra el mejor individuo encontrado con visualización
        """
        if self.best_individual is None:
            logger.warning("No hay mejor individuo para demostrar")
            return
        
        logger.info("Demostrando el mejor individuo encontrado...")
        env = gym.make("CartPole-v1", render_mode="human")
        
        for episode in range(3):  # Mostrar 3 episodios
            logger.info(f"Episodio de demostración {episode + 1}/3")
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 500:
                action = self.policy(obs, self.best_individual)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                steps += 1
                
                # Log menos frecuente para no saturar
                if steps % 50 == 0:
                    logger.info(f"  Paso {steps}, Recompensa acumulada: {total_reward}")
            
            logger.info(f"Episodio {episode + 1} completado: {total_reward} pasos")
        
        env.close()
    
    def plot_results(self, experiments_results: List[dict]):
        """
        Grafica los resultados de múltiples experimentos
        """
        plt.figure(figsize=(15, 10))

        plt.suptitle(
            f"Experimento de {self.generations} generaciones y {self.pop_size} poblaciones",
            fontsize=16,
            fontweight="bold",
            y=1.02  # un poco más arriba para que no choque con los subplots
        )
        
        # Plot 1: Fitness promedio por experimento
        plt.subplot(2, 2, 1)
        for result in experiments_results:
            plt.plot(range(self.generations), result['avg_fitness_history'], 
                    label=f"{result['experiment_name']} (Avg)", alpha=0.7)
        plt.xlabel("Generación")
        plt.ylabel("Fitness Promedio")
        plt.title("Evolución del Fitness Promedio")
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Fitness máximo por experimento
        plt.subplot(2, 2, 2)
        for result in experiments_results:
            plt.plot(range(self.generations), result['max_fitness_history'], 
                    label=f"{result['experiment_name']} (Max)", alpha=0.7)
        plt.xlabel("Generación")
        plt.ylabel("Fitness Máximo")
        plt.title("Evolución del Fitness Máximo")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'results/{self.generations}-gen-{self.pop_size}-pob-results.png', dpi=300, bbox_inches='tight')

        plt.show()