import os
from genetic_algorithm import GeneticAlgorithmCartPole
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pop_sizes = [10, 20, 30]
generations_list = [10, 30, 50]
mutation_rate = 0.1
num_episodes = 3

os.makedirs("modelos", exist_ok=True)
os.makedirs("results", exist_ok=True)

def run_all_combinations():
    for pop_size in pop_sizes:
        for gens in generations_list:
            logger.info(f"\n=== Ejecutando combinación: población={pop_size}, generaciones={gens} ===")
            
            ga = GeneticAlgorithmCartPole(
                pop_size=pop_size,
                generations=gens,
                mutation_rate=mutation_rate,
                num_episodes=num_episodes
            )
            
            experiments_results = []
            
            for crossover_method in ["uniform", "single_point", "two_point"]:
                experiment_name = f"{crossover_method}_{pop_size}pob_{gens}gen"
                logger.info(f"\nIniciando experimento: {experiment_name}")
                
                result = ga.run_experiment(
                    crossover_method=crossover_method,
                    experiment_name=experiment_name
                )
                
                ga.save_best_individual_json(result, crossover_method)
                experiments_results.append(result)
            
            # Una vez ejecutados los 3 experimentos, generar gráfico comparativo
            ga.plot_results(experiments_results)
            logger.info(f"Gráfico comparativo generado para población={pop_size}, generaciones={gens}")

if __name__ == "__main__":
    run_all_combinations()
