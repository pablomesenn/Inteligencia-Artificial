import pandas as pd
import re

def parse_results_table(text):
    """Parsea la tabla de resultados de texto plano a DataFrame"""
    # Busca todas las líneas que contienen datos numéricos
    lines = text.split('\n')
    
    data = []
    for line in lines:
        line = line.strip()
        # Salta líneas vacías, separadores o encabezados
        if not line or '=' in line or 'RESULTADOS' in line or 'Método' in line:
            continue
        
        # Busca líneas que tengan números (las filas de datos)
        if re.search(r'\d+\.\d+', line):
            parts = line.split()
            if len(parts) >= 9:  # Necesitamos al menos método + 8 métricas
                # El método puede ser una o más palabras
                # Los últimos 8 valores son las métricas
                method = ' '.join(parts[:-8])
                metrics = parts[-8:]
                
                # Valida que las métricas sean números
                try:
                    [float(m) for m in metrics]
                    data.append([method] + metrics)
                except ValueError:
                    continue
    
    if not data:
        raise ValueError("No se pudieron extraer datos de la tabla. Verifica el formato del texto.")
    
    # Crea DataFrame
    columns = ['Método', 'Accuracy', 'ROC-AUC', 'Precision_C1', 'Recall_C1', 
               'F1_C1', 'Precision_C0', 'Recall_C0', 'Especificidad']
    df = pd.DataFrame(data, columns=columns)
    
    # Convierte columnas numéricas
    for col in columns[1:]:
        df[col] = pd.to_numeric(df[col])
    
    return df

def parse_recommendations(text):
    """Extrae información de las recomendaciones"""
    recommendations = {}
    
    # Extrae mejor ROC-AUC
    roc_match = re.search(r'Test ROC-AUC:\s+(\w+)\s+\(([\d.]+)\)', text)
    if roc_match:
        recommendations['roc_auc'] = {'method': roc_match.group(1), 'value': float(roc_match.group(2))}
    
    # Extrae mejor Recall
    recall_match = re.search(r'Recall:\s+(\w+)\s+\(([\d.]+)\)', text)
    if recall_match:
        recommendations['recall'] = {'method': recall_match.group(1), 'value': float(recall_match.group(2))}
    
    # Extrae mejor Precision
    precision_match = re.search(r'Precision:\s+([\w\s]+)\s+\(([\d.]+)\)', text)
    if precision_match:
        recommendations['precision'] = {'method': precision_match.group(1).strip(), 'value': float(precision_match.group(2))}
    
    # Extrae mejor F1
    f1_match = re.search(r'F1-Score:\s+([\w\s]+)\s+\(([\d.]+)\)', text)
    if f1_match:
        recommendations['f1'] = {'method': f1_match.group(1).strip(), 'value': float(f1_match.group(2))}
    
    # Extrae menor overfitting
    overfit_match = re.search(r'Menor Overfitting:\s+(\w+)\s+\(diff=([\d.]+)\)', text)
    if overfit_match:
        recommendations['overfitting'] = {'method': overfit_match.group(1), 'value': float(overfit_match.group(2))}
    
    return recommendations

def generate_latex_table(df, selected_columns):
    """Genera tabla LaTeX con valores máximos en negrita"""
    # Prepara los datos
    table_df = df[selected_columns].copy()
    
    # Encuentra máximos por columna (excluyendo la columna Método)
    max_values = {}
    for col in selected_columns[1:]:
        max_values[col] = table_df[col].max()
    
    # Genera encabezado LaTeX
    latex = "\\begin{table}[H]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{l" + "c" * (len(selected_columns) - 1) + "}\n"
    latex += "\\hline\n"
    
    # Encabezados
    headers = [f"\\textbf{{{col.replace('_', ' ')}}}" for col in selected_columns]
    latex += " & ".join(headers) + " \\\\\n"
    latex += "\\hline\n"
    
    # Datos
    for _, row in table_df.iterrows():
        row_data = [row[selected_columns[0]]]  # Método
        for col in selected_columns[1:]:
            value = row[col]
            if value == max_values[col]:
                row_data.append(f"\\textbf{{{value:.6f}}}")
            else:
                row_data.append(f"{value:.6f}")
        latex += " & ".join(row_data) + " \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Resultados de métricas principales por método de balanceo.}\n"
    latex += "\\end{table}\n"
    
    return latex

def generate_analysis_text(df, recommendations):
    """Genera texto en prosa explicando los resultados"""
    text = "\\textbf{Análisis de Resultados}\n\n"
    
    # Encuentra mejores métodos por métrica
    best_accuracy = df.loc[df['Accuracy'].idxmax()]
    best_roc = df.loc[df['ROC-AUC'].idxmax()]
    best_precision = df.loc[df['Precision_C1'].idxmax()]
    best_recall = df.loc[df['Recall_C1'].idxmax()]
    best_f1 = df.loc[df['F1_C1'].idxmax()]
    
    text += "Los resultados obtenidos muestran un comportamiento diferenciado entre los diversos métodos de balanceo evaluados. "
    text += f"En términos de precisión global (accuracy), el método \\textit{{{best_accuracy['Método']}}} alcanzó el mejor desempeño con {best_accuracy['Accuracy']:.4f}, "
    text += "lo que sugiere una alta capacidad para clasificar correctamente ambas clases en el conjunto de prueba.\n\n"
    
    text += f"Respecto al área bajo la curva ROC (ROC-AUC), que es una métrica robusta ante desbalances de clase, "
    text += f"el método \\textit{{{best_roc['Método']}}} obtuvo el valor más alto de {best_roc['ROC-AUC']:.4f}. "
    text += "Esta métrica es particularmente relevante ya que evalúa el desempeño del clasificador en todos los posibles umbrales de decisión, "
    text += "proporcionando una medida integral de la capacidad discriminativa del modelo.\n\n"
    
    text += f"En cuanto a la precisión para la clase positiva (Precision C1), \\textit{{{best_precision['Método']}}} "
    text += f"destacó con un valor de {best_precision['Precision_C1']:.4f}, indicando una baja tasa de falsos positivos. "
    text += f"Por otro lado, el recall más alto fue alcanzado por \\textit{{{best_recall['Método']}}} con {best_recall['Recall_C1']:.4f}, "
    text += "lo que refleja una excelente capacidad para identificar correctamente las instancias de la clase positiva.\n\n"
    
    text += f"El F1-Score, que balancea precisión y recall, fue maximizado por \\textit{{{best_f1['Método']}}} con {best_f1['F1_C1']:.4f}. "
    text += "Esta métrica es crucial cuando se busca un equilibrio entre minimizar falsos positivos y falsos negativos.\n\n"
    
    # Análisis de recomendaciones
    if recommendations:
        text += "\\textbf*{Consideraciones sobre Generalización}\n\n"
        
        if 'overfitting' in recommendations:
            text += f"Las recomendaciones finales destacan que \\textit{{{recommendations['overfitting']['method']}}} "
            text += f"presenta el menor sobreajuste con una diferencia de {recommendations['overfitting']['value']:.4f} "
            text += "entre el desempeño en validación cruzada y en el conjunto de prueba. "
            text += "Esto sugiere una mayor capacidad de generalización del modelo, lo cual es fundamental para su aplicación en datos no vistos.\n\n"
        
        text += "Es importante notar que existe un compromiso (trade-off) entre diferentes métricas. "
        text += "Mientras algunos métodos maximizan la precisión global, otros priorizan el balance entre clases o la estabilidad del modelo. "
        text += "La selección del método óptimo debe considerar los objetivos específicos del problema, "
        text += "el costo relativo de los errores de clasificación y los requisitos de generalización del modelo.\n\n"
    
    text += "En conclusión, los métodos de balanceo basados en técnicas de remuestreo muestran capacidades diferenciadas "
    text += "para abordar el problema de clases desbalanceadas, y la elección del método más apropiado dependerá "
    text += "del contexto específico de aplicación y las prioridades del problema a resolver.\n"
    
    return text

def main(results_text, recommendations_text):
    """Función principal"""
    print("Procesando datos...")
    
    # Parsea datos
    try:
        df = parse_results_table(results_text)
        print(f"✓ Tabla procesada correctamente: {len(df)} métodos encontrados")
    except Exception as e:
        print(f"✗ Error al procesar la tabla: {e}")
        return None
    
    try:
        recommendations = parse_recommendations(recommendations_text)
        print(f"✓ Recomendaciones procesadas: {len(recommendations)} encontradas")
    except Exception as e:
        print(f"⚠ Advertencia al procesar recomendaciones: {e}")
        recommendations = {}
    
    # Selecciona columnas para la tabla
    selected_columns = ['Método', 'Accuracy', 'ROC-AUC', 'Precision_C1', 'Recall_C1']
    
    # Genera tabla LaTeX
    latex_table = generate_latex_table(df, selected_columns)
    
    # Genera texto de análisis
    analysis_text = generate_analysis_text(df, recommendations)
    
    # Combina todo
    output = "\n" + "="*80 + "\n"
    output += "TABLA LATEX\n"
    output += "="*80 + "\n\n"
    output += latex_table
    output += "\n" + "="*80 + "\n"
    output += "TEXTO DE ANÁLISIS\n"
    output += "="*80 + "\n\n"
    output += analysis_text
    
    return output

# Ejemplo de uso
if __name__ == "__main__":
    # Input 1: Tabla de resultados (copia exactamente tu texto aquí)
    results_input = """

    """
    # Input 2: Recomendaciones (copia exactamente tu texto aquí)
    recommendations_input = """

    """
    
    
    
    # Genera output
    result = main(results_input, recommendations_input)
    
    if result:
        print("\n" + "="*80)
        print("¡PROCESO COMPLETADO!")
        print("="*80)
        print(result)
        
        # Opcional: guardar en archivo
        with open('latex_output.txt', 'w', encoding='utf-8') as f:
            f.write(result)
        print("\n✓ Resultado guardado en 'latex_output.txt'")
    else:
        print("\n✗ No se pudo generar el resultado")