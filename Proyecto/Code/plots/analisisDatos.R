# ============================================================================
# ANÁLISIS EXPLORATORIO SIMPLIFICADO - CKD DATASET
# Sin dependencias de paquetes opcionales
# ============================================================================

library(dplyr)
library(ggplot2)
library(tidyr)
library(corrplot)

# ============================================================================
# FUNCIÓN PRINCIPAL SIMPLIFICADA
# ============================================================================

run_simple_eda <- function(data) {
  
  results <- list()
  
  cat("═══════════════════════════════════════════════════════════════\n")
  cat("    ANÁLISIS EXPLORATORIO SIMPLIFICADO - CKD DATASET\n")
  cat("═══════════════════════════════════════════════════════════════\n\n")
  
  # ═══════════════════════════════════════════════════════════════
  # 1. ESTRUCTURA BÁSICA
  # ═══════════════════════════════════════════════════════════════
  
  cat("📊 DIMENSIONES DEL DATASET\n")
  cat("─────────────────────────────────────────────────────────────\n")
  cat(sprintf("Pacientes: %d\n", nrow(data)))
  cat(sprintf("Variables: %d\n\n", ncol(data)))
  
  results$n_rows <- nrow(data)
  results$n_cols <- ncol(data)
  
  # ═══════════════════════════════════════════════════════════════
  # 2. RESUMEN ESTADÍSTICO
  # ═══════════════════════════════════════════════════════════════
  
  cat("📈 RESUMEN ESTADÍSTICO\n")
  cat("─────────────────────────────────────────────────────────────\n")
  
  numeric_summary <- data %>%
    select(where(is.numeric), -PatientID) %>%
    pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
    group_by(Variable) %>%
    summarise(
      N = sum(!is.na(Value)),
      Missing = sum(is.na(Value)),
      Mean = round(mean(Value, na.rm = TRUE), 2),
      SD = round(sd(Value, na.rm = TRUE), 2),
      Min = round(min(Value, na.rm = TRUE), 2),
      Median = round(median(Value, na.rm = TRUE), 2),
      Max = round(max(Value, na.rm = TRUE), 2),
      .groups = "drop"
    )
  
  print(numeric_summary, n = Inf)
  cat("\n")
  
  results$summary <- numeric_summary
  
  # ═══════════════════════════════════════════════════════════════
  # 3. BALANCE DE CLASES
  # ═══════════════════════════════════════════════════════════════
  
  cat("🎯 BALANCE DE DIAGNOSIS\n")
  cat("─────────────────────────────────────────────────────────────\n")
  
  diagnosis_summary <- data %>%
    count(Diagnosis) %>%
    mutate(
      Class = c("No CKD", "CKD"),
      Percentage = round(n / sum(n) * 100, 2)
    ) %>%
    select(Class, Count = n, Percentage)
  
  print(diagnosis_summary)
  
  balance_ratio <- max(diagnosis_summary$Count) / min(diagnosis_summary$Count)
  cat(sprintf("\nRatio de balance: %.2f:1 ", balance_ratio))
  
  if(balance_ratio < 1.5) {
    cat("✓ BALANCEADO\n\n")
  } else if(balance_ratio < 3) {
    cat("⚠ LIGERAMENTE DESBALANCEADO\n\n")
  } else {
    cat("✗ DESBALANCEADO\n\n")
  }
  
  results$diagnosis_balance <- diagnosis_summary
  
  # ═══════════════════════════════════════════════════════════════
  # 4. CORRELACIONES
  # ═══════════════════════════════════════════════════════════════
  
  cat("🔗 ANÁLISIS DE CORRELACIONES\n")
  cat("─────────────────────────────────────────────────────────────\n")
  
  cor_vars <- data %>%
    select(Age, BMI, SystolicBP, DiastolicBP, HbA1c, SerumCreatinine,
           BUNLevels, GFR, ProteinInUrine, ACR, HemoglobinLevels,
           CholesterolTotal, QualityOfLifeScore) %>%
    na.omit()
  
  cor_matrix <- cor(cor_vars)
  
  # Encontrar correlaciones fuertes
  strong_corr <- data.frame()
  for(i in 1:(ncol(cor_matrix)-1)) {
    for(j in (i+1):ncol(cor_matrix)) {
      if(abs(cor_matrix[i, j]) > 0.5) {
        strong_corr <- rbind(strong_corr, data.frame(
          Var1 = rownames(cor_matrix)[i],
          Var2 = colnames(cor_matrix)[j],
          Correlation = round(cor_matrix[i, j], 3)
        ))
      }
    }
  }
  
  if(nrow(strong_corr) > 0) {
    strong_corr <- strong_corr %>% arrange(desc(abs(Correlation)))
    cat("Correlaciones fuertes (|r| > 0.5):\n")
    print(strong_corr, n = 20)
  } else {
    cat("No se encontraron correlaciones > 0.5\n")
  }
  cat("\n")
  
  results$cor_matrix <- cor_matrix
  results$strong_correlations <- strong_corr
  
  # Plot de correlación
  cat("Generando plot de correlación...\n")
  corrplot(cor_matrix, 
           method = "color",
           type = "upper",
           order = "hclust",
           tl.col = "black",
           tl.srt = 45,
           tl.cex = 0.8,
           title = "Matriz de Correlaciones",
           mar = c(0,0,2,0))
  
  # ═══════════════════════════════════════════════════════════════
  # 5. COMPARACIÓN CKD vs NO CKD
  # ═══════════════════════════════════════════════════════════════
  
  cat("\n📊 COMPARACIÓN CKD vs NO CKD\n")
  cat("─────────────────────────────────────────────────────────────\n")
  
  key_vars <- c("Age", "BMI", "GFR", "SerumCreatinine", "BUNLevels",
                "ProteinInUrine", "HbA1c", "SystolicBP", "QualityOfLifeScore")
  
  comparison <- data.frame()
  
  for(var in key_vars) {
    if(var %in% names(data)) {
      no_ckd <- data %>% filter(Diagnosis == 0) %>% pull(!!sym(var))
      ckd <- data %>% filter(Diagnosis == 1) %>% pull(!!sym(var))
      
      no_ckd <- no_ckd[!is.na(no_ckd)]
      ckd <- ckd[!is.na(ckd)]
      
      t_test <- t.test(no_ckd, ckd)
      
      # Cohen's d
      pooled_sd <- sqrt((var(no_ckd) + var(ckd)) / 2)
      cohens_d <- (mean(ckd) - mean(no_ckd)) / pooled_sd
      
      comparison <- rbind(comparison, data.frame(
        Variable = var,
        No_CKD_Mean = round(mean(no_ckd), 2),
        CKD_Mean = round(mean(ckd), 2),
        Difference = round(mean(ckd) - mean(no_ckd), 2),
        P_Value = round(t_test$p.value, 4),
        Cohens_d = round(cohens_d, 3),
        Effect = case_when(
          abs(cohens_d) < 0.2 ~ "Negligible",
          abs(cohens_d) < 0.5 ~ "Small",
          abs(cohens_d) < 0.8 ~ "Medium",
          TRUE ~ "Large"
        )
      ))
    }
  }
  
  comparison <- comparison %>% arrange(desc(abs(Cohens_d)))
  print(comparison)
  cat("\n")
  
  results$comparison <- comparison
  
  # ═══════════════════════════════════════════════════════════════
  # 6. OUTLIERS
  # ═══════════════════════════════════════════════════════════════
  
  cat("🔍 DETECCIÓN DE OUTLIERS\n")
  cat("─────────────────────────────────────────────────────────────\n")
  
  outlier_summary <- data.frame()
  
  for(var in key_vars) {
    if(var %in% names(data)) {
      var_data <- data[[var]][!is.na(data[[var]])]
      
      Q1 <- quantile(var_data, 0.25)
      Q3 <- quantile(var_data, 0.75)
      IQR_val <- Q3 - Q1
      lower_fence <- Q1 - 1.5 * IQR_val
      upper_fence <- Q3 + 1.5 * IQR_val
      
      n_outliers <- sum(var_data < lower_fence | var_data > upper_fence)
      pct_outliers <- round(n_outliers / length(var_data) * 100, 2)
      
      outlier_summary <- rbind(outlier_summary, data.frame(
        Variable = var,
        N_Outliers = n_outliers,
        Percent = pct_outliers,
        Lower_Fence = round(lower_fence, 2),
        Upper_Fence = round(upper_fence, 2)
      ))
    }
  }
  
  outlier_summary <- outlier_summary %>% arrange(desc(Percent))
  print(outlier_summary)
  cat("\n")
  
  results$outliers <- outlier_summary
  
  # ═══════════════════════════════════════════════════════════════
  # 7. VARIABLES CATEGÓRICAS
  # ═══════════════════════════════════════════════════════════════
  
  cat("📋 DISTRIBUCIÓN DE VARIABLES CATEGÓRICAS\n")
  cat("─────────────────────────────────────────────────────────────\n")
  
  # Gender
  if("Gender" %in% names(data)) {
    cat("\nGender:\n")
    gender_table <- table(data$Gender)
    gender_df <- data.frame(
      Category = c("Male", "Female"),
      Count = as.vector(gender_table),
      Percent = round(prop.table(gender_table) * 100, 2)
    )
    print(gender_df)
  }
  
  # Ethnicity
  if("Ethnicity" %in% names(data)) {
    cat("\nEthnicity:\n")
    eth_table <- table(data$Ethnicity)
    eth_df <- data.frame(
      Category = c("Caucasian", "African American", "Asian", "Other"),
      Count = as.vector(eth_table),
      Percent = round(prop.table(eth_table) * 100, 2)
    )
    print(eth_df)
  }
  
  # Factores de riesgo binarios
  cat("\n🔍 PREVALENCIA DE FACTORES DE RIESGO:\n")
  binary_vars <- c("Smoking", "FamilyHistoryKidneyDisease", 
                   "FamilyHistoryHypertension", "FamilyHistoryDiabetes",
                   "Edema", "ACEInhibitors", "Diuretics")
  
  binary_summary <- data.frame()
  for(var in binary_vars) {
    if(var %in% names(data)) {
      present <- sum(data[[var]] == 1, na.rm = TRUE)
      total <- sum(!is.na(data[[var]]))
      pct <- round(present / total * 100, 2)
      
      binary_summary <- rbind(binary_summary, data.frame(
        Variable = var,
        Present = present,
        Percent = pct
      ))
    }
  }
  
  binary_summary <- binary_summary %>% arrange(desc(Percent))
  print(binary_summary)
  cat("\n")
  
  results$binary_prevalence <- binary_summary
  
  # ═══════════════════════════════════════════════════════════════
  # RESUMEN FINAL
  # ═══════════════════════════════════════════════════════════════
  
  cat("═══════════════════════════════════════════════════════════════\n")
  cat("  ✓ ANÁLISIS EXPLORATORIO COMPLETADO\n")
  cat("═══════════════════════════════════════════════════════════════\n\n")
  
  cat("Componentes en results:\n")
  cat("  • results$summary - Estadísticas descriptivas\n")
  cat("  • results$diagnosis_balance - Balance de clases\n")
  cat("  • results$cor_matrix - Matriz de correlaciones\n")
  cat("  • results$strong_correlations - Correlaciones fuertes\n")
  cat("  • results$comparison - Comparación CKD vs No CKD\n")
  cat("  • results$outliers - Análisis de outliers\n")
  cat("  • results$binary_prevalence - Factores de riesgo\n\n")
  
  return(results)
}

# ============================================================================
# VISUALIZACIONES BÁSICAS
# ============================================================================

create_basic_plots <- function(data, eda_results) {
  
  cat("\n═══════════════════════════════════════════════════════════════\n")
  cat("  GENERANDO VISUALIZACIONES BÁSICAS\n")
  cat("═══════════════════════════════════════════════════════════════\n\n")
  
  plots <- list()
  
  # 1. Histogramas por diagnosis
  cat("1. Histogramas de variables clave...\n")
  
  hist_data <- data %>%
    select(GFR, SerumCreatinine, HbA1c, BMI, SystolicBP, Diagnosis) %>%
    mutate(Diagnosis = factor(Diagnosis, labels = c("No CKD", "CKD"))) %>%
    pivot_longer(cols = -Diagnosis, names_to = "Variable", values_to = "Value")
  
  plots$histograms <- ggplot(hist_data, aes(x = Value, fill = Diagnosis)) +
    geom_histogram(alpha = 0.6, bins = 30, position = "identity") +
    facet_wrap(~ Variable, scales = "free", ncol = 3) +
    scale_fill_manual(values = c("No CKD" = "#00BA38", "CKD" = "#F8766D")) +
    labs(title = "Distribuciones por Diagnosis",
         x = "Valor", y = "Frecuencia") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  print(plots$histograms)
  
  # 2. Boxplots comparativos
  cat("2. Boxplots comparativos...\n")
  
  plots$boxplots <- ggplot(hist_data, aes(x = Diagnosis, y = Value, 
                                          fill = Diagnosis)) +
    geom_boxplot(alpha = 0.7) +
    facet_wrap(~ Variable, scales = "free_y", ncol = 3) +
    scale_fill_manual(values = c("No CKD" = "#00BA38", "CKD" = "#F8766D")) +
    labs(title = "Comparación CKD vs No CKD",
         x = "", y = "Valor") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  print(plots$boxplots)
  
  # 3. Scatter GFR vs Creatinina
  cat("3. Scatter plot: GFR vs Creatinina...\n")
  
  plots$scatter <- ggplot(data, aes(x = SerumCreatinine, y = GFR, 
                                    color = factor(Diagnosis))) +
    geom_point(alpha = 0.4) +
    geom_smooth(method = "loess", se = TRUE) +
    scale_color_manual(values = c("0" = "#00BA38", "1" = "#F8766D"),
                       labels = c("No CKD", "CKD"),
                       name = "Diagnosis") +
    geom_hline(yintercept = 60, linetype = "dashed", color = "red") +
    annotate("text", x = max(data$SerumCreatinine) * 0.9, y = 65,
             label = "GFR = 60 (umbral CKD)", color = "red") +
    labs(title = "GFR vs Creatinina Sérica",
         x = "Creatinina Sérica (mg/dL)",
         y = "GFR (mL/min/1.73m²)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  print(plots$scatter)
  
  # 4. Effect sizes
  cat("4. Effect sizes (Cohen's d)...\n")
  
  plots$effect_sizes <- ggplot(eda_results$comparison, 
                               aes(x = reorder(Variable, Cohens_d), 
                                   y = Cohens_d, fill = Effect)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    geom_hline(yintercept = c(-0.8, -0.5, -0.2, 0.2, 0.5, 0.8),
               linetype = "dashed", alpha = 0.3) +
    coord_flip() +
    scale_fill_viridis(discrete = TRUE) +
    labs(title = "Effect Sizes (Cohen's d): CKD vs No CKD",
         subtitle = "Líneas: |d|=0.2 (small), 0.5 (medium), 0.8 (large)",
         x = "", y = "Cohen's d") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 9))
  
  print(plots$effect_sizes)
  
  cat("\n✓ Visualizaciones básicas completadas\n\n")
  
  return(plots)
}

# ============================================================================
# EJEMPLO DE USO
# ============================================================================

cat("═══════════════════════════════════════════════════════════════\n")
cat("  USO:\n")
cat("═══════════════════════════════════════════════════════════════\n\n")
cat("# 1. Cargar datos\n")
ckd_data <- read.csv("/Users/samircabrera/Development/Universidad/Inteligencia Artificial/Inteligencia-Artificial/Proyecto/Dataset/Chronic_Kidney_Dsease_data.csv")
cat("# 2. Ejecutar EDA\n")
eda_results <- run_simple_eda(ckd_data)
cat("# 3. Crear visualizaciones\n")
plots <- create_basic_plots(ckd_data, eda_results)
cat("# 4. Guardar\n")
saveRDS(eda_results, "eda_results.rds")
ggsave('histograms.png', plots$histograms, width=12, height=8)
cat("═══════════════════════════════════════════════════════════════\n")

