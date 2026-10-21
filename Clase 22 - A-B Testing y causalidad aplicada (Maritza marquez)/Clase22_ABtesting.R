# =====================================================================
# Clase 22 — Experimentación, A/B Testing y causalidad aplicada
# Práctica mejorada en R · Diplomado en Data Science UC
# Datos: ab_checkout.csv, ab_srm.csv, did_app_regiones.csv
# =====================================================================

# Paquetes mínimos
library(dplyr)
library(ggplot2)
library(broom)

# ---------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------
srm_test <- function(grupo, ratio = c(0.5, 0.5)) {
  tab <- table(grupo)
  tab <- tab[c("control", "tratamiento")]
  out <- chisq.test(tab, p = ratio)
  tibble(
    grupo = names(tab),
    usuarios = as.integer(tab),
    esperado = as.numeric(sum(tab) * ratio),
    residuo = usuarios - esperado,
    p_valor = out$p.value
  )
}

resumen_ab <- function(data) {
  data |>
    group_by(grupo) |>
    summarise(
      usuarios = n(),
      conversiones = sum(convierte),
      tasa = mean(convierte),
      .groups = "drop"
    )
}

# ---------------------------------------------------------------------
# PARTE 0 · Diseño: ¿cuántos usuarios necesitábamos?
# ---------------------------------------------------------------------
# Conversión histórica ~9%; el equipo busca detectar +1.5 pp.
power.prop.test(p1 = 0.09, p2 = 0.105,
                sig.level = 0.05, power = 0.80)

# Discusión:
# - ¿El experimento real tiene al menos ese n por grupo?
# - ¿El MDE de 1.5 pp es una diferencia de negocio relevante?

# ---------------------------------------------------------------------
# PARTE 1 · Cargar el experimento y chequear sanidad (SRM)
# ---------------------------------------------------------------------
ab <- read.csv("ab_checkout.csv") |>
  mutate(
    grupo = factor(grupo, levels = c("control", "tratamiento")),
    trat = as.integer(grupo == "tratamiento"),
    dispositivo = factor(dispositivo),
    region = factor(region)
  )

str(ab)
summary(ab)

# SRM: si p < 0.001, se audita y no se reporta efecto.
srm_test(ab$grupo)

# ---------------------------------------------------------------------
# PARTE 2 · Análisis del efecto principal
# ---------------------------------------------------------------------
res <- resumen_ab(ab)
res

# (a) Test de dos proporciones: tratamiento vs control
prop.test(x = res$conversiones[c(2, 1)],
          n = res$usuarios[c(2, 1)],
          correct = FALSE)

# (b) Modelo lineal de probabilidad: coeficiente = diferencia en pp
mpl <- lm(convierte ~ trat, data = ab)
tidy(mpl, conf.int = TRUE)

# (c) Regresión logística: odds ratio
logit <- glm(convierte ~ trat, family = binomial, data = ab)
tidy(logit, conf.int = TRUE, exponentiate = TRUE)

# Visualización simple de tasas
res |>
  ggplot(aes(x = grupo, y = tasa)) +
  geom_col(width = 0.55) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(title = "Tasa de conversión por grupo",
       x = NULL, y = "Conversión") +
  theme_minimal(base_size = 13)

# ---------------------------------------------------------------------
# PARTE 3 · Más precisión: covariables y CUPED
# ---------------------------------------------------------------------
# Ajuste con covariable pre-tratamiento. No arregla sesgo: reduce varianza.
mpl_adj <- lm(convierte ~ trat + gasto_pre + sesiones_pre + dispositivo, data = ab)

bind_rows(
  tidy(mpl) |> filter(term == "trat") |> mutate(modelo = "sin ajuste"),
  tidy(mpl_adj) |> filter(term == "trat") |> mutate(modelo = "con covariables")
) |>
  select(modelo, estimate, std.error, statistic, p.value)

# CUPED usando gasto_pre como covariable pre-experimento
theta <- cov(ab$convierte, ab$gasto_pre) / var(ab$gasto_pre)
ab <- ab |>
  mutate(y_cuped = convierte - theta * (gasto_pre - mean(gasto_pre)))

mpl_cuped <- lm(y_cuped ~ trat, data = ab)

se_sin <- tidy(mpl) |> filter(term == "trat") |> pull(std.error)
se_cuped <- tidy(mpl_cuped) |> filter(term == "trat") |> pull(std.error)
reduccion_var <- 1 - (se_cuped / se_sin)^2
cat("Reducción de varianza con CUPED:", round(100 * reduccion_var, 1), "%\n")

tidy(mpl_cuped, conf.int = TRUE)

# ---------------------------------------------------------------------
# PARTE 4 · Heterogeneidad por dispositivo
# ---------------------------------------------------------------------
# Esto debe interpretarse con cautela si no estaba preespecificado.
het <- lm(convierte ~ trat * dispositivo, data = ab)
tidy(het, conf.int = TRUE)

ab |>
  group_by(dispositivo, grupo) |>
  summarise(tasa = mean(convierte), usuarios = n(), .groups = "drop") |>
  ggplot(aes(x = dispositivo, y = tasa, fill = grupo)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(title = "Conversión por dispositivo y grupo",
       x = NULL, y = "Conversión") +
  theme_minimal(base_size = 13)

# ---------------------------------------------------------------------
# PARTE 5 · Diagnóstico de experimento roto: ab_srm.csv
# ---------------------------------------------------------------------
srm <- read.csv("ab_srm.csv") |>
  mutate(
    grupo = factor(grupo, levels = c("control", "tratamiento")),
    dispositivo = factor(dispositivo)
  )

srm_test(srm$grupo)

# ¿Dónde se pierde el balance?
table(srm$grupo, srm$dispositivo)
prop.table(table(srm$grupo, srm$dispositivo), margin = 1)

# Discusión:
# 1) ¿En qué segmento parece haberse roto la asignación/tracking?
# 2) ¿Por qué NO debemos reportar el efecto de conversión en este archivo?

# ---------------------------------------------------------------------
# PARTE 6 · Cuando no se puede aleatorizar: Diferencia-en-diferencias
# ---------------------------------------------------------------------
did <- read.csv("did_app_regiones.csv") |>
  mutate(
    trat = as.integer(zona == "RM"),
    post = as.integer(post == 1),
    mes = factor(mes, levels = sort(unique(mes)))
  )

# Tendencias promedio por zona y mes
tend <- did |>
  group_by(zona, mes) |>
  summarise(uso = mean(uso_app), .groups = "drop")

tend

ggplot(tend, aes(x = mes, y = uso, group = zona, color = zona)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_vline(xintercept = 3.5, linetype = "dashed") +
  labs(title = "Chequeo visual de tendencias paralelas",
       subtitle = "La línea vertical separa pre y post intervención",
       x = "Mes", y = "Uso promedio de la app") +
  theme_minimal(base_size = 13)

# Modelo DiD: el efecto está en trat:post
did_mod <- lm(uso_app ~ trat * post, data = did)
tidy(did_mod, conf.int = TRUE)

# Versión con efectos fijos simples de mes y comuna
# Útil para discutir extensiones cuando hay panel longitudinal.
did_fe <- lm(uso_app ~ trat:post + comuna + mes, data = did)
tidy(did_fe, conf.int = TRUE) |>
  filter(term == "trat:post")

# ---------------------------------------------------------------------
# Preguntas finales
# ---------------------------------------------------------------------
# 1. ¿El nuevo checkout debería lanzarse? Justifique con efecto, IC y guardrails.
# 2. ¿Qué haría si el efecto fuera positivo pero el SRM fuera significativo?
# 3. ¿Qué cambia al usar CUPED: el estimador, la varianza o ambas cosas?
# 4. ¿Qué evidencia visual apoya las tendencias paralelas en el ejemplo DiD?
# =====================================================================

