library(ggplot2)
library(data.table)

set.seed(1)
k <- 3
x0 <- 2.5
n <- 20
x <- runif(n, 0, 5) |> sort()
prob <- 1 / (1 + exp(-k *(x - x0)))
y <- rbinom(n, 1, prob)
df <- data.table(x, prob, y)

ggplot(df, aes(x, y, fill = factor(y))) +
  geom_point(size = 3, pch = 21, color = "black")

# Run logistic model
log_model <- glm(y ~ x, data = df, family = "binomial")
broom::tidy(log_model)

# Run quasibinomial logistic model
qlog_model <- glm(y ~ x, data = df, family = "quasibinomial")
broom::tidy(log_model)