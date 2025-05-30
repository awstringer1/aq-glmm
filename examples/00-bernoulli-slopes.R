### Test the Bernoulli random slopes model in R ###

# Load libraries
library(lme4)
library(Rcpp)
library(nloptr)
library(parallel)

# Compile C++ code
path <- "~/work/projects/aq-glmmtmb"
cpppath <- file.path(path, "examples/cpp")
cppcode <- file.path(cpppath, "bernoulli-randomslope.cpp")
sourceCpp(cppcode)

## Function to generate data
simulate_data <- function(params) {
  m <- params$m
  n <- params$n
  betatrue <- params$beta
  S <- params$S
  
  u <- mvtnorm::rmvnorm(m, sigma = S)
  u <- as.numeric(t(u))
  id <- rep(1:m, each = n)
  x1 <- rnorm(m * n)
  x2 <- rnorm(m * n)
  x3 <- rnorm(m * n)
  x4 <- rnorm(m * n)

  ff <- y ~ x1 + x2 + x3 + x4 + (x1 | id)
  df <- data.frame(id = id, x1 = x1, x2 = x2, x3 = x3, x4 = x4, y = 0)
  reterms <- lme4::glFormula(ff, data = df, family = stats::binomial)
  X <- reterms$X
  Z <- t(reterms$reTrms$Zt)
  eta <- as.numeric(X %*% betatrue + Z %*% u)
  pp <- 1/(1 + exp(-eta))
  df$y <- stats::rbinom(m * n, 1, pp)
  list(
    df = df,
    X = X,
    Z = Z
  )
}

## Function to fit the model 
fitmodel <- function(dat, k) {
  ## Starting values
  startmod <- tryCatch(glm(y ~ x1 + x2 + x3 + x4, data = dat$df, family = binomial()), error = function(e) e, warning = function(w) w)
  if (inherits(startmod, "condition")) {
    startval <- rep(0, ncol(dat$X) + 1)
  } else {
    startval <- c(coef(startmod), c(1, 1, 0))
  }
  ## Quadrature
  gg <- mvQuad::createNIGrid(2, "GHe", k)
  nn <- as.matrix(mvQuad::getNodes(gg))
  ww <- as.numeric(mvQuad::getWeights(gg))
  nvec <- as.integer(unname(table(dat$df$id)))

  ## Optimize
  # Objective function and gradient
  objfun <- function(tt) aqmlfR(tt, dat$df$y, dat$X, dat$Z, nvec, nn, ww)
  objgrad <- function(tt) aqmlgR(tt, dat$df$y, dat$X, dat$Z, nvec, nn, ww)
  
  lower <- c(rep(-Inf, ncol(dat$X)), 0.0001, 0.0001, -.9999)
  upper <- c(rep(Inf, ncol(dat$X) + 2), .9999)
  opt <- optim(startval, objfun, objgrad, method = "L-BFGS-B", lower = lower, upper = upper)

  opt
}

get_results <- function(params) {
  dat <- simulate_data(params)
  modk1 <- fitmodel(dat, 1)
  modk5 <- fitmodel(dat, 5)
  modk15 <- fitmodel(dat, 15)
  modk25 <- fitmodel(dat, 25)

  # Laplace only
  modlme4 <- lme4::glmer(
    y ~ x1 + x2 + x3 + x4 + (x1 | id),
    data = dat$df,
    family = "binomial"
  )
  summlme4 <- summary(modlme4)
  lme4par <- unname(c(summlme4$coefficients[ ,1], diag(summlme4$varcor$id), cov2cor(summlme4$varcor$id)[1, 2]))

  res <- cbind(
    lme4par, modk1$par, modk5$par, modk15$par, modk25$par
  )
  colnames(res) <- c("lme4", "k = 1", "k = 5", "k = 15", "k = 25")
  rownames(res)[6:8] <- c("Var1", "Var2", "Corr")
  res
}




params1 <- list(
  beta = c(1, -1, .3, -.4, 0),
  S = matrix(c(1, .5, .5, 1), 2, 2),
  m = 100,
  n = 3
)
params2 <- params1
params2$n <- 10
set.seed(132546)
res1 <- get_results(params1)
set.seed(132546)
res2 <- get_results(params2)

