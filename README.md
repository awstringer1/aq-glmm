# Multi-level Generalized Linear Mixed Models with Adaptive Quadrature

Beginning of a general implementatio of adaptive quadrature for two/multi-level mixed models.
Relevant recent papers:

  - [https://arxiv.org/abs/2310.01589](https://arxiv.org/abs/2310.01589) (published in Statistics and Computing): empirical work on fitting mixed models with adaptive quadrature.
  - [https://arxiv.org/abs/2202.07864](https://arxiv.org/abs/2202.07864) (published in Bernoulli): theoretical work motivating why adaptive quadrature is needed in this specific type of model,
    including when and why the Laplace approximation may not be sufficiently accurate.

## Initial example

I implemented a Bernoulli random slopes model using adaptive quadrature.
Files:
  
  - `examples/cpp/bernoulli-randomslope.cpp`: functions to be loaded into `R` using `Rcpp::sourceCpp`. Only `jointloglik` would
    need to be modified for implementing alternative response distributions and random effects structures.
    The rest of the code is `boilerplate`.
      - `jointloglik`: function to compute the negative joint log-likelihood of data and random effects.
        Templated for compatibility with `CppAD`. This function contains code which will ideally be replaced
        by code that is set by `glmmTMB`'s formula interface. This function also sets the variance matrix parameterization.
      - `jnllD`: return the gradient and the ordered non-zero elements of the Hessian of the joint negative
        log-likelihood `jointloglik` using automatic differentiation via `CppAD`. The sparsity pattern of the
        Hessian is automatically detected and only non-zero values are computed and stored.
      - `newton_u`: compute the mode of the random effects given a parameter value (and a starting value).
        This function calls `jnllD` and parses the gradient and Hessian of the random effects from it for
        implementing Newton's method. Newton iteration is done group-wise and the full Hessian is never formed,
        rather each group's `2x2` block is used.
      - `aqmlf`: the adaptive quadrature approximate marginal negative log-likelihood for the parameters (regression
        coefficients and variance components). This is the function that should be minimized to obtain maximum
        likelihood estimates. Using one quadrature point returns the Laplace approximation and the results should be
        (and have been observed to be) identical to `lme4::glmer`. This function accepts the nodes and weights of a
        quadrature rule; I have been passing these from `R` where they are computed using the `mvQuad` package. This
        isn't necessary; they are fixed values and we could just tabulate them directly in a text file in the package.
      - `aqmlfR`: exported `double`-typed version of `aqmlf`.
      - `aqmlgR`: compute the gradient of `aqmlf` using `CppAD` and export it to `R`. This is pretty slow because 2nd-order
        AD is already used within `aqmlf`, but it enables the use of `optim(..., method = "L-BFGS-B")` for box-constrained
        quasi-Newton minimization in `R` so I use it for now.
  - `examples/bernoulli-slopes.R`: `R` example of generating data and fitting the random slopes model.
      - The data simulation and model fitting functions are self-explanatory.
      - `get_results` generates data and fits the model with order `k = 1` using both `lme4::glmer` and my
        implementation, as well as for `k = 5, 15, 25` using my implementation. (Note: "order" means "`k` points 'per dimension'";
        for one random effect `k` is the number of points and for `d` random effects the total number of nodes is `k^d`).
      - I fit the model to two sets of data, with `m = 100` groups and `n = 3` or `n = 10` measurements per group, respectively.
        In both cases `lme4` and my implementation with `k = 1` return the same answer. In the `n = 3` case, the answer is different
        for each `k` and in the `n = 10` case it is the same for each `k`, a phenomenon which is partially predicted by our recent
        theory on this topic and aligns with my previous empirical experiments.
