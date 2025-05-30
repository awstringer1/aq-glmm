// Bernoulli random slopes with CppAD
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
using namespace Rcpp;
#include <cppad/cppad.hpp>

using namespace Eigen;
// Note:: CppAD and Eigen conflict...

template <typename Type>
using Vec = Eigen::Matrix<Type, Dynamic, 1>;
template <typename Type>
using Mat = Eigen::Matrix<Type, Dynamic, Dynamic>;
template <typename Type>
using SpMat = Eigen::SparseMatrix<Type>;

typedef CppAD::AD<double> a_double;
typedef Eigen::Matrix<a_double, Eigen::Dynamic, 1> a_vector;
typedef Eigen::Matrix<a_double, Eigen::Dynamic, Eigen::Dynamic> a_matrix;
typedef Eigen::SparseMatrix<a_double> a_spmatrix;
typedef CppAD::AD<CppAD::AD<double>> a2_double;
typedef Eigen::Matrix<a2_double, Eigen::Dynamic, 1> a2_vector;
typedef Eigen::Matrix<a2_double, Eigen::Dynamic, Eigen::Dynamic> a2_matrix;
typedef Eigen::SparseMatrix<a2_double> a2_spmatrix;

template <typename Type>
using ADVec = Eigen::Matrix<CppAD::AD<Type>, Dynamic, 1>;
template <typename Type>
using ADMat = Eigen::Matrix<CppAD::AD<Type>, Dynamic, Dynamic>;
template <typename Type>
using ADSpMat = Eigen::SparseMatrix<CppAD::AD<Type>>;


/*
  Joint log-likelihood function
  This is what could be replaced by glmmTMB's formula interface
  to cpp function construction; it's very similar to a TMB template.

  I marked with !!! the places that would somehow need to be swapped out
  for different formulas and distributions; the rest is pretty boilerplate.
*/

// Templated function to compute the likelihood
template <class Type>
Type jointloglik(
  Vec<Type> theta, // c(beta, sigma)
  Vec<Type> u, // Random effects ordered by group (u_{1,1}, u_{1,2},..., u_{m,n_m})
  Vec<Type> y, // Response
  Mat<Type> X, // Fixed effects design matrix
  SpMat<Type> Z, // Random effects design matrix
  VectorXi nvec // nvec(i) = size of group i, length = m
) {
  int n = y.size(); // Number of data
  int m = nvec.size(); // Number of groups/subjects
  int p = X.cols(); // Number of fixed effects
  int d = 2; // Random slopes
  double dd = (double) d;
  int s = p + d * (d + 1) / 2; // Number of parameters (reg. coef. + var. comp.)
  if (X.rows() != n) throw std::runtime_error("Length of y does not match rows in X.");
  if (Z.rows() != n) throw std::runtime_error("Length of y does not match rows in Z.");
  if (Z.cols() != u.size()) throw std::runtime_error("Length of u does not match cols in Z.");

  /* !!! 
    Variance parameterization
    This is random slopes; this part would have to be set somehow by
    a formula interface.
  */
  Type s11 = theta(p); // theta(p) = Var(u_1)
  Type s22 = theta(p + 1); // theta(p + 1) = Var(u_2)
  Type s21 = theta(p + 2) * sqrt(s11 * s22); // theta(p + 2) = Corr(u_1, u_2)  
  // Create the precision matrix
  Mat<Type> Sigmainv(d, d);
  Type Sigmadet = (s22 * s11 - s21 * s21);
  Sigmainv(0, 0) = s22 / Sigmadet;
  Sigmainv(1, 1) = s11 / Sigmadet;
  Sigmainv(0, 1) = -s21 / Sigmadet;
  Sigmainv(1, 0) = -s21 / Sigmadet;
  /* END variance parameterization */

  Type nll = Type(0.);
  Vec<Type> eta = X * theta.segment(0, X.cols()) + Z * u;
  
  /* !!!
    Bernoulli log-likelihood
    This would be set by the formula interface
  */
  for (size_t i = 0; i < n; i++) nll -= y(i) * eta(i) - log(1. + exp(eta(i)));
  /* END log-likelihood */
  
  // Normal log likelihood
  // NOTE: this is technically correct if Sigmainv is scalar (1x1) but
  // probably not efficient.
  size_t rowidx = 0;
  Type normpart = Type(0.);
  for (size_t i = 0; i < m; i++) {
    normpart += dd * 0.5 * log(2. * 3.141592653589793115998); // For checking, doesn't affect optimization
    normpart += 0.5 * log(Sigmadet); // determinant
    // normpart += 0.5 * logSdet; // determinant
    normpart += 0.5 * u.segment(rowidx, d).dot(Sigmainv * u.segment(rowidx, d));
    rowidx += d;
  }
  return nll + normpart;
}
/*
  END joint log-likelihood
  The rest of the code is boilerplate, I think.
*/

/*
  Gradient and Hessian of jnll with respect to random effects.
  The sparsity pattern of the Hessian is automatically detected
  by CppAD.
  The function returns a vector containing the gradient and 
  non-zero elements of the Hessian, which is then used efficiently
  by the Newton optimization over random effects and the AGHQ
  functions below.
*/ 

template <class Type> 
Vec<Type> jnllD(
  Vec<Type>& theta, // c(beta, sigma)
  Vec<Type> u,
  Vec<Type>& y, // Response
  Mat<Type>& X, // Fixed effects design matrix
  SpMat<Type>& Z, // Random effects design matrix
  VectorXi& nvec // nvec(i) = size of group i, length = m
) {
  // Wrap in AD types
  ADVec<Type> at = theta.template cast<CppAD::AD<Type>>();
  ADVec<Type> au = u.template cast<CppAD::AD<Type>>();
  ADVec<Type> ay = y.template cast<CppAD::AD<Type>>();
  ADMat<Type> aX = X.template cast<CppAD::AD<Type>>();
  ADSpMat<Type> aZ = Z.template cast<CppAD::AD<Type>>();

  // Start taping
  ADVec<Type> result(1);
  CppAD::Independent(au);
  result(0) = jointloglik(at, au, ay, aX, aZ, nvec);
  CppAD::ADFun<Type> f(au, result);

  // Sparsity pattern of the hessian
  size_t ud = u.size(), d = 2, m = nvec.size();
  Vec<std::set<size_t> > r_set(ud);
  for (size_t i = 0; i < ud; i++) {
    r_set[i].insert(i);
    if (i % 2) {
      // Odd
      r_set[i].insert(i - 1);
    } else {
      // Even
      r_set[i].insert(i + 1);
    }
  }
  // UNIQUE to d = 2 TODO make this general
  size_t rcsize = d * d * m;
  Vec<size_t> row(rcsize), col(rcsize);
  size_t tmp = 0;
  for (size_t i = 0; i < rcsize; i++) {
    row(i) = tmp; 
    if (i % 2) tmp++;
  }
  tmp = 0;
  for (size_t i = 0; i < rcsize; i = i + 4) {
    col(i) = tmp; 
    col(i + 1) = tmp + 1;
    col(i + 2) = tmp; 
    col(i + 3) = tmp + 1;
    tmp += 2;
  }
  f.ForSparseJac(ud, r_set);
  Vec<std::set<size_t> > s_set(1);
  s_set[0].insert(0);
  Vec<std::set<size_t> > p_set = f.RevSparseHes(ud, s_set);
  Vec<Type> w(1);
  w[0] = 1.;
  Vec<Type> hes(row.size());
  CppAD::sparse_hessian_work work;
  work.color_method = "cppad.symmetric"; // The default
  size_t nsweep = f.SparseHessian(u, w, p_set, row, col, hes, work);

  Vec<Type> out(u.size() + hes.size());
  out << f.Jacobian(u), hes;
  return out;
}

// Newton optimization for u given theta
template <class Type> 
Vec<Type> newton_u(
  Vec<Type>& theta, // c(beta, sigma)
  Vec<Type> u,
  Vec<Type>& y, // Response
  Mat<Type>& X, // Fixed effects design matrix
  SpMat<Type>& Z, // Random effects design matrix
  VectorXi& nvec, // nvec(i) = size of group i, length = m
  size_t maxitr = 10,
  size_t minitr = 1,
  bool verbose = false
) {

  size_t m = nvec.size(), d = 2, start = 0, hestart = 0;
  double tol = 1e-08;

  // Initialize Newton's method
  size_t itr = 0;
  Vec<Type> step(u.size()), grad(u.size()), deriv(u.size() + d * d * m), hessvec(d * d * m);
  Mat<Type> hessblock(d, d);
  for (size_t i = 0; i < grad.size(); i++) grad(i) = Type(1. + tol);

  while( (grad.template lpNorm<Infinity>() > tol) & itr < maxitr | itr < minitr) {
    // Increment iteration counter
    itr++;
    if (verbose) Rcout << "Inner Newton iteration " << itr << ", gradient norm: " << grad.template lpNorm<Infinity>() << std::endl;
    // compute the step
    // derivative
    deriv = jnllD(theta, u, y, X, Z, nvec);
    grad = deriv.segment(0, u.size());
    hessvec = deriv.segment(u.size(), d * d * m);
    // Fill the step respecting the sparsity pattern of the Hessian
    start = 0;
    hestart = 0;
    for (size_t i = 0; i < m; i++) {
      hessblock = Map<Mat<Type> >(hessvec.segment(hestart, d * d).data(), d, d);
      step.segment(start, d) = -hessblock.ldlt().solve(deriv.segment(start, d));
      start += d;
      hestart += d * d;
    }
    u += step;
  }
  return u;
}

// negative adaptive quadrature marginal log likelihood
// Templated, in case we want its derivatives
template <class Type>
Type aqmlf(
  Vec<Type> theta, // c(beta, sigma)
  Vec<Type> uhat, // Pass in uhat, to facilitate computing it in/out of the computation graph
  Vec<Type> y, // Response
  Mat<Type> X, // Fixed effects design matrix
  SpMat<Type> Z, // Random effects design matrix
  VectorXi nvec, // nvec(i) = size of group i, length = m
  Mat<Type> nn,
  Vec<Type> ww,
  bool verbose = false
) {
  // Compute the aqml
  size_t kd = nn.rows(), m = nvec.size(), nstart = 0, ustart = 0, hestart = 0, d = 2;
  Type nll = Type(0.), jnlli = Type(0.);
  Vec<Type> uadapt(d), deriv(d * m + d * d * m), tmpvec(kd), yi, zz(d);
  Mat<Type> hessblock(d, d), Xi;
  SpMat<Type> Zi;
  VectorXi nveci(1);
  // Compute the derivatives
  deriv = jnllD(theta, uhat, y, X, Z, nvec);
  Mat<Type> Li(d, d);
  LLT<Mat<Type>> LLt;
  for (size_t i = 0; i < m; i++) {
    hessblock = Map<Mat<Type> >(deriv.segment(d * m + hestart, d * d).data(), d, d);
    LLt.compute(hessblock);
    Li = LLt.matrixL();
    yi = y.segment(nstart, nvec(i));
    Xi = X.block(nstart, 0, nvec(i), X.cols());
    Zi = Z.block(nstart, ustart, nvec(i), d);
    nveci(0) = nvec(i);
    for (size_t j = 0; j < kd; j++) {
      // Adapt
      zz = nn.row(j);
      uadapt = uhat.segment(ustart, d) + Li.template triangularView<Lower>().solve(zz);
      // Joint log likelihood
      jnlli = -jointloglik(theta, uadapt, yi, Xi, Zi, nveci);
      // Add the weight
      tmpvec(j) = jnlli + log(ww(j));
    }
    // NOTE: haven't implemented logsumexp
    nll += -log(tmpvec.array().exp().sum()) + Li.diagonal().array().log().sum();
  
    nstart += nvec(i);
    ustart += d;
    hestart += d * d;
  }
  return nll;
}

// [[Rcpp::export]]
double aqmlfR(
  VectorXd theta, // c(beta, sigma)
  VectorXd y, // Response
  MatrixXd X, // Fixed effects design matrix
  SpMat<double> Z, // Random effects design matrix
  VectorXi nvec, // nvec(i) = size of group i, length = m
  MatrixXd nn,
  VectorXd ww,
  bool verbose = false
) {
  size_t dm = Z.cols(); // Length of u
  VectorXd u(dm);
  u.setZero(); // Initialize Newton for random effects
  VectorXd uhat = newton_u(theta, u, y, X, Z, nvec, 100, 1, verbose);
  return aqmlf(theta, uhat, y, X, Z, nvec, nn, ww, verbose);
}

// [[Rcpp::export]]
VectorXd aqmlgR(
  VectorXd theta, // c(beta, sigma)
  VectorXd y, // Response
  MatrixXd X, // Fixed effects design matrix
  SpMat<double> Z, // Random effects design matrix
  VectorXi nvec, // nvec(i) = size of group i, length = m
  MatrixXd nn,
  VectorXd ww,
  bool verbose = false
) {
  size_t dm = Z.cols(); // Length of u
  VectorXd u(dm);
  u.setZero(); // Initialize Newton for random effects
  VectorXd uhat = newton_u(theta, u, y, X, Z, nvec, 100, 1, verbose);
  // VectorXd uhat = u;
  a_vector at = theta.template cast<a_double>();
  a_vector au = uhat.template cast<a_double>();
  a_vector ay = y.template cast<a_double>();
  a_matrix aX = X.template cast<a_double>();
  ADSpMat<double> aZ = Z.template cast<a_double>();
  a_matrix ann = nn.template cast<a_double>();
  a_vector aww = ww.template cast<a_double>();
  
  a_vector result(1);
  CppAD::Independent(at);
  au = newton_u(at, au, ay, aX, aZ, nvec, 2, 2, verbose);
  result(0) = aqmlf(at, au, ay, aX, aZ, nvec, ann, aww, verbose);
  CppAD::ADFun<double> f(at, result);
  
  return f.Jacobian(theta);
}