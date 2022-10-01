/*
 * glm.h
 *
 *  Created on: Apr 11, 2014
 *      Author: Peng Jiang
 *  Linear models such as OLS, logistic regression
 */

#ifndef GLM_H_
#define GLM_H_

#include <gsl/gsl_blas.h>

// all fail flags must have value < 0
#define REG_FAIL -1
#define CONVERGE_FAIL -2

// workspace for logistic regression
typedef struct logistic_regression_workspace
{
	// global input parameters, no need to be allocated
	const gsl_matrix *X;
	const gsl_vector *Y;

	// dimension of X : n*p. length of Y: n
	size_t n, p;

	// internal variables for iterative optimization
	// For Newton Raphson
	gsl_vector *P, *W, *error, *H, *U, *delta, *beta, *stderr, *z, *pvalue;
	gsl_matrix *wX, *H_t, *I, *L;

} LGR_space;

LGR_space * alloc_LGR_space(const gsl_matrix *X, const gsl_vector *Y);

void free_LGR_space(LGR_space *ls, const int delete_result);


// logistic regression with Y as vector. Return regression status
int logistic_regression(
		LGR_space *ls,
		const size_t maxIter, const double tol, const double max_delta,
		const int correction, const int verbose);

#endif /* GLM_H_ */
