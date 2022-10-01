#include "glm.h"
#include "util.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multiroots.h>

// numerical tolerance for really small value
#define EPS 1e-10

// EPS correction for sqrt function on nearly 0 value
#define SQRT_EPS(x) (fabs(x)<EPS?0:sqrt(x))


// allocate the internal variable space
LGR_space * alloc_LGR_space(const gsl_matrix *X, const gsl_vector *Y)
{
	size_t n = X->size1, p = X->size2;
	LGR_space *lr = (LGR_space*)malloc(sizeof(LGR_space));

	lr->X = X;
	lr->Y = Y;

	lr->n = n;
	lr->p = p;

	lr->P = gsl_vector_alloc(n);		// P(Y=1|X)
	lr->W = gsl_vector_alloc(n);		// diagonal elements : P(1-P)
	lr->error = gsl_vector_alloc(n);	// Y-P
	lr->H = gsl_vector_alloc(n);		// diagonal elements of Hat matrix: (W^0.5)X I^-1 X'(W^0.5)
	lr->U = gsl_vector_alloc(p);		// gradient: t(Y-P)X
	lr->delta = gsl_vector_alloc(p);	// change step of beta

	lr->wX = gsl_matrix_alloc(n, p);	// (W^0.5)X
	lr->H_t = gsl_matrix_alloc(n, p);// auxiliary space for H diagonal calculation
	lr->I = gsl_matrix_alloc(p, p);	// Information matrix: X'WX
	lr->L = gsl_matrix_alloc(p, p);	// Cholesky decomposition L

	// results
	lr->beta = gsl_vector_alloc(p);
	lr->stderr = gsl_vector_alloc(p);
	lr->z = gsl_vector_alloc(p);
	lr->pvalue = gsl_vector_alloc(p);

	return lr;
}


void free_LGR_space(LGR_space *ls, const int delete_result)
{
	gsl_vector_free(ls->P);
	gsl_vector_free(ls->W);
	gsl_vector_free(ls->error);
	gsl_vector_free(ls->U);
	gsl_vector_free(ls->delta);
	gsl_vector_free(ls->H);

	gsl_matrix_free(ls->wX);
	gsl_matrix_free(ls->H_t);
	gsl_matrix_free(ls->I);
	gsl_matrix_free(ls->L);

	if(delete_result){
		gsl_vector_free(ls->beta);
		gsl_vector_free(ls->stderr);
		gsl_vector_free(ls->z);
		gsl_vector_free(ls->pvalue);
	}else{
		// still keep the data block
		gsl_vector_partial_free(ls->beta);
		gsl_vector_partial_free(ls->stderr);
		gsl_vector_partial_free(ls->z);
		gsl_vector_partial_free(ls->pvalue);
	}

	free(ls);
}


// swap the content of pointer for two vectors
inline void swap_gsl_vector(gsl_vector **a, gsl_vector **b)
{
	gsl_vector *c = *a;
	*a = *b;
	*b = c;
}


// logistic regression core procedure: return number of iterations if succeeded.
int logistic_regression(
		LGR_space *ls,
		const size_t maxIter, const double tol, const double max_delta,
		const int correction, const int verbose)
{
	double w, norm_delta, norm_beta, step_ratio;

	int status=GSL_CONTINUE, returnflag = 0;

	size_t i,j, n=ls->n, p=ls->p;

	// hook up to workspace
	const gsl_matrix *X = ls->X;
	const gsl_vector *Y = ls->Y;

	gsl_vector *P = ls->P, *W = ls->W, *error = ls->error, *H = ls->H,
			*U = ls->U, *delta = ls->delta;

	gsl_matrix *wX = ls->wX, *H_t = ls->H_t, *I = ls->I, *L = ls->L;

	// set initial start as all zeros
	gsl_vector_set_zero(ls->beta);

	for(i=0 ; i<maxIter ; i++)
	{
		// probability vector P = sigmod(X*beta)
		gsl_blas_dgemv(CblasNoTrans, 1, X, ls->beta, 0, P);

		for(j=0;j<P->size;j++)
		{
			w = gsl_vector_get(P,j);
			gsl_vector_set(P, j, gsl_cdf_logistic_P(w,1));
		}

		// diagonal(W) = P*(1-P)
		gsl_vector_memcpy(W, P);
		gsl_vector_mul(W,P);
		gsl_vector_sub(W,P);
		gsl_vector_scale(W,-1);

		// error = Y-P
		gsl_vector_memcpy(error, Y);
		gsl_vector_sub(error,P);


		// Information matrix I = X'WX , Hessian matrix = -I

		// wX = W^0.5 * X
		gsl_matrix_memcpy(wX,X);

		for(j=0;j<n;j++)
		{
			gsl_vector_view X_row = gsl_matrix_row(wX, j);
			w = gsl_vector_get(W, j);
			gsl_vector_scale(&X_row.vector, SQRT_EPS(w));
		}

		// I = (W^0.5 *X)' (W^0.5 *X) = X'WX
		gsl_blas_dsyrk(CblasLower, CblasTrans, 1, wX, 0, I);

		// diagonal and lower-triangular part of the matrix are used
		if(gsl_linalg_cholesky_decomp(I) == GSL_EDOM)
		{
			if(verbose) fprintf(stderr, "Cholesky decomposition failed on information matrix X'WX\n");

			// clear up all results
			gsl_vector_set_zero(ls->beta);
			gsl_vector_set_zero(ls->stderr);
			gsl_vector_set_zero(ls->z);
			gsl_vector_set_all(ls->pvalue, 1);

			return REG_FAIL;
		}

		// we will change I to I^-1, so copy the Cholesky L
		gsl_matrix_memcpy(L, I);

		// compute the inverse of information matrix, now I is I^-1
		gsl_linalg_cholesky_invert(I);

		// Correct on delta, but still use uncorrected I^-1 as first order approximation
		if(correction)
		{
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, wX, I, 0, H_t);

			// get diagonal elements Hii
			for(j=0;j<n;j++)
			{
				gsl_vector_view H_t_r = gsl_matrix_row(H_t, j);
				gsl_vector_view wX_r = gsl_matrix_row(wX, j);

				gsl_blas_ddot(&H_t_r.vector, &wX_r.vector, &w);
				gsl_vector_set(H, j, w);
			}

			// P - 0.5. P is changed from now
			gsl_vector_add_constant (P, -0.5);

			// H*(P - 0.5)
			gsl_vector_mul(H, P);

			// correct the error vector
			gsl_vector_sub(error, H);
		}

		//gradient U = X'error
		gsl_blas_dgemv(CblasTrans, 1, X, error, 0, U);

		// stop criterion by |U| almost 0
		status = gsl_multiroot_test_residual(U, tol);
		if(status == GSL_SUCCESS) break;

		// next round
		// Newton Raphson delta: I^-1 * U
		gsl_linalg_cholesky_solve(L, U, delta);

		norm_delta = gsl_blas_dnrm2(delta);
		norm_beta = gsl_blas_dnrm2(ls->beta);
		step_ratio = (norm_delta + 1)/(norm_beta + 1);

		if (step_ratio > max_delta)
		{
			step_ratio = max_delta/step_ratio;
			gsl_vector_scale(delta, step_ratio);
		}

		// beta = beta + delta
		gsl_vector_add(ls->beta, delta);
	}

	if(i==maxIter && status != GSL_SUCCESS)
	{
		if(verbose) fprintf(stderr, "Exceed maximum number of iterations %lu norm U = %lf\n", maxIter, gsl_blas_dnrm2(U));
		returnflag = CONVERGE_FAIL;
	}else{
		returnflag = (int)i;	// return number of iterations
	}

	// calculate statistic measures for inference

	// take diagonal elements of information matrix as variance
	gsl_vector_view Irev_D = gsl_matrix_diagonal(I);
	gsl_vector_memcpy(ls->stderr, &Irev_D.vector);

	for(i=0 ; i<p ; i++)
	{
		w = SQRT_EPS(gsl_vector_get(ls->stderr, i));
		gsl_vector_set(ls->stderr,i,w);

		if(w>0){
			w = gsl_vector_get(ls->beta, i)/w;
			gsl_vector_set(ls->z,i,w);

			// Wald test: Pr(X>|z|)
			w = 2*(1-gsl_cdf_gaussian_P(fabs(w),1));
			gsl_vector_set(ls->pvalue,i,w);

		}else{
			if(verbose) fprintf(stderr, "standard error of beta is zero\n");

			gsl_vector_set(ls->z,i,0);
			gsl_vector_set(ls->pvalue,i,1);
		}
	}

	return returnflag;
}
