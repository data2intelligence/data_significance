#ifndef UTIL_H_
#define UTIL_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_myext
#include <arrayobject.h>

#include <stdio.h>
#include <gsl/gsl_matrix.h>

// convert numpy 2d matrix to gsl_matrix
gsl_matrix *PyArrayObject_to_gsl_matrix(PyArrayObject *x);
gsl_vector *PyArrayObject_to_gsl_vector(PyArrayObject *x);

// convert gsl_matrix to numpy 2d, new_memory flag for allocating separate memory space
PyArrayObject *gsl_matrix_to_PyArrayObject(gsl_matrix *x, const int new_memory);
PyArrayObject *gsl_vector_to_PyArrayObject(gsl_vector *x, const int new_memory);

//free the converted gsl_matrix, which is just a wrap without full memory allocation
void gsl_matrix_partial_free(gsl_matrix *x);
void gsl_vector_partial_free(gsl_vector *x);

void print_vector(const gsl_vector *v, const char *title, FILE *fp);
void print_matrix(const gsl_matrix *X, const char *title, FILE *fp);


// check if vector X is n, if equal_stride is not 0, also check stride == 1.
void check_vector_dimension( const gsl_vector *X, const size_t n,
		const char *title, const int equal_stride);

// check if matrix X is n*p, if equal_stride is not 0, also check stride == size2.
void check_matrix_dimension( const gsl_matrix *X, const size_t n, const size_t p,
		const char *title, const int equal_stride);


#endif /* UTIL_H_ */
