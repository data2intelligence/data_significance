#include "util.h"

void print_matrix(const gsl_matrix *X, const char *title, FILE *fp)
{
	size_t i,j;

	if(title != NULL) fprintf(fp,"Matrix %s:\n", title);

	for(i=0;i<X->size1;i++)
	{
		for(j=0;j<X->size2;j++)
		{
			fprintf(fp, "%f%c", gsl_matrix_get(X,i,j), (j<X->size2-1?'\t':'\n'));
		}
	}
}

void print_vector(const gsl_vector *X, const char *title, FILE *fp)
{
	size_t i;

	if(title != NULL) fprintf(fp,"Vector %s:\n", title);

	for(i=0;i<X->size;i++)
	{
		fprintf(fp, "%f%c", gsl_vector_get(X,i), (i<X->size-1?'\t':'\n'));
	}
}

gsl_matrix *PyArrayObject_to_gsl_matrix(PyArrayObject *x)
{
	gsl_block *b;
	gsl_matrix *r;

	if(x->nd != 2 || x->descr->type_num != NPY_DOUBLE)
	{
		PyErr_SetString(PyExc_ValueError, "Cannot convert non 2D matrix to gsl_matrix");
		return NULL;
	}

	b = (gsl_block*)malloc(sizeof(gsl_block));
	r = (gsl_matrix*)malloc(sizeof(gsl_matrix));

	r->size1 = x->dimensions[0];
	r->tda = r->size2 = x->dimensions[1];
	r->owner = 1;
	b->data = r->data = (double*)x->data;
	r->block = b;
	b->size = r->size1 * r->size2;

	return r;
}


gsl_vector *PyArrayObject_to_gsl_vector(PyArrayObject *x)
{
	gsl_block *b;
	gsl_vector *r;

	if(x->nd != 1 || x->descr->type_num != NPY_DOUBLE)
	{
		PyErr_SetString(PyExc_ValueError, "Cannot convert non 1D vector to gsl_vector");
		return NULL;
	}

	b = (gsl_block*)malloc(sizeof(gsl_block));
	r = (gsl_vector*)malloc(sizeof(gsl_vector));

	r->size = x->dimensions[0];
	r->stride = 1;
	r->owner = 1;
	b->data = r->data = (double*)x->data;
	r->block = b;
	b->size = r->size;

	return r;
}


PyArrayObject *gsl_matrix_to_PyArrayObject(gsl_matrix *x, const int new_memory)
{
	npy_intp dimensions[2] = {x->size1, x->size2};
	PyArrayObject *ptr;

	if(new_memory){
		ptr = (PyArrayObject*)PyArray_SimpleNew(2, dimensions, NPY_DOUBLE);
		memcpy(ptr->data, x->data, x->block->size * sizeof(double));
	}else{
		ptr = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimensions, NPY_DOUBLE, (void*)x->data);
	}

	return (PyArrayObject*)PyArray_Return(ptr);
}


PyArrayObject *gsl_vector_to_PyArrayObject(gsl_vector *x, const int new_memory)
{
	npy_intp dimensions[1] = {x->size};
	PyArrayObject *ptr;

	if(new_memory){
		ptr = (PyArrayObject*)PyArray_SimpleNew(1, dimensions, NPY_DOUBLE);
		memcpy(ptr->data, x->data, x->block->size * sizeof(double));
	}else{
		ptr = (PyArrayObject*)PyArray_SimpleNewFromData(1, dimensions, NPY_DOUBLE, (void*)x->data);
	}

	return (PyArrayObject*)PyArray_Return(ptr);
}


void gsl_matrix_partial_free(gsl_matrix *x)
{
	// data fields is not my own
	free(x->block);
	free(x);
}

void gsl_vector_partial_free(gsl_vector *x)
{
	// data fields is not my own
	free(x->block);
	free(x);
}

void check_vector_dimension( const gsl_vector *X, const size_t n, const char *title, const int equal_stride)
{
	if(X->size != n)
	{
		fprintf(stderr, "Vector %s length is %lu but not expected %lu.\n", title, X->size, n);
		exit(1);
	}

	if(equal_stride != 0 && X->stride != 1)
	{
		fprintf(stderr, "Vector %s stride %lu is not equal to 1.\n", title, X->stride);
		exit(1);
	}
}

void check_matrix_dimension(const gsl_matrix *X, const size_t n, const size_t p,
		const char *title, const int equal_stride)
{
	if(X->size1 != n || X->size2 != p)
	{
		fprintf(stderr, "Matrix %s dimension is %lu * %lu but not expected %lu * %lu.\n", title, X->size1, X->size2, n, p);
		exit(1);
	}

	if(equal_stride != 0 && X->size2 != X->tda)
	{
		fprintf(stderr, "Matrix %s tda %lu is not equal to column %lu.\n", title, X->tda, X->size2);
		exit(1);
	}
}
