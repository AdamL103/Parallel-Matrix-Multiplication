/*
 * Authors: Brian Laboissonniere - laboissb@bc.edu, Adam Laboissonniere - laboissa@bc.edu
*/
#include <stdbool.h>

#define DIM 1024
#define NUM_WORKERS 4
#define SUCCESS 0
#define FAILURE -1

typedef void (* multiply_function)(const double * const, const double * const, double * const, const int, const int);

typedef struct Args {
    const double * a;
    const double * b;
    double * c;
    int dim;
    int row_start;
    int chunk;
} Args;

void init_matrix(double *, int);
void print_matrix(double *, int);
void multiply_serial(const double * const, const double * const, double * const, const int, const int);
void multiply_parallel_processes(const double * const, const double * const, double * const, const int, const int);
void multiply_parallel_threads(const double * const, const double * const, double * const, const int, const int);
void run_and_time(
        multiply_function,
        const double * const,
        const double * const,
        double * const,
        const double * const,
        const int,
        const char * const,
        const int,
        const bool);
