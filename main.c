/*
 * main.c
 * Driver for demonstration of parallelized matrix multiplication.
 * Author: Amittai Aviram - aviram@bc.edu
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "matrix_mult.h"

typedef struct RunArgs {
    multiply_function func;
    double * product;
    const int num_workers;
    const char * const name;
    const bool verify;
} RunArgs;


int main() {
    int size = DIM * DIM;
    double * matrix_a = calloc(size, sizeof(double));
    double * matrix_b = calloc(size, sizeof(double));
    init_matrix(matrix_a, DIM);
    init_matrix(matrix_b, DIM);
    RunArgs args[] = {
        {multiply_serial, NULL, 1, "serial", false},
        {multiply_parallel_processes, NULL, NUM_WORKERS, "parallel processes", true},
        {multiply_parallel_threads, NULL, NUM_WORKERS, "parallel threads", true}
    };
    const int num_functions = sizeof(args) / sizeof(args[0]);
    for (int i = 0; i < num_functions; ++i) {
       args[i].product = calloc(size, sizeof(double));
    }
    for (int i = 0; i < num_functions; ++i) {
        run_and_time(
                args[i].func,
                matrix_a,
                matrix_b,
                args[i].product,
                args[0].product,
                DIM,
                args[i].name,
                args[i].num_workers,
                args[i].verify
                );
    }
    for (int i = 0; i < num_functions; ++i) {
        free(args[i].product);
    }
    return EXIT_SUCCESS;
}

