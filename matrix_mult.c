/*
 * Authors: Brian Laboissonniere - laboissb@bc.edu, Adam Laboissonniere - laboissa@bc.edu
*/
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>

#include "matrix_mult.h"

void * mmap_checked(size_t size) {
    void * mem = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0);
    if (mem == MAP_FAILED) {
        perror("Demand-zero memory allocation failure");
        exit(EXIT_FAILURE);
    }
    return mem;
}

void munmap_checked(void * mapping, size_t size) {
    int ret = munmap(mapping, size);
    if (ret == -1) {
        perror("Munmap failed to run");
        exit(EXIT_FAILURE);
    }
}

pid_t fork_checked() {
    pid_t ret = fork();
    if (ret == FAILURE) {
        perror("Failed to call fork");
        exit(EXIT_FAILURE);
    }
    return ret;
}

void init_matrix(double * matrix, int dim) {
    int k = 1;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            matrix[i * dim + j] = k++;
        }
    }
}

void multiply_chunk(const double * const a, const double * const b, double * const c, const int dim, int row_start, int chunk) {
    for (int i = row_start; i < row_start + chunk; ++i) {
        for (int j = 0; j < dim; ++j) {
            double sum = 0;
            for (int k = 0; k < dim; ++k) {
                sum += a[i * dim + k] * b[k * dim + j];
            }
            c[i * dim + j] = sum;
        }
    }    
}

void multiply_serial(const double * const a, const double * const b, double * const c, const int dim, const int num_workers) {
    multiply_chunk(a, b, c, dim, 0, dim);
}

void multiply_parallel_processes(const double * const a, const double * const b, double * c, const int dim, const int num_workers) {
    double * c_shared = mmap_checked(dim * dim * sizeof(double));
    int num_procs = num_workers - 1;
    int chunk_size = dim / num_workers;
    int row_start = 0;
    for (int i = 0; i < num_procs; ++i) {
        pid_t pid = fork_checked();
        if (!pid) {
            multiply_chunk(a, b, c_shared, dim, row_start, chunk_size);
            exit(EXIT_SUCCESS);
        }
        row_start += chunk_size;
    }
    while (wait(NULL) > 0);
    multiply_chunk(a, b, c_shared, dim, row_start, dim - row_start);
    memcpy(c, c_shared, dim * dim * sizeof(double));
    munmap_checked(c_shared, dim * dim * sizeof(double));
}

void * task(void * arg) {
    Args * args = (Args *)arg;
    multiply_chunk(args->a, args->b, args->c, args->dim, args->row_start, args->chunk);
    return EXIT_SUCCESS;
}

void multiply_parallel_threads(const double * const a, const double * const b, double * c, const int dim, const int num_workers) {
    int num_threads = num_workers - 1;
    int chunk = dim / num_workers;
    Args * arg_set = malloc(num_workers * sizeof(Args));
    int row_start = 0;
    pthread_t thread_set[num_workers];
    for (int i = 0; i < num_workers; ++i) {
        arg_set[i].a = a;
        arg_set[i].b = b;
        arg_set[i].c = c;
        arg_set[i].dim = dim;
        arg_set[i].row_start = row_start;
        arg_set[i].chunk = (i == num_workers - 1) ? dim - row_start : chunk;
        row_start += chunk;
    }
    int id = 0;
    for (; id < num_threads; ++id) {
        pthread_create(&thread_set[id], NULL, task, &arg_set[id]);
    }
    chunk = dim - arg_set[id].row_start;
    task(&arg_set[id]);
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(thread_set[i], NULL);
    }
    free(arg_set);
}

int verify(const double * const m1, const double * const m2, const int dim) {
    for (int i = 0; i < dim * dim; i++) {
        if (m1[i] != m2[i]) {
            return(FAILURE);
        }
    }
    return SUCCESS;
}

void print_matrix(double * matrix, int dim) {
    for (int i = 0; i < dim * dim; ++i) {
        printf("%0.0f ", matrix[i]);
        if (((i + 1) % dim) == 0) {
            putchar('\n');
        }
    }   
    putchar('\n');
}

struct timeval time_diff(struct timeval * start, struct timeval * end) {
    end->tv_sec -= start->tv_sec;
    end->tv_usec -= start->tv_usec;
    if (end->tv_usec < 0) {
        end->tv_sec -= 1;
        end->tv_usec += 1000000;
    }
    return *end;
}

void print_elapsed_time(struct timeval * start, struct timeval * end, const char * const name) {
    struct timeval time = time_diff(start, end);
    if (time.tv_sec == 1) {
        printf("Time elapsed for %s: %ld second and %ld microseconds.\n", name, time.tv_sec, time.tv_usec);
    } else {
    printf("Time elapsed for %s: %ld seconds and %ld mircoseconds.\n", name, time.tv_sec, time.tv_usec);
    }
}

void print_verification(const double  * const m1, const double * const m2, const int dim, const char * const name) {
    int ver = verify(m1, m2, dim);
    if (ver == SUCCESS) {
        printf("Verification for %s: success.\n", name);
    } else {
        printf("Verification for %s: failure.\n", name);
    }
}

void run_and_time(
        multiply_function multiply_matrices,
        const double * const a,
        const double * const b,
        double * const c,
        const double * const gold,
        const int dim,
        const char * const name,
        const int num_workers,
        const bool verify
        ) {
    struct timeval start;
    struct timeval end;
    if (num_workers == 1) {
        printf("Algorithm: %s with %d worker.\n", name, num_workers);
    } else {
        printf("Algorithm: %s with %d workers.\n", name, num_workers);
    }
    gettimeofday(&start, 0);
    multiply_matrices(a, b, c, dim, num_workers);
    gettimeofday(&end, 0);
    print_elapsed_time(&start, &end, name);
    if (verify) {
        print_verification(c, gold, dim, name);
    }
}
