FLAGS = -Wall -Werror -std=gnu99
DEP = main.c matrix_mult.c

all: matrix_mult

matrix_mult: $(DEP)
	gcc $(FLAGS) $^ -o $@ -lpthread

clean:
	rm -f matrix_mult
