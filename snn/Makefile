# snn/Makefile

CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -g -O2 -fopenmp \
          -I. \
          -Inetwork \
          -Ilibs \
          -IEONS \
          -Itests
LDLIBS  = -lhdf5 -lm -fopenmp

COMMON_SRC = arena_impl.c \
             network/network.c \
             network/neuron.c \
             EONS/engine.c \
             EONS/evolution.c \
             EONS/export.c \
             libs/dataset.c

ENGINE_SRC = main.c $(COMMON_SRC)
TEST_SRC   = tests/testbenches.c $(COMMON_SRC)

ENGINE_TARGET = snn_engine
TEST_TARGET   = snn_tests

.PHONY: all engine test clean run run-test

all: engine test

engine: $(ENGINE_TARGET)

test: $(TEST_TARGET)

$(ENGINE_TARGET): main.c $(COMMON_SRC)
	$(CC) $(CFLAGS) -o $@ main.c $(COMMON_SRC) $(LDLIBS)

$(TEST_TARGET): tests/test_main.c $(TEST_SRC)
	$(CC) $(CFLAGS) -o $@ tests/test_main.c $(TEST_SRC) $(LDLIBS)

run: engine
	./$(ENGINE_TARGET)

run-test: test
	./$(TEST_TARGET)

clean:
	rm -f $(ENGINE_TARGET) $(TEST_TARGET)
	rm -f best_network_neurons.csv best_network_synapses.csv
