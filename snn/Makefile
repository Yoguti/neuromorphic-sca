# snn/Makefile

CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -g -O2 -fopenmp \
          -D_POSIX_C_SOURCE=200809L \
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

ENGINE_SRC  = main.c $(COMMON_SRC)
TEST_SRC    = tests/testbenches.c $(COMMON_SRC)

PREDICT_SRC = tests/snn_predict.c \
              arena_impl.c \
              network/network.c \
              network/neuron.c \
              EONS/engine.c \
              EONS/export.c

ENGINE_TARGET  = snn_engine
TEST_TARGET    = snn_tests
PREDICT_TARGET = snn_predict

.PHONY: all engine test predict clean run run-test

all: engine test predict

engine: $(ENGINE_TARGET)

test: $(TEST_TARGET)

predict: $(PREDICT_TARGET)

$(ENGINE_TARGET): main.c $(COMMON_SRC)
	$(CC) $(CFLAGS) -o $@ main.c $(COMMON_SRC) $(LDLIBS)

$(TEST_TARGET): tests/test_main.c $(TEST_SRC)
	$(CC) $(CFLAGS) -o $@ tests/test_main.c $(TEST_SRC) $(LDLIBS)

$(PREDICT_TARGET): $(PREDICT_SRC)
	$(CC) $(CFLAGS) -o $@ $(PREDICT_SRC) -lm

run: engine
	./$(ENGINE_TARGET)

run-test: test
	./$(TEST_TARGET)

clean:
	rm -f $(ENGINE_TARGET) $(TEST_TARGET) $(PREDICT_TARGET)
	rm -f best_network_neurons.csv best_network_synapses.csv