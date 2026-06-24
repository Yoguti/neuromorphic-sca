CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -g -O2 -fopenmp \
          -D_POSIX_C_SOURCE=200809L \
          -I. \
          -Inetwork \
          -Ilibs \
          -IEONS

LDLIBS  = -lhdf5 -lm -fopenmp

COMMON_SRC = arena_impl.c \
             network/network.c \
             network/neuron.c \
             EONS/engine.c \
             EONS/evolution.c \
             EONS/export.c \
             libs/dataset.c

ENGINE_TARGET = snn_engine

.PHONY: all engine clean run

all: engine

engine: $(ENGINE_TARGET)

$(ENGINE_TARGET): main.c $(COMMON_SRC)
	$(CC) $(CFLAGS) -o $@ main.c $(COMMON_SRC) $(LDLIBS)

run: engine
	./$(ENGINE_TARGET)

clean:
	rm -f $(ENGINE_TARGET)
	rm -f best_network_neurons.csv best_network_synapses.csv