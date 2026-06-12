# snn/Makefile

CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -g \
           -I. \
           -Inetwork \
           -Ilibs

# arena_impl.c owns the single ARENA_IMPLEMENTATION define
SRC = main.c \
      arena_impl.c \
      network/network.c \
      network/neuron.c

TARGET = snn_debug

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)