#include "../libs/arena.h"
#include "../network/neuron.h"
#include "../network/network.h"
#include "../EONS/export.h"
#include "../EONS/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARENA_SIZE (8 * 1024 * 1024)  /* 8 MB — plenty for one network */

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <network_basename> <trace_length> [traces_csv]\n"
            "  network_basename : path without extension, e.g. network-csvs/best_network\n"
            "  trace_length     : number of samples per trace\n"
            "  traces_csv       : optional file; reads stdin if omitted\n",
            argv[0]);
        return 1;
    }

    const char *basename     = argv[1];
    size_t      trace_length = (size_t)atoi(argv[2]);
    if (trace_length == 0) {
        fprintf(stderr, "Error: trace_length must be > 0\n");
        return 1;
    }

    FILE *fin = stdin;
    if (argc >= 4) {
        fin = fopen(argv[3], "r");
        if (!fin) {
            fprintf(stderr, "Error: cannot open traces file '%s'\n", argv[3]);
            return 1;
        }
    }

    Arena *arena = arena_init(ARENA_SIZE);
    if (!arena) {
        fprintf(stderr, "Error: arena allocation failed\n");
        return 1;
    }

    snn_network_t *net = import_network_csv(arena, basename);
    if (!net) {
        fprintf(stderr,
            "Error: could not load network from '%s_neurons.csv' / '%s_synapses.csv'\n",
            basename, basename);
        arena_free(arena);
        return 1;
    }

    fprintf(stderr, "Loaded: %u hidden neurons, %u synapses, trace_length=%zu\n",
            net->num_hidden, net->num_synapses, trace_length);

    int8_t *trace_buf = malloc(trace_length * sizeof(int8_t));
    if (!trace_buf) {
        fprintf(stderr, "Error: could not allocate trace buffer\n");
        arena_free(arena);
        return 1;
    }

    char  *line    = NULL;
    size_t line_sz = 0;
    size_t n_done  = 0;

    while (getline(&line, &line_sz, fin) != -1) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '#') continue;

        size_t idx  = 0;
        char  *ptr  = line;
        char  *endp = NULL;

        while (idx < trace_length) {
            long val = strtol(ptr, &endp, 10);
            if (endp == ptr) break;   /* no more numbers */
            trace_buf[idx++] = (int8_t)val;
            ptr = endp;
            if (*ptr == ',') ptr++;   /* skip comma separator */
        }

        if (idx < trace_length) {
            /* incomplete trace — skip silently */
            continue;
        }

        uint8_t pred = evaluate_network(net, trace_buf, trace_length);
        printf("%u\n", (unsigned)pred);
        n_done++;
    }

    fprintf(stderr, "Predicted %zu traces.\n", n_done);

    free(line);
    free(trace_buf);
    if (fin != stdin) fclose(fin);
    arena_free(arena);
    return 0;
}