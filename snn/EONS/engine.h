#ifndef ENGINE_H
#define ENGINE_H

#include "network/network.h"
#include "libs/dataset.h"

#define POPULATION_SIZE 500
#define SNN_NUM_HW_CLASSES 9

typedef struct {
    snn_network_t *network;
    float fitness_score;
} candidate_t;

void engine_init(void);

uint8_t evaluate_network(snn_network_t *net, const int8_t *trace, size_t trace_length);
void engine_evaluate_generation(candidate_t *pop, size_t population_size, const ascad_dataset_t *ds);

candidate_t* engine_get_population(void);
Arena* engine_get_arena_a(void);
Arena* engine_get_arena_b(void);
void engine_swap_arenas(void);

#endif // ENGINE_H