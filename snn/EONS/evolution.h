#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "engine.h"

typedef struct {
    float crossover_rate;
    float merge_rate;
    float mutation_rate;
    int   num_mutations;
    float add_node_rate;
    float delete_node_rate;
    float add_edge_rate;
    float delete_edge_rate;
    float node_param_rate;
    float edge_param_rate;
    int   tournament_size;
    float tournament_p;
    float random_factor;
    int   num_best;
    int   population_size;
    int   num_generations;
} eons_params_t;

eons_params_t eons_default_params(void);

void eons_do_epoch(
    candidate_t *current,
    candidate_t *next,
    Arena *next_arena,
    const eons_params_t *params
);

#endif // EVOLUTION_H