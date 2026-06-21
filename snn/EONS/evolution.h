#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "engine.h"

typedef struct {
    float crossover_rate;
    float merge_rate;
    float mutation_rate;
    int   num_mutations;        // mutation count ceiling at generation 0 (start of annealing)
    int   num_mutations_min;    // mutation count floor reached by num_generations (end of annealing);
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

// generation is the current generation index (0-based); used to linearly
// anneal the mutation count from params->num_mutations down to
// params->num_mutations_min over params->num_generations, so early
// generations explore more broadly and later generations refine more
// conservatively.
void eons_do_epoch(
    candidate_t *current,
    candidate_t *next,
    Arena *next_arena,
    const eons_params_t *params,
    int generation
);

#endif // EVOLUTION_H
