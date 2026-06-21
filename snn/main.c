#include <stdio.h>
#include <stdlib.h>
#include "EONS/engine.h"
#include "EONS/evolution.h"
#include "EONS/export.h"
#include "libs/dataset.h"
#include "tests/testbenches.h"
#include <time.h>


int main(void) {
    srand(time(NULL));
    printf("Neuromorphic SCA: EONS Training\n");

    const char *h5_path = "../ATMEGA-AES-ASCAD_databases/ascad_modulated.h5";
    size_t num_traces_to_load = 6000;

    printf("Loading dataset from %s...\n", h5_path);

    ascad_dataset_t *ds = dataset_load(h5_path, num_traces_to_load);
    if (!ds) {
        printf("Error: Could not load dataset.\n");
        return 1;
    }

    printf("Dataset loaded! Traces: %zu, Length: %zu\n", ds->num_traces, ds->trace_length);

    printf("Initializing engine...\n");
    engine_init();

    candidate_t *population = engine_get_population();
    eons_params_t params = eons_default_params();
    params.num_generations = 500;

    candidate_t next_generation[POPULATION_SIZE];

    int best_idx = 0;

    for (int gen = 0; gen < params.num_generations; gen++) {

        engine_evaluate_generation(population, POPULATION_SIZE, ds);

        float best_fitness = -9999.0f;
        for (int i = 0; i < POPULATION_SIZE; i++) {
            if (population[i].fitness_score > best_fitness) {
                best_fitness = population[i].fitness_score;
                best_idx = i;
            }
        }
        printf("Generation %d | Best Fitness: %f | Neurons (hidden): %u | Synapses: %u\n",
               gen, best_fitness,
               population[best_idx].network->num_hidden,
               population[best_idx].network->num_synapses);

        Arena *inactive_arena = engine_get_arena_b();
        arena_reset(inactive_arena);

        eons_do_epoch(population, next_generation, inactive_arena, &params);

        for (int i = 0; i < POPULATION_SIZE; i++) {
            population[i] = next_generation[i];
        }
        engine_swap_arenas();
    }

    printf("\nEvolution complete!\n");

    engine_evaluate_generation(population, POPULATION_SIZE, ds);
    best_idx = 0;
    float final_best = -9999.0f;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (population[i].fitness_score > final_best) {
            final_best = population[i].fitness_score;
            best_idx = i;
        }
    }
    printf("Final best fitness: %f (neurons=%u, synapses=%u)\n",
           final_best,
           population[best_idx].network->num_hidden,
           population[best_idx].network->num_synapses);

    export_network_csv(population[best_idx].network, "best_network");

    dataset_free(ds);
    return 0;
}
