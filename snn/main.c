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
    params.num_generations = 100; 


    char seed_names[4][128] = {
        "network-csvs/seed_1",
        "network-csvs/seed_2",
        "network-csvs/seed_3",
        "network-csvs/best_network"
    };
    
    snn_network_t *seeds[4] = {NULL, NULL, NULL, NULL};
    int num_seeds_loaded = 0;

    for (int i = 0; i < 4; i++) {
        seeds[i] = import_network_csv(engine_get_arena_a(), seed_names[i]);
        if (seeds[i]) num_seeds_loaded++;
    }

    if (num_seeds_loaded > 0) {
        printf("\n>>> SUCCESS: %d checkpoints found! Injecting seeds...\n\n", num_seeds_loaded);
        int clones_per_seed = (POPULATION_SIZE / 10) / num_seeds_loaded; 
        int pop_idx = 0;
        
        for (int i = 0; i < 4; i++) {
            if (seeds[i]) {
                for (int c = 0; c < clones_per_seed; c++) {
                    population[pop_idx].network = snn_clone_into(engine_get_arena_a(), seeds[i]);
                    pop_idx++;
                }
            }
        }
    } else {
        printf("\n>>> No previous seeds found in 'network-csvs/'. Starting from scratch.\n\n");
    }

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

        if (gen == 30 || gen == 60 || gen == 90) {
            int seed_idx = gen / 30; 
            char fname[128];
            snprintf(fname, sizeof(fname), "network-csvs/seed_%d", seed_idx);
            
            export_network_csv(population[best_idx].network, fname);
            printf("  [!] Checkpoint saved/overwritten successfully: %s\n", fname);
        }

        Arena *inactive_arena = engine_get_arena_b();
        arena_reset(inactive_arena);

        eons_do_epoch(population, next_generation, inactive_arena, &params, gen);

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

    export_network_csv(population[best_idx].network, "network-csvs/best_network");

    dataset_free(ds);
    return 0;
}