#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "EONS/engine.h"
#include "EONS/evolution.h"
#include "EONS/export.h"
#include "libs/dataset.h"

int main(int argc, char *argv[]) {
    int main_gen = 0;
    if (argc > 1) {
        main_gen = atoi(argv[1]);
        if (main_gen < 0) main_gen = 0;
    }
    srand((unsigned)time(NULL));
    printf("Neuromorphic SCA: EONS on AES_HD\n");
    printf("Running for %d generations\n", main_gen > 0 ? main_gen : 100);

    const char *h5_path = "../modulated_dataset/aes_hd_snn_ready.h5";

    ascad_dataset_t *ds = dataset_load(h5_path, 0, false);
    if (!ds) { printf("Error: Could not load dataset.\n"); return 1; }

    printf("Dataset: %zu traces, %zu POIs\n", ds->num_traces, ds->trace_length);

    if (ds->trace_length != SNN_NUM_POIS) {
        printf("FATAL: trace_length=%zu != SNN_NUM_POIS=%d\n", ds->trace_length, SNN_NUM_POIS);
        dataset_free(ds);
        return 1;
    }

    engine_init();

    candidate_t   *population = engine_get_population();
    eons_params_t  params     = eons_default_params();
    params.num_generations    = main_gen > 0 ? main_gen : 100;

    char seed_names[4][128] = {
        "network-csvs/seed_1", "network-csvs/seed_2",
        "network-csvs/seed_3", "network-csvs/best_network"
    };
    snn_network_t *seeds[4] = {0};
    int num_seeds = 0;
    for (int i = 0; i < 4; i++) {
        seeds[i] = import_network_csv(engine_get_arena_a(), seed_names[i]);
        if (seeds[i]) num_seeds++;
    }
    if (num_seeds > 0) {
        int clones = (POPULATION_SIZE / 10) / num_seeds;
        int idx = 0;
        for (int i = 0; i < 4; i++)
            if (seeds[i])
                for (int c = 0; c < clones && idx < POPULATION_SIZE; c++, idx++)
                    population[idx].network = snn_clone_into(engine_get_arena_a(), seeds[i]);
        printf("Injected %d seeds\n", num_seeds);
    }

    candidate_t next_generation[POPULATION_SIZE];
    int best_idx = 0;

    for (int gen = 0; gen < params.num_generations; gen++) {
        engine_evaluate_generation(population, POPULATION_SIZE, ds, 1e-4f);

        float best_fit = -1e9f;
        for (int i = 0; i < POPULATION_SIZE; i++)
            if (population[i].fitness_score > best_fit) {
                best_fit = population[i].fitness_score;
                best_idx = i;
            }

        printf("Gen %3d | Fit: %.4f | H:%u S:%u\n",
               gen, best_fit,
               population[best_idx].network->num_hidden,
               population[best_idx].network->num_synapses);

        if (gen > 0 && gen % 50 == 0) {
            char fn[128];
            snprintf(fn, sizeof(fn), "network-csvs/seed_%d", gen / 50);
            export_network_csv(population[best_idx].network, fn);
        }

        Arena *ia = engine_get_arena_b();
        arena_reset(ia);
        eons_do_epoch(population, next_generation, ia, &params, gen);
        for (int i = 0; i < POPULATION_SIZE; i++) population[i] = next_generation[i];
        engine_swap_arenas();
    }

    engine_evaluate_generation(population, POPULATION_SIZE, ds, 1e-4f);
    best_idx = 0; float fb = -1e9f;
    for (int i = 0; i < POPULATION_SIZE; i++)
        if (population[i].fitness_score > fb) { fb = population[i].fitness_score; best_idx = i; }

    printf("Final: %.4f (H:%u S:%u)\n", fb,
           population[best_idx].network->num_hidden, population[best_idx].network->num_synapses);
    export_network_csv(population[best_idx].network, "network-csvs/best_network");
    dataset_free(ds);
    return 0;
}
