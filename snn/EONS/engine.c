#include "engine.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define ARENA_CAPACITY (1024 * 1024 * 64)
#define INFERENCE_TICKS 40
#define READOUT_ALPHA   1.0f

#define EVAL_BATCH_SIZE 2000
#define SPIKE_PENALTY_WEIGHT 0.0001f

static Arena *arena_A      = NULL;
static Arena *arena_B      = NULL;
static Arena *active_arena   = NULL;
static Arena *inactive_arena = NULL;
static candidate_t population[POPULATION_SIZE];

void engine_init(void) {
    arena_A = arena_init(ARENA_CAPACITY);
    arena_B = arena_init(ARENA_CAPACITY);
    if (!arena_A || !arena_B) { printf("Fatal: arena init failed\n"); return; }
    active_arena   = arena_A;
    inactive_arena = arena_B;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        population[i].network       = snn_create(active_arena);
        population[i].fitness_score = 0.0f;
    }
}

candidate_t* engine_get_population(void) { return population; }
Arena* engine_get_arena_a(void)          { return active_arena; }
Arena* engine_get_arena_b(void)          { return inactive_arena; }

void engine_swap_arenas(void) {
    Arena *tmp   = active_arena;
    active_arena   = inactive_arena;
    inactive_arena = tmp;
}

static void run_inference(snn_network_t *net, const float *trace, size_t trace_length,
                          uint32_t spike_counts[SNN_NUM_OUTPUTS]) {
    snn_reset(net);
    for (int c = 0; c < SNN_NUM_OUTPUTS; c++) spike_counts[c] = 0;

    size_t n = trace_length < SNN_NUM_POIS ? trace_length : SNN_NUM_POIS;

    for (int tick = 0; tick < INFERENCE_TICKS; tick++) {
        for (size_t p = 0; p < n; p++)
            net->input_values[p] = trace[p];
        snn_tick(net);
        for (uint16_t o = 0; o < SNN_NUM_OUTPUTS; o++)
            if (net->nodes[o].has_fired) spike_counts[o]++;
    }
}

uint8_t evaluate_network(snn_network_t *net, const float *trace, size_t trace_length) {
    uint32_t counts[SNN_NUM_OUTPUTS];
    run_inference(net, trace, trace_length, counts);
    uint8_t  best = 0; uint32_t best_c = counts[0], total = counts[0];
    for (uint8_t c = 1; c < SNN_NUM_HW_CLASSES; c++) {
        total += counts[c];
        if (counts[c] > best_c) { best_c = counts[c]; best = c; }
    }
    return (total == 0) ? 255 : best;
}

void snn_predict_proba(snn_network_t *net, const float *trace, size_t trace_length,
                       float out_probs[SNN_NUM_OUTPUTS]) {
    uint32_t counts[SNN_NUM_OUTPUTS];
    run_inference(net, trace, trace_length, counts);
    uint32_t total = 0;
    for (int c = 0; c < SNN_NUM_OUTPUTS; c++) total += counts[c];
    float denom = (float)total + (float)SNN_NUM_OUTPUTS * READOUT_ALPHA;
    for (int c = 0; c < SNN_NUM_OUTPUTS; c++)
        out_probs[c] = ((float)counts[c] + READOUT_ALPHA) / denom;
}

void engine_evaluate_generation(candidate_t *pop, size_t population_size,
                                const ascad_dataset_t *ds, float alpha) {
    if (!pop || !ds || ds->num_traces == 0) return;

    size_t batch = ds->num_traces < EVAL_BATCH_SIZE ? ds->num_traces : EVAL_BATCH_SIZE;

    size_t *indices = malloc(ds->num_traces * sizeof(size_t));
    for (size_t i = 0; i < ds->num_traces; i++) indices[i] = i;
    for (size_t i = ds->num_traces - 1; i > 0; i--) {
        size_t j = (size_t)rand() % (i + 1);
        size_t tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < population_size; i++) {
        snn_network_t *net = pop[i].network;
        if (!net) { pop[i].fitness_score = 0.0f; continue; }
        
        uint32_t correct = 0;
        uint64_t total_spikes_all_batches = 0;

        for (size_t b = 0; b < batch; b++) {
            size_t t = indices[b];
            const float *trace = &ds->traces[t * ds->trace_length];
            
            uint32_t counts[SNN_NUM_OUTPUTS];
            run_inference(net, trace, ds->trace_length, counts);
            
            for (int o = 0; o < SNN_NUM_OUTPUTS; o++) {
                total_spikes_all_batches += counts[o];
            }

            uint8_t best = 0; uint32_t best_c = counts[0];
            for (uint8_t c = 1; c < SNN_NUM_HW_CLASSES; c++) {
                if (counts[c] > best_c) { best_c = counts[c]; best = c; }
            }
            if (best == ds->labels[t]) correct++;
        }
        
        float accuracy      = (float)correct / (float)batch;
        float size_penalty  = (float)(net->num_synapses + SNN_LIF_COUNT(net)) * alpha;
        float spike_penalty = (float)total_spikes_all_batches * SPIKE_PENALTY_WEIGHT;
        
        pop[i].fitness_score = accuracy - size_penalty - spike_penalty;
    }

    free(indices);
}
