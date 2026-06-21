#include "engine.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#define ARENA_CAPACITY (1024 * 1024 * 32) // 32 Megabytes per arena

static Arena *arena_A = NULL;
static Arena *arena_B = NULL;
static Arena *active_arena = NULL;
static Arena *inactive_arena = NULL;

static candidate_t population[POPULATION_SIZE];

void engine_init(void) {
    arena_A = arena_init(ARENA_CAPACITY);
    arena_B = arena_init(ARENA_CAPACITY);

    if (!arena_A || !arena_B) {
        printf("Fatal Error: Failed to initialize Arenas.\n");
        return;
    }

    active_arena = arena_A;
    inactive_arena = arena_B;

    for (int i = 0; i < POPULATION_SIZE; i++) {
        snn_network_t *net = snn_create(active_arena);
        population[i].network = net;
        population[i].fitness_score = 0.0f;
    }
}

candidate_t* engine_get_population(void) { return population; }
Arena* engine_get_arena_a(void) { return active_arena; }
Arena* engine_get_arena_b(void) { return inactive_arena; }

void engine_swap_arenas(void) {
    Arena *temp = active_arena;
    active_arena = inactive_arena;
    inactive_arena = temp;
}

uint8_t evaluate_network(snn_network_t *net, const int8_t *trace, size_t trace_length) {
    snn_reset(net);
    
    uint32_t spike_counts[SNN_NUM_OUTPUTS] = {0};

    for (size_t t = 0; t < trace_length; t++) {
        int8_t sample = trace[t];

        net->input_spikes[UP_INPUT_ID]   = (sample == 1)  ? 1 : 0;
        net->input_spikes[DOWN_INPUT_ID] = (sample == -1) ? 1 : 0;

        snn_tick(net);

        for(uint16_t out = 0; out < SNN_NUM_OUTPUTS; out++) {
            if(net->nodes[out].has_fired) {
                spike_counts[out]++; 
            }
        }
    }

    uint8_t best_class = 0;
    uint32_t best_count = spike_counts[0];
    for (uint8_t c = 1; c < SNN_NUM_HW_CLASSES; c++) {
        if (spike_counts[c] > best_count) {
            best_count = spike_counts[c];
            best_class = c;
        }
    }

    // a silent network (no output ever fired) hasn't actually classified
    // anything; returning 0 by default tie-break would let it silently
    // collect credit whenever the true label happens to be 0. Signal "no
    // decision" with the sentinel 255 so the caller can penalize it instead.
    if (best_count == 0) {
        return 255;
    }

    return best_class;
}


void engine_evaluate_generation(candidate_t *pop, size_t population_size, const ascad_dataset_t *ds) {
    if (!pop || !ds || ds->num_traces == 0) return;

    const float w_fidelity  = 0.70f;
    const float w_diversity = 0.15f;
    const float w_utility   = 0.15f;

    const uint16_t THRESHOLD_SYNAPSES = 30;
    const uint16_t THRESHOLD_NEURONS  = 15;
    const float PENALTY_SYNAPSE = 0.002f;
    const float PENALTY_NEURON  = 0.005f;

    const float FIDELITY_THRESHOLD_FOR_PENALTY = 0.30f;

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < population_size; i++) {
        snn_network_t *net = pop[i].network;
        if (!net) {
            pop[i].fitness_score = 0.0f;
            continue;
        }

        uint32_t gt[SNN_NUM_HW_CLASSES]   = {0};
        uint32_t pred[SNN_NUM_HW_CLASSES] = {0};
        uint32_t tp[SNN_NUM_HW_CLASSES]   = {0};
        uint32_t err_sum[SNN_NUM_HW_CLASSES] = {0};
        uint32_t valid = 0;

        for (size_t t = 0; t < ds->num_traces; t++) {
            const uint8_t label = ds->labels[t];
            gt[label]++;

            const int8_t *trace = &ds->traces[t * ds->trace_length];
            const uint8_t out = evaluate_network(net, trace, ds->trace_length);

            if (out != 255) {
                pred[out]++;
                valid++;
                err_sum[label] += abs((int)label - (int)out);
                if (out == label) tp[label]++;
            } else {
                err_sum[label] += 8;
            }
        }


        if (valid == 0) {
            pop[i].fitness_score = 0.0f;
            continue;
        }

        float class_score_sum = 0.0f;
        int active = 0;
        for (int c = 0; c < SNN_NUM_HW_CLASSES; c++) {
            if (gt[c] > 0) {
                float recall = (float)tp[c] / (float)gt[c];
                float mae = (float)err_sum[c] / (float)gt[c];
                float dist_score = (8.0f - mae) / 8.0f;
                class_score_sum += 0.70f * recall + 0.30f * dist_score;
                active++;
            }
        }
        float fidelity = (active > 0) ? class_score_sum / active : 0.0f;

        // utility: fraction of traces the network actually rendered a decision on
        float utility = (float)valid / (float)ds->num_traces;

        // diversity: how many distinct classes the network is capable of producing
        int classes_usadas = 0;
        for (int c = 0; c < SNN_NUM_HW_CLASSES; c++)
            if (pred[c] > 0) classes_usadas++;
        float raw_div = (float)classes_usadas / (float)SNN_NUM_HW_CLASSES;
        float diversity = raw_div * utility;

        float penalty = 0.0f;
        if (fidelity > FIDELITY_THRESHOLD_FOR_PENALTY) {
            if (net->num_synapses > THRESHOLD_SYNAPSES)
                penalty += (net->num_synapses - THRESHOLD_SYNAPSES) * PENALTY_SYNAPSE;
            if (net->num_hidden > THRESHOLD_NEURONS)
                penalty += (net->num_hidden - THRESHOLD_NEURONS) * PENALTY_NEURON;
        }

        float raw = w_fidelity * fidelity
                  + w_diversity * diversity
                  + w_utility * utility
                  - penalty;

        pop[i].fitness_score = (raw > 0.0f) ? raw : 0.0f;
    }
}