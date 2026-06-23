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

    if (best_count == 0) {
        return 255;
    }

    return best_class;
}


void engine_evaluate_generation(candidate_t *pop, size_t population_size, const ascad_dataset_t *ds) {
    if (!pop || !ds || ds->num_traces == 0) return;

    const float w_accuracy  = 0.60f;
    const float w_diversity = 0.20f;
    const float w_utility   = 0.20f;

    const uint16_t THRESHOLD_SYNAPSES = 30;
    const uint16_t THRESHOLD_NEURONS  = 15;
    const float PENALTY_SYNAPSE = 0.008f; 
    const float PENALTY_NEURON  = 0.020f;

    // Maximum entropy for 9 classes = log2(9) ≈ 3.17
    const float MAX_ENTROPY = logf((float)SNN_NUM_HW_CLASSES);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < population_size; i++) {
        snn_network_t *net = pop[i].network;
        if (!net) {
            pop[i].fitness_score = 0.0f;
            continue;
        }

        uint32_t gt[SNN_NUM_HW_CLASSES]   = {0};  // ground truth count per class
        uint32_t pred[SNN_NUM_HW_CLASSES] = {0};  // prediction count per class
        uint32_t tp[SNN_NUM_HW_CLASSES]   = {0};  // true positives per class
        uint32_t valid = 0;

        for (size_t t = 0; t < ds->num_traces; t++) {
            const uint8_t label = ds->labels[t];
            gt[label]++;

            const int8_t *trace = &ds->traces[t * ds->trace_length];
            const uint8_t out = evaluate_network(net, trace, ds->trace_length);

            if (out != 255) {
                pred[out]++;
                valid++;
                if (out == label) tp[label]++;
            }
        }

        if (valid == 0) {
            pop[i].fitness_score = 0.0f;
            continue;
        }

        // each class contributes equally regardless of how many samples preventing the network from ignoring rare classes
        // (HW 0, 1, 7, 8) in favor of the common ones (HW 3, 4, 5).
        float recall_sum = 0.0f;
        int active_classes = 0;
        for (int c = 0; c < SNN_NUM_HW_CLASSES; c++) {
            if (gt[c] > 0) {
                recall_sum += (float)tp[c] / (float)gt[c];
                active_classes++;
            }
        }
        float accuracy = (active_classes > 0) ? recall_sum / active_classes : 0.0f;

        // normalized Shannon entropy of predictions
        // Measures how spread out the predictions are across classes.
        float entropy = 0.0f;
        for (int c = 0; c < SNN_NUM_HW_CLASSES; c++) {
            if (pred[c] > 0) {
                float p = (float)pred[c] / (float)valid;
                entropy -= p * logf(p);
            }
        }
        float diversity = entropy / MAX_ENTROPY;  // normalized to [0, 1]

        // fraction of traces that produced a decision
        float utility = (float)valid / (float)ds->num_traces;

        // complexity penalty
        float penalty = 0.0f;
        if (net->num_synapses > THRESHOLD_SYNAPSES) {
            penalty += (net->num_synapses - THRESHOLD_SYNAPSES) * PENALTY_SYNAPSE;
        }
        if (net->num_hidden > THRESHOLD_NEURONS) {
            penalty += (net->num_hidden - THRESHOLD_NEURONS) * PENALTY_NEURON;
        }

        float raw = w_accuracy  * accuracy
                   + w_diversity * diversity
                   + w_utility   * utility
                   - penalty;

        pop[i].fitness_score = (raw > 0.0f) ? raw : 0.0f;
    }
}
