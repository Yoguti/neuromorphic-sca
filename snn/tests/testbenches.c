#include "testbenches.h"
#include "libs/dataset.h"
#include "EONS/engine.h"
#include "EONS/export.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void run_test(const char *h5_path, const char *network_base) {
    printf("Searching for network CSV files with prefix: %s\n", network_base);

    Arena *arena = engine_get_arena_a();

    snn_network_t *net = import_network_csv(arena, network_base);
    if (!net) {
        printf("[ERROR] Could not load network %s_neurons.csv / %s_synapses.csv\n", network_base, network_base);
        printf("Make sure the files exist in the network-csvs/ folder.\n");
        return;
    }

    printf("[SUCCESS] Network loaded! Hidden neurons: %u, Synapses: %u\n",
           net->num_hidden, net->num_synapses);

    size_t test_traces_count = 10000;
    printf("Loading %zu test traces from %s...\n", test_traces_count, h5_path);

    ascad_dataset_t *ds = dataset_load(h5_path, test_traces_count, true);
    if (!ds) {
        printf("[ERROR] Failed to load HDF5 dataset.\n");
        return;
    }

    printf("[SUCCESS] Dataset loaded.\n");

    printf("Evaluating network on %zu traces...\n", ds->num_traces);

    uint32_t correct = 0;
    uint32_t class_predictions[SNN_NUM_HW_CLASSES] = {0};

    for (size_t t = 0; t < ds->num_traces; t++) {
        const int8_t *trace = &ds->traces[t * ds->trace_length];
        uint8_t actual_label = ds->labels[t];

        uint8_t pred = evaluate_network(net, trace, ds->trace_length);

        class_predictions[pred]++;

        if (pred == actual_label) {
            correct++;
        }
    }

    float accuracy = (float)correct / (float)ds->num_traces;

    int unique_classes = 0;
    printf("\nTest Results\n");
    printf("Prediction distribution per class:\n");

    for (int c = 0; c < SNN_NUM_HW_CLASSES; c++) {
        printf("  HW Class %d: %u times\n", c, class_predictions[c]);
        if (class_predictions[c] > 0) {
            unique_classes++;
        }
    }

    printf("Total Traces:     %zu\n", ds->num_traces);
    printf("Correct:          %u\n", correct);
    printf("Unique Classes:   %d of %d\n", unique_classes, SNN_NUM_HW_CLASSES);
    printf("Final Accuracy:   %.2f%%\n", accuracy * 100.0f);


    dataset_free(ds);
}