#ifndef DATASET_H
#define DATASET_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    int8_t *traces;       // Flattened array: [num_traces * trace_length]
    uint8_t *labels;      // Array of Hamming Weight labels: [num_traces]
    size_t num_traces;
    size_t trace_length;
} ascad_dataset_t;

// Load traces and labels from your modulated HDF5 file
// if is_attack is true, loads from /Attack_traces. Otherwise, from /Profiling_traces.
ascad_dataset_t *dataset_load(const char *filepath, size_t max_traces, bool is_attack);

// Free the dataset memory
void dataset_free(ascad_dataset_t *dataset);

#endif // DATASET_H