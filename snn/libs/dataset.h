#ifndef DATASET_H
#define DATASET_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    float   *traces;
    uint8_t *labels;
    size_t   num_traces;
    size_t   trace_length;
} ascad_dataset_t;

ascad_dataset_t *dataset_load(const char *filepath, size_t max_traces, bool is_attack);
void dataset_free(ascad_dataset_t *dataset);

#endif
