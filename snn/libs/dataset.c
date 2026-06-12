#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include "hdf5.h"

#define TRACE_DATASET_NAME "/Profiling_traces/traces"
#define LABEL_DATASET_NAME "/Profiling_traces/labels_hw_byte0"
#define TRACE_LENGTH 700

ascad_dataset_t *dataset_load(const char *filepath, size_t max_traces) {
    ascad_dataset_t *ds = malloc(sizeof(ascad_dataset_t));
    if (!ds) return NULL;

    ds->num_traces = max_traces;
    ds->trace_length = TRACE_LENGTH;
    ds->traces = malloc(max_traces * TRACE_LENGTH * sizeof(int8_t));
    ds->labels = malloc(max_traces * sizeof(uint8_t));

    if (!ds->traces || !ds->labels) {
        printf("Failed to allocate memory for dataset.\n");
        free(ds->traces);
        free(ds->labels);
        free(ds);
        return NULL;
    }

    hid_t file_id = H5Fopen(filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        printf("Failed to open HDF5 file: %s\n", filepath);
        dataset_free(ds);
        return NULL;
    }

    hid_t trace_dataset = H5Dopen2(file_id, TRACE_DATASET_NAME, H5P_DEFAULT);
    
    hid_t trace_space = H5Dget_space(trace_dataset);
    hsize_t offset[2] = {0, 0};
    hsize_t count[2] = {max_traces, TRACE_LENGTH};
    H5Sselect_hyperslab(trace_space, H5S_SELECT_SET, offset, NULL, count, NULL);
    
    hid_t mem_space_traces = H5Screate_simple(2, count, NULL);
    H5Dread(trace_dataset, H5T_NATIVE_INT8, mem_space_traces, trace_space, H5P_DEFAULT, ds->traces);
    
    H5Sclose(mem_space_traces);
    H5Sclose(trace_space);
    H5Dclose(trace_dataset);

    hid_t label_dataset = H5Dopen2(file_id, LABEL_DATASET_NAME, H5P_DEFAULT);
    hid_t label_space = H5Dget_space(label_dataset);
    hsize_t offset_labels[1] = {0};
    hsize_t count_labels[1] = {max_traces};
    H5Sselect_hyperslab(label_space, H5S_SELECT_SET, offset_labels, NULL, count_labels, NULL);
    
    hid_t mem_space_labels = H5Screate_simple(1, count_labels, NULL);
    H5Dread(label_dataset, H5T_NATIVE_UINT8, mem_space_labels, label_space, H5P_DEFAULT, ds->labels);

    H5Sclose(mem_space_labels);
    H5Sclose(label_space);
    H5Dclose(label_dataset);
    H5Fclose(file_id);

    return ds;
}

void dataset_free(ascad_dataset_t *dataset) {
    if (dataset) {
        free(dataset->traces);
        free(dataset->labels);
        free(dataset);
    }
}