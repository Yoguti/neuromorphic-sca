#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "hdf5.h"

#define IN_FILE "../ATMEGA-AES-ASCAD_databases/ASCAD.h5"
#define OUT_FILE "ascad_modulated.h5"
#define GROUP_NAME "/Profiling_traces"
#define DATASET_NAME "/Profiling_traces/traces"
#define LABELS_NAME "/Profiling_traces/labels"
#define METADATA_NAME "/Profiling_traces/metadata"

#define NUM_TRACES 50000
#define TRACE_LENGTH 700
#define THRESHOLD 6

//gcc modulate_data.c -std=c11 -lhdf5 -o modulate_data

int main() {
    hid_t file_in, file_out, dataset_in, dataset_out, dataspace_id, group_out, dcpl_id;
    hsize_t dims[2] = {NUM_TRACES, TRACE_LENGTH};

    int8_t *traces_in = (int8_t *)calloc(NUM_TRACES * TRACE_LENGTH, sizeof(int8_t));
    int8_t *traces_out = (int8_t *)calloc(NUM_TRACES * TRACE_LENGTH, sizeof(int8_t));

    if (!traces_in || !traces_out) {
        printf("calloc failed\n");
        return 1;
    }

    file_in = H5Fopen(IN_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_in < 0) {
        fprintf(stderr, "Failed to open input file '%s'\n", IN_FILE);
        return 1;
    }

    dataset_in = H5Dopen2(file_in, DATASET_NAME, H5P_DEFAULT);
    if (dataset_in < 0) {
        fprintf(stderr, "Failed to open dataset '%s'\n", DATASET_NAME);
        H5Fclose(file_in);
        return 1;
    }

    H5Dread(dataset_in, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, traces_in);
    H5Dclose(dataset_in);

    printf("Data loaded. Starting Profiling traces modulation.\n");
    
    for (int trace = 0; trace < NUM_TRACES; trace++) {
        int offset = trace * TRACE_LENGTH;
        int reference = traces_in[offset];
        traces_out[offset] = 0;

        for (int sample = 1; sample < TRACE_LENGTH; sample++) {
            int8_t current_sample = traces_in[offset + sample];
            int delta = current_sample - reference;
            
            if (delta >= THRESHOLD) {
                traces_out[offset + sample] = 1; 
                reference += THRESHOLD;
            } else if (delta <= -THRESHOLD) {
                traces_out[offset + sample] = -1;
                reference -= THRESHOLD;
            } else {
                traces_out[offset + sample] = 0;
            }
        }
    }

    printf("Modulation complete. Saving and copying Profiling metadata.\n");
    
    file_out = H5Fcreate(OUT_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    group_out = H5Gcreate2(file_out, GROUP_NAME, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dataspace_id = H5Screate_simple(2, dims, NULL);

    dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[2] = {100, TRACE_LENGTH};
    H5Pset_chunk(dcpl_id, 2, chunk_dims);
    H5Pset_deflate(dcpl_id, 6);

    dataset_out = H5Dcreate2(file_out, DATASET_NAME, H5T_NATIVE_INT8, dataspace_id,
                              H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

    H5Dwrite(dataset_out, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, traces_out);
    H5Pclose(dcpl_id);

    H5Ocopy(file_in, LABELS_NAME, file_out, LABELS_NAME, H5P_DEFAULT, H5P_DEFAULT);
    H5Ocopy(file_in, METADATA_NAME, file_out, METADATA_NAME, H5P_DEFAULT, H5P_DEFAULT);

    H5Dclose(dataset_out);
    H5Sclose(dataspace_id);
    H5Gclose(group_out);

    printf("\nData loaded. Starting Attack traces modulation.\n");

    dataset_in = H5Dopen2(file_in, "/Attack_traces/traces", H5P_DEFAULT);
    H5Dread(dataset_in, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, traces_in);
    H5Dclose(dataset_in);

    for (int trace = 0; trace < 10000; trace++) {
        int offset = trace * TRACE_LENGTH;
        int reference = traces_in[offset];
        traces_out[offset] = 0;

        for (int sample = 1; sample < TRACE_LENGTH; sample++) {
            int8_t current_sample = traces_in[offset + sample];
            int delta = current_sample - reference;
            
            if (delta >= THRESHOLD) {
                traces_out[offset + sample] = 1; 
                reference += THRESHOLD;
            } else if (delta <= -THRESHOLD) {
                traces_out[offset + sample] = -1;
                reference -= THRESHOLD;
            } else {
                traces_out[offset + sample] = 0;
            }
        }
    }

    printf("Modulation complete. Saving and copying Attack metadata.\n");
    
    group_out = H5Gcreate2(file_out, "/Attack_traces", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    dims[0] = 10000; 
    dataspace_id = H5Screate_simple(2, dims, NULL);

    dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl_id, 2, chunk_dims);
    H5Pset_deflate(dcpl_id, 6);

    dataset_out = H5Dcreate2(file_out, "/Attack_traces/traces", H5T_NATIVE_INT8, dataspace_id,
                              H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

    H5Dwrite(dataset_out, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, traces_out);
    H5Pclose(dcpl_id);

    H5Ocopy(file_in, "/Attack_traces/labels", file_out, "/Attack_traces/labels", H5P_DEFAULT, H5P_DEFAULT);
    H5Ocopy(file_in, "/Attack_traces/metadata", file_out, "/Attack_traces/metadata", H5P_DEFAULT, H5P_DEFAULT);

    H5Dclose(dataset_out);
    H5Sclose(dataspace_id);
    H5Gclose(group_out);
    H5Fclose(file_out);
    H5Fclose(file_in);

    free(traces_in);
    free(traces_out);

    printf("Done!\n");
    return 0;
}