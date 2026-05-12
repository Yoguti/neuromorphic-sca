#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include "hdf5.h"

#define FILE_NAME "../ATMEGA-AES-ASCAD_databases/ASCAD.h5"
#define DATASET "/Profiling_traces/traces"
#define THRESHOLD 6
#define TRACE_LEN 700

//gcc benchmark_conversion.c -lhdf5 -o benchmark_conversion

int main(void) {
    hid_t file = H5Fopen(FILE_NAME, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen2(file, DATASET, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dataset);

    hsize_t start[2] = {0, 0};
    hsize_t count[2] = {1, TRACE_LEN};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t memspace = H5Screate_simple(2, count, NULL);
    int8_t trace[TRACE_LEN];
    H5Dread(dataset, H5T_NATIVE_INT8, memspace, filespace, H5P_DEFAULT, trace);

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    int ref = trace[0];
    int checksum = ref;
    for (int i = 1; i < TRACE_LEN; ++i) {
        int delta = trace[i] - ref;
        if (delta >= THRESHOLD) {
            ref += THRESHOLD;
            checksum += 1;
        } else if (delta <= -THRESHOLD) {
            ref -= THRESHOLD;
            checksum -= 1;
        }
    }

    gettimeofday(&t1, NULL);
    double seconds = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / 1e6;

    printf("one trace delta modulation time: %.9f s, checksum=%d\n", seconds, checksum);

    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dataset);
    H5Fclose(file);
    return 0;
}
