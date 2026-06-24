#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include "hdf5.h"

ascad_dataset_t *dataset_load(const char *filepath, size_t max_traces, bool is_attack) {
    ascad_dataset_t *ds = calloc(1, sizeof(ascad_dataset_t));
    if (!ds) return NULL;

    hid_t fid = H5Fopen(filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) { printf("Failed to open: %s\n", filepath); free(ds); return NULL; }

    const char *grp        = is_attack ? "/Attack_traces"    : "/Profiling_traces";
    char trace_path[256], label_path[256];
    snprintf(trace_path, sizeof(trace_path), "%s/traces",    grp);
    snprintf(label_path, sizeof(label_path), "%s/labels_hw", grp);

    hid_t tds = H5Dopen2(fid, trace_path, H5P_DEFAULT);
    if (tds < 0) { printf("Cannot open: %s\n", trace_path); H5Fclose(fid); free(ds); return NULL; }

    hid_t tsp = H5Dget_space(tds);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(tsp, dims, NULL);
    H5Sclose(tsp);

    size_t trace_length = (size_t)dims[1];
    if (max_traces == 0 || max_traces > (size_t)dims[0]) max_traces = (size_t)dims[0];

    ds->num_traces   = max_traces;
    ds->trace_length = trace_length;
    ds->traces = malloc(max_traces * trace_length * sizeof(float));
    ds->labels = malloc(max_traces * sizeof(uint8_t));
    if (!ds->traces || !ds->labels) {
        free(ds->traces); free(ds->labels); free(ds);
        H5Dclose(tds); H5Fclose(fid); return NULL;
    }

    hid_t fsp = H5Dget_space(tds);
    hsize_t off2[2] = {0, 0}, cnt2[2] = {(hsize_t)max_traces, (hsize_t)trace_length};
    H5Sselect_hyperslab(fsp, H5S_SELECT_SET, off2, NULL, cnt2, NULL);
    hid_t msp = H5Screate_simple(2, cnt2, NULL);
    H5Dread(tds, H5T_NATIVE_FLOAT, msp, fsp, H5P_DEFAULT, ds->traces);
    H5Sclose(msp); H5Sclose(fsp); H5Dclose(tds);

    hid_t lds = H5Dopen2(fid, label_path, H5P_DEFAULT);
    if (lds < 0) { printf("Cannot open: %s\n", label_path); H5Fclose(fid); dataset_free(ds); return NULL; }
    hid_t lsp = H5Dget_space(lds);
    hsize_t loff = 0, lcnt = (hsize_t)max_traces;
    H5Sselect_hyperslab(lsp, H5S_SELECT_SET, &loff, NULL, &lcnt, NULL);
    hid_t lms = H5Screate_simple(1, &lcnt, NULL);
    H5Dread(lds, H5T_NATIVE_UINT8, lms, lsp, H5P_DEFAULT, ds->labels);
    H5Sclose(lms); H5Sclose(lsp); H5Dclose(lds);

    H5Fclose(fid);
    return ds;
}

void dataset_free(ascad_dataset_t *ds) {
    if (ds) {
        free(ds->traces);
        free(ds->labels);
        free(ds);
    }
}
