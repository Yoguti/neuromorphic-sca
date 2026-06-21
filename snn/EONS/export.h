#ifndef EXPORT_H
#define EXPORT_H

#include "network/network.h"

// Export net's topology and parameters to two CSV files for downstream VHDL
// synthesis: "<basename>_neurons.csv" and "<basename>_synapses.csv".
// Returns 0 on success, -1 on failure (e.g. file couldn't be opened).
int export_network_csv(const snn_network_t *net, const char *basename);
snn_network_t* import_network_csv(Arena *arena, const char *base_filename);
#endif // EXPORT_H
