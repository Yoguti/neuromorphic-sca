#ifndef EXPORT_H
#define EXPORT_H

#include "network/network.h"

// export topology and parameters to two CSV files for downstream VHDL
int export_network_csv(const snn_network_t *net, const char *basename);
snn_network_t* import_network_csv(Arena *arena, const char *base_filename);
#endif // EXPORT_H
