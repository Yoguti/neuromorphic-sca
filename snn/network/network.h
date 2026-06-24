#include "neuron.h"
#include "../libs/arena.h"
#include <string.h>

#ifndef NETWORK_H
#define NETWORK_H

#define SNN_NUM_POIS      20
#define SNN_NUM_INPUTS    SNN_NUM_POIS
#define SNN_NUM_OUTPUTS   9
#define SNN_MAX_LIF_COUNT 256
#define SNN_MAX_SYNAPSES  2048

#define SNN_NODE_TO_LIF(net_id)  ((net_id) - SNN_NUM_INPUTS)
#define SNN_LIF_TO_NODE(lif_idx) ((lif_idx) + SNN_NUM_INPUTS)
#define SNN_LIF_COUNT(net)       ((net)->num_outputs + (net)->num_hidden)

typedef struct {
    uint16_t source_node;
    uint16_t target_node;
    int8_t   weight;
} synapse_t;

typedef struct {
    uint16_t num_inputs;
    uint16_t num_outputs;
    uint16_t num_hidden;
    lif_neuron_t *nodes;
    float         input_values[SNN_NUM_INPUTS];
    uint16_t   num_synapses;
    synapse_t *synapses;
    int32_t accumulated_inputs[SNN_MAX_LIF_COUNT];
} snn_network_t;

snn_network_t *snn_create(Arena *arena);
snn_network_t *snn_clone_into(Arena *dest_arena, const snn_network_t *src);
void snn_reset(snn_network_t *net);
void snn_tick(snn_network_t *net);
uint16_t snn_add_hidden(snn_network_t *net);
int snn_delete_hidden(snn_network_t *net, uint16_t node_id);
int snn_add_synapse(snn_network_t *net, uint16_t src, uint16_t tgt, int8_t w);
void snn_delete_synapse(snn_network_t *net, uint16_t synapse_idx);
lif_neuron_t *get_neuron(snn_network_t *net, uint16_t node_id);

#endif
