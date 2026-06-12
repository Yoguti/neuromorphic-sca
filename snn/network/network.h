#include "neuron.h"
#include "../libs/arena.h"
#include <string.h>

#ifndef NETWORK_H
#define NETWORK_H

#define UP_INPUT_ID   0
#define DOWN_INPUT_ID 1

#define SNN_NUM_INPUTS       2    // fixed: UP (0), DOWN (1)
#define SNN_NUM_OUTPUTS      9    // fixed: HW 0..8

// maximum LIF neurons (outputs + hidden) that can ever exist in one network;
// used to size the accumulated_inputs scratch buffer in the struct
#define SNN_MAX_LIF_COUNT    64


// MACROS FOR CONVERTING BETWEEN NETWORK IDS AND LIF ARRAY INDICES {

// Network ID --> lif array index  (only valid for non-input nodes)
// // offsets the reserved input nodes to find the real indexes
#define SNN_NODE_TO_LIF(net_id)  ((net_id) - SNN_NUM_INPUTS)

// lif array index --> network ID
#define SNN_LIF_TO_NODE(lif_idx) ((lif_idx) + SNN_NUM_INPUTS)

// Total LIF neurons allocated (outputs + hidden) (input nodes are not LIFs)
#define SNN_LIF_COUNT(net) ((net)->num_outputs + (net)->num_hidden)

// }

typedef struct {
    uint16_t source_node;  // network ID of the firing node (can be 0 or 1 for inputs)
    uint16_t target_node;  // network ID of the receiving node (NEVER 0 or 1)
    int8_t   weight;       // synaptic weight [-127, 127]
} synapse_t;


typedef struct {
    uint16_t num_inputs;   // always SNN_NUM_INPUTS  (2)
    uint16_t num_outputs;  // always SNN_NUM_OUTPUTS (9)
    uint16_t num_hidden;   // mutable by EONS, starts at 0

    Arena *arena;

    lif_neuron_t *nodes;

    int8_t input_spikes[SNN_NUM_INPUTS]; // inputs gates for db data don't need neurons

    uint16_t   num_synapses;  // mutable by EONS
    synapse_t *synapses;

    // scratch buffer reused every tick; avoids a VLA on the stack
    int16_t accumulated_inputs[SNN_MAX_LIF_COUNT];
} snn_network_t;

// Allocate and initialise a minimal network (no hidden nodes, no synapses)
// All LIF output neurons are initialised with lif_init_default(lif_neuron_t *)
snn_network_t *snn_create(Arena *arena);

// Reset all LIF states to resting (call between traces during inference/training)
void snn_reset(snn_network_t *net);

/* advance network by one tick.

  1. For every synapse: if source fired last tick, accumulate
    weight * source_fired into the target LIF neuron's input
  2. For every LIF neuron: call lif_step() with the accumulated input
  3. Clear accumulated inputs for next tick.

 caller must set net->input_spikes[] BEFORE calling snn_tick()*/

void snn_tick(snn_network_t *net);

// Add a new hidden node, returns network id
uint16_t snn_add_hidden(snn_network_t *net);

// delete a hidden node by network id
int snn_delete_hidden(snn_network_t *net, uint16_t node_id);

// add a synapse, returns 0 on success and -1 on failure
int snn_add_synapse(snn_network_t *net, uint16_t src, uint16_t tgt, int8_t w);

// remove synapse by index in the synapses array.
void snn_delete_synapse(snn_network_t *net, uint16_t synapse_idx);

// Get pointer to LIF neuron by network ID (returns NULL for input nodes or out-of-bounds)
lif_neuron_t *get_neuron(snn_network_t *net, uint16_t node_id);

#endif // NETWORK_H
