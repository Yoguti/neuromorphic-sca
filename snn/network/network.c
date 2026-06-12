#include "network.h"


// Allocate and initialise a minimal network (no hidden nodes, no synapses)
// All LIF output neurons are initialised with lif_init_default(lif_neuron_t *)
snn_network_t *snn_create(Arena *arena) {
    snn_network_t *net = arena_push_struct(arena, snn_network_t);
    if (!net) return NULL;

    net->arena = arena;

    net->num_inputs = SNN_NUM_INPUTS;
    net->num_outputs = SNN_NUM_OUTPUTS;
    net->num_hidden = 0;

    net->nodes = arena_push_array(arena, lif_neuron_t, SNN_LIF_COUNT(net));
    if (!net->nodes) {
        return NULL;
    }

    for (uint16_t i = 0; i < net->num_outputs; i++) {
        lif_init_default(&net->nodes[i]);
    }

    net->num_synapses = 0;
    net->synapses = NULL;

    for (uint16_t i = 0; i < SNN_NUM_INPUTS; i++) {
        net->input_spikes[i] = 0;
    }

    for (uint16_t i = 0; i < SNN_MAX_LIF_COUNT; i++) {
        net->accumulated_inputs[i] = 0;
    }

    return net;
}

// Reset all LIF states to resting (call between traces during inference/training)
void snn_reset(snn_network_t *net) {
    for (uint16_t i = 0; i < SNN_LIF_COUNT(net); i++) {
        lif_reset(&net->nodes[i]);
    }
}

// Get pointer to LIF neuron by network ID (returns NULL for input nodes or out-of-bounds)
lif_neuron_t *get_neuron(snn_network_t *net, uint16_t node_id) {
    if (node_id < SNN_NUM_INPUTS) {
        return NULL;
    }
    uint16_t lif_idx = SNN_NODE_TO_LIF(node_id);
    if (lif_idx >= SNN_LIF_COUNT(net)) {
        return NULL;
    }
    return &net->nodes[lif_idx];
}

/* advance network by one tick.
  1. For every synapse: if source fired last tick, accumulate
    weight into the target LIF neuron's input
  2. For every LIF neuron: call lif_step() with the accumulated input
  3. Clear accumulated inputs for next tick.
*/

void snn_tick(snn_network_t *net) {
    uint16_t num_lif = SNN_LIF_COUNT(net);

    // accumulation phase (1)
    for (uint16_t i = 0; i < net->num_synapses; i++) {
        const synapse_t *syn = &net->synapses[i];
        uint16_t src = syn->source_node;
        uint16_t tgt = syn->target_node;

        int8_t source_fired = 0; // int8_t to carry signed ADM input spikes (-1, 0, +1)

        if (src < SNN_NUM_INPUTS) {
            source_fired = net->input_spikes[src];
        } else {
            source_fired = net->nodes[SNN_NODE_TO_LIF(src)].has_fired;
        }

        if (source_fired) {
            uint16_t tgt_idx = SNN_NODE_TO_LIF(tgt);
            net->accumulated_inputs[tgt_idx] += syn->weight * source_fired;
        }
    }

    // stepping phase: update every LIF neuron exactly once (2)
    for (uint16_t i = 0; i < num_lif; i++) {
        lif_step(&net->nodes[i], net->accumulated_inputs[i]);
    }

    // clear for next tick (3)
    for (uint16_t i = 0; i < num_lif; i++) {
        net->accumulated_inputs[i] = 0;
    }
    for (uint16_t i = 0; i < SNN_NUM_INPUTS; i++) {
        net->input_spikes[i] = 0;
    }
}

// Add a new hidden node, returns network id
uint16_t snn_add_hidden(snn_network_t *net) {
    if (SNN_LIF_COUNT(net) >= SNN_MAX_LIF_COUNT) {
        return UINT16_MAX;
    }

    uint16_t new_node_id = net->num_inputs + net->num_outputs + net->num_hidden;
    uint16_t current_count = SNN_LIF_COUNT(net);

    lif_neuron_t *new_nodes = arena_push_array(net->arena, lif_neuron_t, current_count + 1);
    if (!new_nodes) {
        return UINT16_MAX;
    }

    if (net->nodes != NULL && current_count > 0) {
        memcpy(new_nodes, net->nodes, sizeof(lif_neuron_t) * current_count);
    }

    net->nodes = new_nodes;

    lif_init_default(&net->nodes[current_count]);

    net->num_hidden++;
    return new_node_id;
}

// delete a hidden node by network id
int snn_delete_hidden(snn_network_t *net, uint16_t node_id) {
    if (node_id < SNN_NUM_INPUTS + net->num_outputs) {
        return -1;
    }
    uint16_t lif_idx = SNN_NODE_TO_LIF(node_id);
    if (lif_idx >= SNN_LIF_COUNT(net)) {
        return -1;
    }

    // shift neurons down to fill the gap
    for (uint16_t i = lif_idx; i < SNN_LIF_COUNT(net) - 1; i++) {
        net->nodes[i] = net->nodes[i + 1];
    }

    net->num_hidden--;

    // remove synapses connected to the deleted node
    for (uint16_t i = 0; i < net->num_synapses; ) {
        const synapse_t *syn = &net->synapses[i];
        if (syn->source_node == node_id || syn->target_node == node_id) {
            snn_delete_synapse(net, i);
        } else {
            i++;
        }
    }

    // hidden nodes above the deleted one shift down by one in the LIF array,
    // so their network IDs must follow; output nodes (IDs < node_id) are unaffected
    for (uint16_t i = 0; i < net->num_synapses; i++) {
        synapse_t *syn = &net->synapses[i];
        if (syn->source_node > node_id) {
            syn->source_node--;
        }
        if (syn->target_node > node_id) {
            syn->target_node--;
        }
    }

    return 0;
}

int snn_add_synapse(snn_network_t *net, uint16_t src, uint16_t tgt, int8_t w) {
    if (!net) {
        return -1;
    }
    if (tgt < SNN_NUM_INPUTS || SNN_NODE_TO_LIF(tgt) >= SNN_LIF_COUNT(net)) {
        return -1;
    }
    uint16_t total_nodes = net->num_inputs + net->num_outputs + net->num_hidden;
    if (src >= total_nodes) {
        return -1;
    }

    synapse_t *new_synapses = arena_push_array(net->arena, synapse_t, net->num_synapses + 1);
    if (!new_synapses) {
        return -1;
    }

    if (net->synapses != NULL && net->num_synapses > 0) {
        memcpy(new_synapses, net->synapses, sizeof(synapse_t) * net->num_synapses);
    }

    net->synapses = new_synapses;

    net->synapses[net->num_synapses].source_node = src;
    net->synapses[net->num_synapses].target_node = tgt;
    net->synapses[net->num_synapses].weight = w;

    net->num_synapses++;
    return 0; 
}

// remove synapse by index in the synapses array.
void snn_delete_synapse(snn_network_t *net, uint16_t synapse_idx) {
    if (synapse_idx >= net->num_synapses) {
        return;
    }
    for (uint16_t i = synapse_idx; i < net->num_synapses - 1; i++) {
        net->synapses[i] = net->synapses[i + 1];
    }
    net->num_synapses--;
}
