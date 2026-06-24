#include "network.h"

snn_network_t *snn_create(Arena *arena) {
    snn_network_t *net = arena_push_struct(arena, snn_network_t);
    if (!net) return NULL;
    net->num_inputs = SNN_NUM_INPUTS; net->num_outputs = SNN_NUM_OUTPUTS;
    net->num_hidden = 0; net->num_synapses = 0;
    net->nodes    = arena_push_array(arena, lif_neuron_t, SNN_MAX_LIF_COUNT);
    net->synapses = arena_push_array(arena, synapse_t,    SNN_MAX_SYNAPSES);
    if (!net->nodes || !net->synapses) return NULL;
    for (uint16_t i = 0; i < net->num_outputs; i++) lif_init_default(&net->nodes[i]);
    for (uint16_t i = 0; i < SNN_NUM_INPUTS; i++) net->input_values[i] = 0.0f;
    for (uint16_t i = 0; i < SNN_MAX_LIF_COUNT; i++) net->accumulated_inputs[i] = 0;
    return net;
}

snn_network_t *snn_clone_into(Arena *dest_arena, const snn_network_t *src) {
    if (!src) return NULL;
    snn_network_t *clone = arena_push_struct(dest_arena, snn_network_t);
    if (!clone) return NULL;
    clone->num_inputs = src->num_inputs; clone->num_outputs = src->num_outputs;
    clone->num_hidden = src->num_hidden; clone->num_synapses = src->num_synapses;
    clone->nodes    = arena_push_array(dest_arena, lif_neuron_t, SNN_MAX_LIF_COUNT);
    clone->synapses = arena_push_array(dest_arena, synapse_t,    SNN_MAX_SYNAPSES);
    if (!clone->nodes || !clone->synapses) return NULL;
    memcpy(clone->nodes,    src->nodes,    sizeof(lif_neuron_t) * SNN_LIF_COUNT(src));
    memcpy(clone->synapses, src->synapses, sizeof(synapse_t)    * src->num_synapses);
    memcpy(clone->input_values, src->input_values, sizeof(src->input_values));
    for (uint16_t i = 0; i < SNN_MAX_LIF_COUNT; i++) clone->accumulated_inputs[i] = 0;
    return clone;
}

void snn_reset(snn_network_t *net) {
    for (uint16_t i = 0; i < SNN_LIF_COUNT(net); i++) lif_reset(&net->nodes[i]);
}

lif_neuron_t *get_neuron(snn_network_t *net, uint16_t node_id) {
    if (node_id < SNN_NUM_INPUTS) return NULL;
    uint16_t lif_idx = SNN_NODE_TO_LIF(node_id);
    if (lif_idx >= SNN_LIF_COUNT(net)) return NULL;
    return &net->nodes[lif_idx];
}

void snn_tick(snn_network_t *net) {
    uint16_t num_lif = SNN_LIF_COUNT(net);
    for (uint16_t i = 0; i < net->num_synapses; i++) {
        const synapse_t *syn = &net->synapses[i];
        uint16_t src = syn->source_node, tgt = syn->target_node;
        int32_t drive;
        if (src < SNN_NUM_INPUTS)
            drive = (int32_t)((float)syn->weight * net->input_values[src]);
        else if (net->nodes[SNN_NODE_TO_LIF(src)].has_fired)
            drive = (int32_t)syn->weight;
        else
            drive = 0;
        if (drive != 0) net->accumulated_inputs[SNN_NODE_TO_LIF(tgt)] += drive;
    }
    for (uint16_t i = 0; i < num_lif; i++) lif_step(&net->nodes[i], net->accumulated_inputs[i]);
    for (uint16_t i = 0; i < num_lif; i++) net->accumulated_inputs[i] = 0;
}

uint16_t snn_add_hidden(snn_network_t *net) {
    if (SNN_LIF_COUNT(net) >= SNN_MAX_LIF_COUNT) return UINT16_MAX;
    uint16_t new_node_id = net->num_inputs + net->num_outputs + net->num_hidden;
    lif_init_default(&net->nodes[SNN_LIF_COUNT(net)]);
    net->num_hidden++;
    return new_node_id;
}

int snn_delete_hidden(snn_network_t *net, uint16_t node_id) {
    if (node_id < SNN_NUM_INPUTS + net->num_outputs) return -1;
    uint16_t lif_idx = SNN_NODE_TO_LIF(node_id);
    if (lif_idx >= SNN_LIF_COUNT(net)) return -1;
    for (uint16_t i = lif_idx; i < SNN_LIF_COUNT(net) - 1; i++) net->nodes[i] = net->nodes[i+1];
    net->num_hidden--;
    for (uint16_t i = 0; i < net->num_synapses; )
        if (net->synapses[i].source_node == node_id || net->synapses[i].target_node == node_id)
            snn_delete_synapse(net, i);
        else i++;
    for (uint16_t i = 0; i < net->num_synapses; i++) {
        if (net->synapses[i].source_node > node_id) net->synapses[i].source_node--;
        if (net->synapses[i].target_node > node_id) net->synapses[i].target_node--;
    }
    return 0;
}

int snn_add_synapse(snn_network_t *net, uint16_t src, uint16_t tgt, int8_t w) {
    if (!net) return -1;
    if (tgt < SNN_NUM_INPUTS || SNN_NODE_TO_LIF(tgt) >= SNN_LIF_COUNT(net)) return -1;
    if (src >= (uint16_t)(net->num_inputs + net->num_outputs + net->num_hidden)) return -1;
    if (net->num_synapses >= SNN_MAX_SYNAPSES) return -1;
    net->synapses[net->num_synapses].source_node = src;
    net->synapses[net->num_synapses].target_node = tgt;
    net->synapses[net->num_synapses].weight      = w;
    net->num_synapses++;
    return 0;
}

void snn_delete_synapse(snn_network_t *net, uint16_t idx) {
    if (idx >= net->num_synapses) return;
    for (uint16_t i = idx; i < net->num_synapses - 1; i++) net->synapses[i] = net->synapses[i+1];
    net->num_synapses--;
}
