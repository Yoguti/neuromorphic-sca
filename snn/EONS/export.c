#include "export.h"
#include <stdio.h>
#include <string.h>

int export_network_csv(const snn_network_t *net, const char *basename) {
    if (!net || !basename) return -1;

    char neuron_path[512];
    char synapse_path[512];
    snprintf(neuron_path, sizeof(neuron_path), "%s_neurons.csv", basename);
    snprintf(synapse_path, sizeof(synapse_path), "%s_synapses.csv", basename);

    FILE *nf = fopen(neuron_path, "w");
    if (!nf) {
        fprintf(stderr, "export_network_csv: could not open %s for writing\n", neuron_path);
        return -1;
    }

    fprintf(nf, "node_id,type,threshold,leak_factor,reset_potential,resting_potential,refractory_period\n");

    uint16_t num_lif = SNN_LIF_COUNT(net);
    for (uint16_t lif_idx = 0; lif_idx < num_lif; lif_idx++) {
        uint16_t node_id = SNN_LIF_TO_NODE(lif_idx);
        const char *type = (lif_idx < net->num_outputs) ? "output" : "hidden";
        const lif_neuron_t *n = &net->nodes[lif_idx];

        fprintf(nf, "%u,%s,%d,%d,%d,%d,%u\n",
                node_id, type,
                n->threshold, n->leak_factor,
                n->reset_potential, n->resting_potential,
                n->refractory_period);
    }
    fclose(nf);

    FILE *sf = fopen(synapse_path, "w");
    if (!sf) {
        fprintf(stderr, "export_network_csv: could not open %s for writing\n", synapse_path);
        return -1;
    }

    fprintf(sf, "source_node,target_node,weight\n");
    for (uint16_t i = 0; i < net->num_synapses; i++) {
        const synapse_t *syn = &net->synapses[i];
        fprintf(sf, "%u,%u,%d\n", syn->source_node, syn->target_node, syn->weight);
    }
    fclose(sf);

    printf("Exported network: %u neurons -> %s, %u synapses -> %s\n",
           num_lif, neuron_path, net->num_synapses, synapse_path);

    return 0;
}

snn_network_t* import_network_csv(Arena *arena, const char *base_filename) {
    char neuron_path[512], synapse_path[512];
    snprintf(neuron_path, sizeof(neuron_path), "%s_neurons.csv", base_filename);
    snprintf(synapse_path, sizeof(synapse_path), "%s_synapses.csv", base_filename);

    FILE *fn = fopen(neuron_path, "r");
    FILE *fs = fopen(synapse_path, "r");
    if (!fn || !fs) {
        if(fn) fclose(fn);
        if(fs) fclose(fs);
        return NULL;
    }
    snn_network_t *net = snn_create(arena);
    char line[256];

    fgets(line, sizeof(line), fn);
    while (fgets(line, sizeof(line), fn)) {
        int node_id, thresh, leak, reset, resting, refrac;
        if (sscanf(line, "%d,%*[^,],%d,%d,%d,%d,%d", &node_id, &thresh, &leak, &reset, &resting, &refrac) >= 6) {
            if (node_id >= SNN_NUM_INPUTS + SNN_NUM_OUTPUTS) {
                snn_add_hidden(net);
            }
            lif_neuron_t *n = get_neuron(net, node_id);
            if (n) {
                lif_init(n, thresh, leak, reset, resting);
                n->refractory_period = refrac;
            }
        }
    }
    fclose(fn);

    fgets(line, sizeof(line), fs); 
    while (fgets(line, sizeof(line), fs)) {
        int src, tgt, weight;
        if (sscanf(line, "%d,%d,%d", &src, &tgt, &weight) == 3) {
            snn_add_synapse(net, src, tgt, (int8_t)weight);
        }
    }
    fclose(fs);

    return net;
}