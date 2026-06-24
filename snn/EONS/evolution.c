#include "evolution.h"
#include <stdio.h>
#include <stdlib.h>

eons_params_t eons_default_params(void) {
    eons_params_t p;
    p.crossover_rate    = 0.50f;
    p.merge_rate        = 0.00f;
    p.mutation_rate     = 0.90f;
    p.num_mutations     = 7;
    p.num_mutations_min = 2;
    p.add_node_rate     = 0.09f;
    p.delete_node_rate  = 0.07f;
    p.add_edge_rate     = 0.16f;
    p.delete_edge_rate  = 0.14f;
    p.node_param_rate   = 0.27f;
    p.edge_param_rate   = 0.27f;
    p.tournament_size   = 4;
    p.tournament_p      = 0.75f;
    p.random_factor     = 0.10f;
    p.num_best          = 3;
    p.population_size   = POPULATION_SIZE;
    p.num_generations   = 200;
    return p;
}

static float rand_unit(void) { return (float)rand() / (float)RAND_MAX; }

static int rand_range(int lo, int hi) {
    if (hi <= lo) return lo;
    return lo + rand() % (hi - lo + 1);
}

static int annealed_mutations(const eons_params_t *p, int gen) {
    if (p->num_generations <= 1) return p->num_mutations;
    float t = (float)gen / (float)(p->num_generations - 1);
    if (t > 1.0f) t = 1.0f;
    int r = (int)((float)p->num_mutations
                  + t * (float)(p->num_mutations_min - p->num_mutations) + 0.5f);
    return (r < 1) ? 1 : r;
}

static int cmp_fitness(const void *a, const void *b) {
    float fa = ((const candidate_t *)a)->fitness_score;
    float fb = ((const candidate_t *)b)->fitness_score;
    return (fa > fb) ? -1 : (fa < fb) ? 1 : 0;
}

static int tournament(candidate_t *pop, int n, const eons_params_t *p) {
    int b = -1, s = -1;
    float bf = -1e30f, sf = -1e30f;
    for (int i = 0; i < p->tournament_size; i++) {
        int   idx = rand_range(0, n - 1);
        float f   = pop[idx].fitness_score;
        if      (f > bf) { sf = bf; s = b; bf = f; b = idx; }
        else if (f > sf) { sf = f; s = idx; }
    }
    if (s != -1 && rand_unit() > p->tournament_p) return s;
    return b;
}

static void mutate_one(snn_network_t *net, const eons_params_t *p) {
    float sum = p->add_node_rate + p->delete_node_rate +
                p->add_edge_rate + p->delete_edge_rate +
                p->node_param_rate + p->edge_param_rate;
    if (sum <= 0.0f) return;
    float r = rand_unit() * sum;

    if (r < p->add_node_rate) {
        uint16_t nid = snn_add_hidden(net);
        if (nid != UINT16_MAX) {
            uint16_t src = (uint16_t)rand_range(0, SNN_NUM_INPUTS - 1);
            uint16_t tgt = (uint16_t)(SNN_NUM_INPUTS + rand_range(0, net->num_outputs - 1));
            snn_add_synapse(net, src, nid, (int8_t)rand_range(-127, 127));
            snn_add_synapse(net, nid, tgt, (int8_t)rand_range(-127, 127));
        }
        return;
    }
    r -= p->add_node_rate;

    if (r < p->delete_node_rate) {
        if (net->num_hidden > 0)
            snn_delete_hidden(net, SNN_NUM_INPUTS + net->num_outputs +
                              rand_range(0, net->num_hidden - 1));
        return;
    }
    r -= p->delete_node_rate;

    if (r < p->add_edge_rate) {
        uint16_t total = (uint16_t)(SNN_NUM_INPUTS + net->num_outputs + net->num_hidden);
        uint16_t src   = (uint16_t)rand_range(0, total - 1);
        uint16_t tgt   = (uint16_t)(SNN_NUM_INPUTS +
                          rand_range(0, net->num_outputs + net->num_hidden - 1));
        snn_add_synapse(net, src, tgt, (int8_t)rand_range(-127, 127));
        return;
    }
    r -= p->add_edge_rate;

    if (r < p->delete_edge_rate) {
        if (net->num_synapses > 0)
            snn_delete_synapse(net, rand_range(0, net->num_synapses - 1));
        return;
    }
    r -= p->delete_edge_rate;

    if (r < p->node_param_rate) {
        uint16_t nlif = net->num_outputs + net->num_hidden;
        if (nlif > 0) {
            lif_neuron_t *n = &net->nodes[rand_range(0, nlif - 1)];
            switch (rand_range(0, 3)) {
                case 0: n->threshold += (int16_t)rand_range(-10, 10); break;
                case 1: n->leak_factor += (int8_t)rand_range(-1, 1);
                        if (n->leak_factor < 1) n->leak_factor = 1;
                        if (n->leak_factor > 7) n->leak_factor = 7;
                        break;
                case 2: n->reset_potential   += (int16_t)rand_range(-5, 5); break;
                case 3: n->resting_potential += (int16_t)rand_range(-5, 5); break;
            }
        }
        return;
    }

    if (net->num_synapses > 0) {
        uint16_t si = rand_range(0, net->num_synapses - 1);
        int w = net->synapses[si].weight + rand_range(-20, 20);
        if (w >  127) w =  127;
        if (w < -127) w = -127;
        net->synapses[si].weight = (int8_t)w;
    }
}

static void crossover(snn_network_t *child, const snn_network_t *donor) {
    if (donor->num_hidden == 0) return;
    int max_n = (donor->num_hidden < 3) ? donor->num_hidden : 3;
    int num   = rand_range(1, max_n);

    for (int i = 0; i < num; i++) {
        int      di    = rand_range(0, donor->num_hidden - 1);
        uint16_t d_nid = SNN_NUM_INPUTS + donor->num_outputs + (uint16_t)di;
        uint16_t d_lif = donor->num_outputs + (uint16_t)di;

        uint16_t t_nid, t_lif;
        if (child->num_hidden > 0 && rand_unit() < 0.5f) {
            int ci = rand_range(0, child->num_hidden - 1);
            t_nid  = SNN_NUM_INPUTS + child->num_outputs + (uint16_t)ci;
            t_lif  = child->num_outputs + (uint16_t)ci;
        } else {
            t_nid = snn_add_hidden(child);
            if (t_nid == UINT16_MAX) return;
            t_lif = child->num_outputs + (child->num_hidden - 1);
        }

        child->nodes[t_lif] = donor->nodes[d_lif];
        child->nodes[t_lif].membrane_potential = child->nodes[t_lif].resting_potential;
        child->nodes[t_lif].has_fired          = 0;
        child->nodes[t_lif].refractory_counter = 0;

        uint16_t boundary = SNN_NUM_INPUTS + donor->num_outputs;
        for (uint16_t s = 0; s < donor->num_synapses; s++) {
            uint16_t src = donor->synapses[s].source_node;
            uint16_t tgt = donor->synapses[s].target_node;
            int8_t   w   = donor->synapses[s].weight;
            if (tgt == d_nid && src < boundary) snn_add_synapse(child, src, t_nid, w);
            if (src == d_nid && tgt < boundary) snn_add_synapse(child, t_nid, tgt, w);
        }
    }
}

void eons_do_epoch(candidate_t *current, candidate_t *next, Arena *next_arena,
                   const eons_params_t *params, int generation) {
    int pop     = params->population_size;
    int max_mut = annealed_mutations(params, generation);

    candidate_t *ranked = malloc(pop * sizeof(candidate_t));
    for (int i = 0; i < pop; i++) ranked[i] = current[i];
    qsort(ranked, pop, sizeof(candidate_t), cmp_fitness);

    int nc = 0;

    for (int i = 0; i < params->num_best && nc < pop; i++) {
        snn_network_t *n = snn_clone_into(next_arena, ranked[i].network);
        if (!n) break;
        next[nc].network       = n;
        next[nc].fitness_score = ranked[i].fitness_score;
        nc++;
    }
    free(ranked);

    int nr = (int)(params->random_factor * (float)pop);
    for (int i = 0; i < nr && nc < pop; i++) {
        snn_network_t *n = snn_create(next_arena);
        for (int e = 0; e < rand_range(1, 5); e++) mutate_one(n, params);
        next[nc].network       = n;
        next[nc].fitness_score = 0.0f;
        nc++;
    }

    while (nc < pop) {
        int pi = tournament(current, pop, params);
        snn_network_t *child = snn_clone_into(next_arena, current[pi].network);
        if (!child) break;

        if (rand_unit() < params->crossover_rate) {
            int di = tournament(current, pop, params);
            crossover(child, current[di].network);
        }

        if (rand_unit() < params->mutation_rate) {
            int nm = rand_range(1, max_mut);
            for (int m = 0; m < nm; m++) mutate_one(child, params);
        }

        next[nc].network       = child;
        next[nc].fitness_score = 0.0f;
        nc++;
    }
}
