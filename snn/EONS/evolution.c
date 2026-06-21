#include "evolution.h"
#include <stdio.h>
#include <stdlib.h>

eons_params_t eons_default_params(void)
{
    eons_params_t p;

    p.crossover_rate   = 0.0f;
    p.merge_rate       = 0.0f;
    p.mutation_rate    = 0.75f;
    // mutation count anneals linearly from num_mutations (gen 0) down to
    // num_mutations_min (final generation): broad exploration early,
    // conservative refinement once the search has found useful structure.
    p.num_mutations     = 7;
    p.num_mutations_min = 2;

    p.add_node_rate    = 0.15f;
    p.delete_node_rate = 0.05f;
    p.add_edge_rate    = 0.25f;
    p.delete_edge_rate = 0.10f;
    p.node_param_rate  = 0.20f;
    p.edge_param_rate  = 0.25f;

    p.tournament_size  = 4;
    p.tournament_p     = 0.60f;
    p.random_factor    = 0.15f;
    p.num_best         = 2;

    p.population_size  = 500;
    p.num_generations  = 200;

    return p;
}

static float rand_unit(void) {
    return (float)rand() / (float)RAND_MAX;
}

static int rand_range(int min, int max) {
    if (max <= min) return min;
    return min + rand() % (max - min + 1);
}

// Linearly interpolate the mutation-count ceiling between
// params->num_mutations (generation 0) and params->num_mutations_min
// (generation == num_generations - 1), clamped to that range. With
// num_generations <= 1 or num_mutations_min == num_mutations, this is
// just a constant, so annealing is opt-in by construction.
static int annealed_num_mutations(const eons_params_t *params, int generation) {
    if (params->num_generations <= 1) return params->num_mutations;

    float t = (float)generation / (float)(params->num_generations - 1);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    float interpolated = (float)params->num_mutations +
                          t * (float)(params->num_mutations_min - params->num_mutations);

    int result = (int)(interpolated + 0.5f); // round to nearest
    if (result < 1) result = 1; // at least one mutation if mutation_rate triggers at all
    return result;
}

static int compare_candidates(const void *a, const void *b) {
    const candidate_t *ca = (const candidate_t *)a;
    const candidate_t *cb = (const candidate_t *)b;
    if (ca->fitness_score > cb->fitness_score) return -1;
    if (ca->fitness_score < cb->fitness_score) return 1;
    return 0;
}

static int tournament_select(candidate_t *pop, int pop_size, const eons_params_t *params) {
    int best_idx = -1;
    float best_fit = -99999.0f;
    int second_idx = -1;
    float second_fit = -99999.0f;

    for (int i = 0; i < params->tournament_size; i++) {
        int idx = rand_range(0, pop_size - 1);
        float fit = pop[idx].fitness_score;
        if (fit > best_fit) {
            second_fit = best_fit;
            second_idx = best_idx;
            best_fit = fit;
            best_idx = idx;
        } else if (fit > second_fit) {
            second_fit = fit;
            second_idx = idx;
        }
    }

    if (second_idx != -1 && rand_unit() > params->tournament_p) {
        return second_idx;
    }
    return best_idx;
}

static void apply_one_mutation(snn_network_t *net, const eons_params_t *p) {
    float sum = p->add_node_rate + p->delete_node_rate +
                p->add_edge_rate + p->delete_edge_rate +
                p->node_param_rate + p->edge_param_rate;
    if (sum <= 0.0f) return;

    float r = rand_unit() * sum;

    if (r < p->add_node_rate) {
        snn_add_hidden(net);
        return;
    }
    r -= p->add_node_rate;

    if (r < p->delete_node_rate) {
        if (net->num_hidden > 0) {
            uint16_t node_to_delete = SNN_NUM_INPUTS + net->num_outputs + rand_range(0, net->num_hidden - 1);
            snn_delete_hidden(net, node_to_delete);
        }
        return;
    }
    r -= p->delete_node_rate;

    if (r < p->add_edge_rate) {
        uint16_t total_nodes = SNN_NUM_INPUTS + net->num_outputs + net->num_hidden;
        uint16_t src = rand_range(0, total_nodes - 1);
        uint16_t tgt = SNN_NUM_INPUTS + rand_range(0, net->num_outputs + net->num_hidden - 1);
        int8_t weight = (int8_t)rand_range(-127, 127);
        snn_add_synapse(net, src, tgt, weight);
        return;
    }
    r -= p->add_edge_rate;

    if (r < p->delete_edge_rate) {
        if (net->num_synapses > 0) {
            uint16_t syn_idx = rand_range(0, net->num_synapses - 1);
            snn_delete_synapse(net, syn_idx);
        }
        return;
    }
    r -= p->delete_edge_rate;

    if (r < p->node_param_rate) {
        uint16_t total_lif = net->num_outputs + net->num_hidden;
        if (total_lif > 0) {
            uint16_t lif_idx = rand_range(0, total_lif - 1);
            lif_neuron_t *n = &net->nodes[lif_idx];
            int param_choice = rand_range(0, 3);
            switch (param_choice) {
                case 0: n->threshold += rand_range(-10, 10); break;
                case 1: n->leak_factor += rand_range(-1, 1); if(n->leak_factor < 1) n->leak_factor = 1; break;
                case 2: n->reset_potential += rand_range(-5, 5); break;
                case 3: n->resting_potential += rand_range(-5, 5); break;
            }
        }
        return;
    }

    if (net->num_synapses > 0) {
        uint16_t syn_idx = rand_range(0, net->num_synapses - 1);
        net->synapses[syn_idx].weight += (int8_t)rand_range(-10, 10);
    }
}

void eons_do_epoch(candidate_t *current, candidate_t *next, Arena *next_arena, const eons_params_t *params, int generation) {
    int pop_size = params->population_size;
    int max_mutations = annealed_num_mutations(params, generation);

    candidate_t *ranked = malloc(pop_size * sizeof(candidate_t));
    for (int i = 0; i < pop_size; i++) {
        ranked[i] = current[i];
    }
    qsort(ranked, pop_size, sizeof(candidate_t), compare_candidates);

    int next_count = 0;

    for (int i = 0; i < params->num_best && next_count < pop_size; i++) {
        snn_network_t *net = snn_clone_into(next_arena, ranked[i].network);
        if (!net) break;
        next[next_count].network = net;
        next[next_count].fitness_score = ranked[i].fitness_score;
        next_count++;
    }
    free(ranked);

    int num_random = (int)(params->random_factor * (float)pop_size);
    for (int i = 0; i < num_random && next_count < pop_size; i++) {
        snn_network_t *net = snn_create(next_arena);
        int seed_edges = rand_range(1, 5);
        for (int e = 0; e < seed_edges; e++) {
            apply_one_mutation(net, params);
        }
        next[next_count].network = net;
        next[next_count].fitness_score = 0.0f;
        next_count++;
    }

    while (next_count < pop_size) {
        int parent_idx = tournament_select(current, pop_size, params);
        snn_network_t *child = snn_clone_into(next_arena, current[parent_idx].network);
        if (!child) break;

        if (rand_unit() < params->mutation_rate) {
            int n = rand_range(1, max_mutations);
            for (int m = 0; m < n; m++) {
                apply_one_mutation(child, params);
            }
        }

        next[next_count].network = child;
        next[next_count].fitness_score = 0.0f;
        next_count++;
    }
}
