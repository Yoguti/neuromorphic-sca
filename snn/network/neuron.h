#ifndef NEURON_H
#define NEURON_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#define LIF_DEFAULT_THRESHOLD         20
#define LIF_DEFAULT_RESTING           0
#define LIF_DEFAULT_RESET            -5
#define LIF_DEFAULT_LEAK_FACTOR       3
#define LIF_DEFAULT_REFRACTORY_PERIOD 2

typedef struct {
    int16_t membrane_potential;
    int16_t threshold;
    int8_t  leak_factor;
    int16_t reset_potential;
    int16_t resting_potential;
    uint8_t has_fired;
    uint8_t refractory_period;
    uint8_t refractory_counter;
} lif_neuron_t;

void lif_init(lif_neuron_t *neuron, int16_t threshold, int8_t leak_factor,
              int16_t reset_potential, int16_t resting_potential);
void lif_init_default(lif_neuron_t *n);
void lif_reset(lif_neuron_t *neuron);
void lif_step(lif_neuron_t *neuron, int32_t input_current);

#endif
