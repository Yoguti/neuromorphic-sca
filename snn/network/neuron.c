#include "neuron.h"

void lif_init(lif_neuron_t *neuron, int16_t threshold, int8_t leak_factor,
              int16_t reset_potential, int16_t resting_potential) {
    neuron->membrane_potential = resting_potential;
    neuron->threshold          = threshold;
    neuron->leak_factor        = leak_factor;
    neuron->reset_potential    = reset_potential;
    neuron->resting_potential  = resting_potential;
    neuron->has_fired          = 0;
    neuron->refractory_period  = LIF_DEFAULT_REFRACTORY_PERIOD;
    neuron->refractory_counter = 0;
}

void lif_init_default(lif_neuron_t *n) {
    lif_init(n, LIF_DEFAULT_THRESHOLD, LIF_DEFAULT_LEAK_FACTOR,
             LIF_DEFAULT_RESET, LIF_DEFAULT_RESTING);
}

void lif_reset(lif_neuron_t *neuron) {
    neuron->membrane_potential = neuron->resting_potential;
    neuron->has_fired          = 0;
    neuron->refractory_counter = 0;
}

void lif_step(lif_neuron_t *neuron, int32_t input_current) {
    if (neuron->refractory_counter > 0) {
        neuron->refractory_counter--;
        neuron->has_fired = 0;
        return;
    }

    int32_t v = (int32_t)neuron->membrane_potential + input_current;
    if (v >  32767) v =  32767;
    if (v < -32768) v = -32768;
    neuron->membrane_potential = (int16_t)v;

    int32_t delta = (int32_t)neuron->membrane_potential - (int32_t)neuron->resting_potential;
    int32_t leak  = delta >> neuron->leak_factor;
    if (delta != 0 && leak == 0) leak = (delta > 0) ? 1 : -1;
    neuron->membrane_potential = (int16_t)(neuron->membrane_potential - leak);

    if (neuron->membrane_potential >= neuron->threshold) {
        neuron->has_fired          = 1;
        neuron->refractory_counter = neuron->refractory_period;
        neuron->membrane_potential = neuron->reset_potential;
    } else {
        neuron->has_fired = 0;
    }
}
