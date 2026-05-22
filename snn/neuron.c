#include "neuron.h"

void lif_init(
    lif_neuron_t *neuron,
    int16_t threshold,
    int8_t leak_factor,
    int16_t reset_potential,
    int16_t resting_potential
) {
    neuron->membrane_potential = resting_potential;
    neuron->threshold = threshold;
    neuron->leak_factor = leak_factor;
    neuron->reset_potential = reset_potential;
    neuron->resting_potential = resting_potential;
    neuron->has_fired = 0;
}

void lif_init_default(lif_neuron_t *n)
{
    lif_init(n,
        LIF_DEFAULT_THRESHOLD,
        LIF_DEFAULT_LEAK_FACTOR,
        LIF_DEFAULT_RESET,
        LIF_DEFAULT_RESTING
    );
    // fixed at initialization to avoid bursting behavior/hyperactivity
    n->refractory_period = LIF_DEFAULT_REFRACTORY_PERIOD;
    n->refractory_counter = 0;
}

void lif_reset(lif_neuron_t *neuron) {
    neuron->membrane_potential = neuron->resting_potential;
    neuron->has_fired = 0;
    neuron->refractory_counter = 0;
}

void lif_add_input(lif_neuron_t *neuron, int16_t input_current) {
    neuron->membrane_potential += input_current;
}

void lif_apply_leak(lif_neuron_t *neuron) {
    // apply the rule: v(t+1) = v(t) - (v(t) - v_rest)/τ
    // distance from current membrane voltage to resting potential
    int32_t delta = (int32_t)neuron->membrane_potential - (int32_t)neuron->resting_potential;

    // delta / (2 ^ leak_factor) models exponential decay
    int32_t leakage = delta / (1 << neuron->leak_factor);
    neuron->membrane_potential = (int16_t)( neuron->membrane_potential - leakage);
    
}

uint8_t lif_check_spike(lif_neuron_t *neuron) {
    // check refractory period
    if (neuron->refractory_counter > 0)
    {
        neuron->refractory_counter--;
        neuron->has_fired = 0;
    } else {
        if (neuron->membrane_potential < neuron->threshold)
        {
            neuron->has_fired = 0;
        } else {
            neuron->has_fired = 1;
            neuron->refractory_counter = neuron->refractory_period;
            neuron->membrane_potential = neuron->reset_potential;
            return 1;
        }
    }
    return 0;
}

void lif_step(lif_neuron_t *neuron, int16_t input_current) {
    // absolute/adiabatic refractory period implementation:
    // neuron cannot fire or accumulate input in refractory period
    if (neuron->refractory_counter == 0) {   
        lif_add_input(neuron, input_current);
        lif_apply_leak(neuron);
    }
    lif_check_spike(neuron);
}