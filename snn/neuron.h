#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#define LIF_DEFAULT_THRESHOLD        -50   // Typical spiking threshold (~ -50mV in biological LIF models)
#define LIF_DEFAULT_RESTING          -65   // Resting membrane potential (~ -65mV biological baseline)
#define LIF_DEFAULT_RESET            -70   // Reset after spike (slightly below resting for refractoriness)
#define LIF_DEFAULT_LEAK_FACTOR      4     // Moderate decay
#define LIF_DEFAULT_REFRACTORY_PERIOD 2    // 2 steps of inactivity after spike


typedef struct {
    int16_t membrane_potential; // Current membrane voltage/state
    int16_t threshold;          // Spike threshold

    int8_t leak_factor;         // Leak strength as right-shift factor:
                                // v += (resting_potential - v) >> leak_factor

    int16_t reset_potential;    // Membrane potential after spike
    int16_t resting_potential;  // Natural resting voltage
    uint8_t has_fired;          // 1 if neuron fired on current step

    uint8_t refractory_period;   // ammount of inactive steps after a spike
    uint8_t refractory_counter;  // count down of inactive steps after a spike
} lif_neuron_t;

void lif_init(
    lif_neuron_t *neuron,
    int16_t threshold,
    int8_t leak_factor,
    int16_t reset_potential,
    int16_t resting_potential
);

void lif_init_default(lif_neuron_t *n);

void lif_reset(lif_neuron_t *neuron);

void lif_add_input(lif_neuron_t *neuron, int16_t input_current);

void lif_apply_leak(lif_neuron_t *neuron);

uint8_t lif_check_spike(lif_neuron_t *neuron);

void lif_step(lif_neuron_t *neuron, int16_t input_current);
