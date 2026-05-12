# Send-On-Delta Modulation Implementation

def delta_modulation(trace, threshold):
    reference = int(trace[0])
    for sample in trace:
        diff = int(sample) - reference
        
        if diff >= threshold:
            yield 1
            reference += threshold
        elif diff <= -threshold:
            yield -1
            reference -= threshold
        else:
            yield 0

def rebuild_trace(modulated_trace, init_val, threshold):
    current_value = int(init_val)

    for pulse in modulated_trace:
        if pulse == 1:
            current_value += threshold
        elif pulse == -1:
            current_value -= threshold
        yield current_value