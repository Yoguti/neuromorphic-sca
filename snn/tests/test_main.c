#include "testbenches.h"
#include "libs/dataset.h"
#include "EONS/engine.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {    
    engine_init();
    
    const char *h5_path = "../modulated_dataset/ascad_modulated.h5";
    
    const char *network_base = "network-csvs/best_network";

    if (argc > 1) {
        network_base = argv[1];
    }

    run_test(h5_path, network_base);

    return 0;
}