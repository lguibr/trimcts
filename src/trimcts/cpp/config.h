#pragma once

#include <cstdint> // For uint32_t etc.

namespace trimcts
{

    // Matches the Python SearchConfiguration Pydantic model
    struct SearchConfig
    {
        uint32_t max_simulations = 50;
        uint32_t max_depth = 10;
        double cpuct = 1.25;
        double dirichlet_alpha = 0.3;
        double dirichlet_epsilon = 0.25;
        double discount = 1.0;
        uint32_t mcts_batch_size = 1; 
        // Add other fields as needed
    };

} // namespace trimcts