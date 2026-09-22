#include "transport.hpp"
#include <iostream>
#include <numeric>

int main() {
    try {
        for (auto scheme : {fluxledger::Scheme::upwind, fluxledger::Scheme::mc}) {
            for (double a : {-1.0, 0.0, 1.0}) {
                for (double nu : {0.0, 0.01}) {
                    std::vector<double> pulse(64, 0.0);
                    std::fill(pulse.begin() + 16, pulse.begin() + 32, 1.0);
                    const auto result = fluxledger::solve(pulse, a, nu, 1.0, 0.3, 1.0, scheme);
                    const double mass =
                        std::accumulate(result.values.begin(), result.values.end(), 0.0);
                    if (std::abs(mass - 16.0) > 1e-10)
                        throw std::runtime_error("mass drift");
                    for (double v : result.values)
                        if (v < -1e-12 || v > 1 + 1e-12)
                            throw std::runtime_error("bound violation");
                }
            }
        }
        bool rejected = false;
        try {
            fluxledger::solve({1, 2, 3, 4}, 1, -1, 1, 1, 0.8, fluxledger::Scheme::mc);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("negative diffusion accepted");
        std::cout << "12 transport cases: conservation and bounds passed; invalid input rejected\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
