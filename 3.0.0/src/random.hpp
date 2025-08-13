#pragma once

#include <random>
#include <vector>
#include <cstdint>
#include <sstream>


// Random number generator
class Random
{
private:
    std::mt19937 rng; // Mersenne Twister PRNG
    std::uniform_real_distribution<double> dist; // Uniform distribution [0,1)

public:
    // Initializes with a seed
    explicit Random(long seed = 0) : rng(seed), dist(0.0, 1.0) {}

    // Re-seed the generator
    void seed(long new_seed)
    {
        rng.seed(new_seed);
    }

    // Generate a random number in [0,1)
    double next()
    {
        return dist(rng);
    }

    // Save the current state of the generator
    std::vector<std::uint32_t> state() const
    {
        std::ostringstream oss;
        oss << rng; // Serialize generator state

        std::vector<std::uint32_t> status;
        std::istringstream iss(oss.str());
        std::copy(std::istream_iterator<std::uint32_t>(iss),
                  std::istream_iterator<std::uint32_t>(),
                  std::back_inserter(status));

        return status;
    }

    // Restore generator state
    void restore_state(const std::vector<std::uint32_t>& status)
    {
        std::ostringstream oss;
        for (std::uint32_t value : status) {
            oss << value << " ";
        }

        std::istringstream iss(oss.str());
        iss >> rng; // Deserialize back into mt19937
    }
};
