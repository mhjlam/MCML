#pragma once

#include <cstdint>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>

/**
 * @brief High-quality random number generator using Mersenne Twister
 * 
 * Provides uniform random numbers in [0,1) with state save/restore functionality
 * for reproducible Monte Carlo simulations.
 */
class Random
{
private:
	std::mt19937 rng;                            // Mersenne Twister PRNG
	std::uniform_real_distribution<double> dist; // Uniform distribution [0,1)

public:
	/** 
	 * @brief Initialize random generator with seed
	 * @param seed Initial seed value (default: 0)
	 */
	explicit Random(long seed = 0) : rng(static_cast<unsigned int>(seed)), dist(0.0, 1.0) {}

	/** 
	 * @brief Re-seed the generator with new seed
	 * @param new_seed New seed value
	 */
	void seed(long new_seed) { rng.seed(static_cast<unsigned int>(new_seed)); }

	/** 
	 * @brief Generate random number in [0,1) interval
	 * @return Random double in [0,1)
	 */
	double next() { return dist(rng); }

	/** 
	 * @brief Save current generator state for later restoration
	 * @return Vector containing serialized generator state
	 */
	std::vector<std::uint32_t> state() const {
		std::ostringstream oss;
		oss << rng; // Serialize generator state

		std::vector<std::uint32_t> status;
		std::istringstream iss(oss.str());
		std::copy(std::istream_iterator<std::uint32_t>(iss), std::istream_iterator<std::uint32_t>(),
				  std::back_inserter(status));

		return status;
	}

	/** 
	 * @brief Restore generator state from previously saved state
	 * @param status Previously saved generator state vector
	 */
	void restore_state(const std::vector<std::uint32_t>& status) {
		std::ostringstream oss;
		for (std::uint32_t value : status) {
			oss << value << " ";
		}

		std::istringstream iss(oss.str());
		iss >> rng; // Deserialize back into mt19937
	}
};
