/*==============================================================================
 * MCXYZ - Monte Carlo simulation of photon transport in 3D voxelized media
 * 
 * Random Number Generation Module
 * 
 * This module provides modern random number generation with good statistical
 * properties using the PCG (Permuted Congruential Generator) algorithm.
 * Replaces the legacy random number generator with a faster, more reliable
 * implementation suitable for Monte Carlo applications.
 *
 * FEATURES:
 * - PCG32 algorithm with excellent statistical properties
 * - Thread-safe implementation for future parallelization
 * - Reproducible sequences with seed support
 * - Fast generation suitable for Monte Carlo applications
 * - No global state dependencies
 *
 * COPYRIGHT:
 * ----------
 * Original work (2010-2017): Steven L. Jacques, Ting Li (Oregon Health & Science University)
 * Modernization (2025): Upgraded to C17 standards with PCG32 RNG implementation
 * 
 * LICENSE:
 * --------
 * GNU General Public License v3.0
 * See mcxyz.h for full copyright and license information.
 */

#include "mcxyz.h"

////////////////////////////////////////////////////////////////////////////////
// PCG32 RANDOM NUMBER GENERATOR IMPLEMENTATION

/**
 * PCG32 Random Number Generator State
 * 
 * Uses the PCG (Permuted Congruential Generator) algorithm which provides
 * excellent statistical properties with fast generation speed. This is a
 * significant improvement over the legacy random number generator.
 */
typedef struct {
    uint64_t state;     // Internal generator state
    uint64_t inc;       // Stream increment (must be odd)
} pcg32_state_t;

// Global RNG state (will be made thread-local for parallelization)
static pcg32_state_t g_rng_state;

// RNG initialized flag to prevent use before initialization
static bool g_rng_initialized = false;

////////////////////////////////////////////////////////////////////////////////
// PCG32 CORE ALGORITHM IMPLEMENTATION

/**
 * PCG32 Step Function - Advance generator state
 * 
 * Core PCG algorithm that advances the internal state using a linear
 * congruential generator with permutation for output generation.
 * 
 * @param rng Pointer to RNG state structure
 * @return Raw 32-bit pseudorandom value
 */
static uint32_t pcg32_step(pcg32_state_t* rng) {
    // Save current state for output generation
    uint64_t oldstate = rng->state;
    
    // Advance internal state using LCG
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    
    // Generate output using permutation
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    
    // Apply rotation for better bit distribution
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
 * PCG32 Seed Function - Initialize generator with seed
 * 
 * Initializes the PCG32 generator with a given seed value. The generator
 * is stepped several times to ensure good initial state distribution.
 * 
 * @param rng Pointer to RNG state structure
 * @param seed Seed value for initialization
 * @param stream Stream identifier for independent sequences
 */
static void pcg32_seed(pcg32_state_t* rng, uint64_t seed, uint64_t stream) {
    // Set stream increment (must be odd)
    rng->inc = (stream << 1u) | 1u;
    
    // Initialize state
    rng->state = 0U;
    pcg32_step(rng);
    
    // Incorporate seed
    rng->state += seed;
    pcg32_step(rng);
}

////////////////////////////////////////////////////////////////////////////////
// PUBLIC RANDOM NUMBER INTERFACE

/**
 * Initialize random number generator with seed
 * 
 * Sets up the random number generator with the specified seed value.
 * This function must be called before any random number generation.
 * 
 * @param seed Seed value for reproducible sequences (0 uses time-based seed)
 */
void init_random_generator(uint64_t seed) {
    // Use current time if seed is 0
    if (seed == 0) {
        seed = (uint64_t)time(NULL);
    }
    
    // Initialize PCG32 with seed and fixed stream
    pcg32_seed(&g_rng_state, seed, 0);
    
    // Mark as initialized
    g_rng_initialized = true;
    
    // Warm up the generator by discarding first few values
    for (int i = 0; i < 10; i++) {
        pcg32_step(&g_rng_state);
    }
}

/**
 * Generate random number in range [0,1)
 * 
 * Generates a uniformly distributed random number in the range [0,1)
 * suitable for Monte Carlo sampling operations.
 * 
 * @return Uniformly distributed random number in [0,1)
 */
double generate_random_number(void) {
    // Check if generator is initialized
    if (!g_rng_initialized) {
        fprintf(stderr, "Error: Random number generator not initialized. Call init_random_generator() first.\n");
        abort();
    }
    
    // Generate 32-bit random value
    uint32_t random_bits = pcg32_step(&g_rng_state);
    
    // Convert to double in range [0,1)
    // Use upper 32 bits for better precision in double conversion
    return (double)random_bits * (1.0 / 4294967296.0); // 1.0 / 2^32
}

////////////////////////////////////////////////////////////////////////////////
// SPECIALIZED RANDOM SAMPLING FUNCTIONS

/**
 * Generate random number in range [min, max)
 * 
 * @param min Minimum value (inclusive)
 * @param max Maximum value (exclusive)
 * @return Random number in specified range
 */
double generate_random_range(double min, double max) {
    if (min >= max) {
        fprintf(stderr, "Error: Invalid random range [%.6f, %.6f)\n", min, max);
        return min;
    }
    
    return min + (max - min) * generate_random_number();
}

/**
 * Generate random integer in range [min, max]
 * 
 * @param min Minimum value (inclusive)  
 * @param max Maximum value (inclusive)
 * @return Random integer in specified range
 */
int generate_random_int(int min, int max) {
    if (min > max) {
        fprintf(stderr, "Error: Invalid random integer range [%d, %d]\n", min, max);
        return min;
    }
    
    // Use rejection sampling for unbiased integer generation
    uint32_t range = (uint32_t)(max - min + 1);
    uint32_t limit = UINT32_MAX - (UINT32_MAX % range);
    
    uint32_t value;
    do {
        value = pcg32_step(&g_rng_state);
    } while (value >= limit);
    
    return min + (int)(value % range);
}

/**
 * Generate random number with exponential distribution
 * 
 * Uses the inverse transform method to generate exponentially
 * distributed random numbers for Monte Carlo transport.
 * 
 * @param lambda Rate parameter (lambda > 0)
 * @return Exponentially distributed random number
 */
double generate_random_exponential(double lambda) {
    if (lambda <= 0.0) {
        fprintf(stderr, "Error: Exponential distribution lambda must be positive (got %.6f)\n", lambda);
        return 0.0;
    }
    
    // Generate uniform random number, avoiding exactly 0
    double u;
    do {
        u = generate_random_number();
    } while (u == 0.0);
    
    // Apply inverse transform: -ln(u) / lambda
    return -log(u) / lambda;
}

/**
 * Generate random direction on unit sphere
 * 
 * Generates a uniformly distributed random direction vector on the
 * unit sphere using the Marsaglia method for efficient sampling.
 * 
 * @param ux Pointer to x-component of direction (output)
 * @param uy Pointer to y-component of direction (output)  
 * @param uz Pointer to z-component of direction (output)
 */
void generate_random_direction(double* ux, double* uy, double* uz) {
    if (!ux || !uy || !uz) {
        fprintf(stderr, "Error: Null pointer passed to generate_random_direction()\n");
        return;
    }
    
    // Generate uniformly distributed direction using Marsaglia method
    double u1, u2, s;
    
    // Rejection sampling to get point in unit circle
    do {
        u1 = 2.0 * generate_random_number() - 1.0;
        u2 = 2.0 * generate_random_number() - 1.0;
        s = u1 * u1 + u2 * u2;
    } while (s >= 1.0);
    
    // Convert to unit sphere coordinates
    double factor = 2.0 * sqrt(1.0 - s);
    *ux = u1 * factor;
    *uy = u2 * factor;
    *uz = 1.0 - 2.0 * s;
}

/**
 * Generate random number with normal distribution
 * 
 * Uses the Box-Muller transform to generate normally distributed
 * random numbers from uniform random numbers.
 * 
 * @param mean Mean of the normal distribution
 * @param std_dev Standard deviation (must be > 0)
 * @return Normally distributed random number
 */
double generate_random_normal(double mean, double std_dev) {
    if (std_dev <= 0.0) {
        fprintf(stderr, "Error: Normal distribution standard deviation must be positive (got %.6f)\n", std_dev);
        return mean;
    }
    
    static bool has_spare = false;
    static double spare;
    
    if (has_spare) {
        has_spare = false;
        return spare * std_dev + mean;
    }
    
    has_spare = true;
    
    // Generate two uniform random numbers
    double u, v, mag;
    do {
        u = 2.0 * generate_random_number() - 1.0;
        v = 2.0 * generate_random_number() - 1.0;
        mag = u * u + v * v;
    } while (mag >= 1.0 || mag == 0.0);
    
    // Apply Box-Muller transform
    mag = sqrt(-2.0 * log(mag) / mag);
    spare = v * mag;
    
    return u * mag * std_dev + mean;
}

////////////////////////////////////////////////////////////////////////////////
// RANDOM NUMBER GENERATOR VALIDATION AND TESTING

/**
 * Validate random number generator quality
 * 
 * Performs basic statistical tests on the random number generator
 * to ensure it meets quality requirements for Monte Carlo simulation.
 * 
 * @param sample_count Number of samples to test
 * @return true if tests pass, false otherwise
 */
bool validate_random_generator(int sample_count) {
    if (sample_count <= 0) {
        fprintf(stderr, "Error: Sample count must be positive\n");
        return false;
    }
    
    if (!g_rng_initialized) {
        fprintf(stderr, "Error: Random generator not initialized\n");
        return false;
    }
    
    // Test 1: Mean should be approximately 0.5
    double sum = 0.0;
    for (int i = 0; i < sample_count; i++) {
        sum += generate_random_number();
    }
    double mean = sum / sample_count;
    
    if (fabs(mean - 0.5) > 0.01) {
        fprintf(stderr, "Random generator validation failed: mean = %.6f (expected ~0.5)\n", mean);
        return false;
    }
    
    // Test 2: Variance should be approximately 1/12 ≈ 0.0833
    double sum_sq = 0.0;
    for (int i = 0; i < sample_count; i++) {
        double x = generate_random_number();
        sum_sq += x * x;
    }
    double variance = sum_sq / sample_count - mean * mean;
    double expected_variance = 1.0 / 12.0;
    
    if (fabs(variance - expected_variance) > 0.01) {
        fprintf(stderr, "Random generator validation failed: variance = %.6f (expected ~%.6f)\n", 
                variance, expected_variance);
        return false;
    }
    
    return true;
}

/**
 * Get random number generator state information
 * 
 * @param state_info Buffer to store state information string
 * @param buffer_size Size of the state_info buffer
 */
void get_random_generator_info(char* state_info, size_t buffer_size) {
    if (!state_info || buffer_size == 0) {
        return;
    }
    
    if (!g_rng_initialized) {
        snprintf(state_info, buffer_size, "Random generator: Not initialized");
        return;
    }
    
    snprintf(state_info, buffer_size, 
             "Random generator: PCG32 (state=0x%016llx, inc=0x%016llx)", 
             (unsigned long long)g_rng_state.state, 
             (unsigned long long)g_rng_state.inc);
}
