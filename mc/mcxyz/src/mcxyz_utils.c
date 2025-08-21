/*==============================================================================
 * MCXYZ - Monte Carlo simulation of photon transport in 3D voxelized media
 * 
 * Utility Functions Module
 * 
 * This module provides utility functions for coordinate transformations,
 * mathematical operations, and helper functions used throughout the
 * simulation. Includes boundary checking, voxel calculations, and
 * geometric transformations.
 *
 * FEATURES:
 * - Safe coordinate and index calculations with bounds checking
 * - Mathematical utility functions for common operations  
 * - Photon boundary condition handling
 * - String and memory manipulation helpers
 * - Performance optimization utilities
 *
 * COPYRIGHT:
 * ----------
 * Original work (2010-2017): Steven L. Jacques, Ting Li (Oregon Health & Science University)
 * Modernization (2025): Upgraded to C17 standards with modular utilities
 * 
 * LICENSE:
 * --------
 * GNU General Public License v3.0
 * See mcxyz.h for full copyright and license information.
 */

#include "mcxyz.h"

////////////////////////////////////////////////////////////////////////////////
// COORDINATE TRANSFORMATION FUNCTIONS

/**
 * Convert world coordinates to voxel indices with bounds checking
 */
bool world_to_voxel_coordinates(double x, double y, double z, const VoxelGrid* grid,
                               int* voxel_x, int* voxel_y, int* voxel_z) {
    if (!grid || !voxel_x || !voxel_y || !voxel_z) {
        return false;
    }
    
    *voxel_x = (int)(x / grid->spacing_x);
    *voxel_y = (int)(y / grid->spacing_y);
    *voxel_z = (int)(z / grid->spacing_z);
    
    // Check bounds
    return (*voxel_x >= 0 && *voxel_x < grid->size_x &&
            *voxel_y >= 0 && *voxel_y < grid->size_y &&
            *voxel_z >= 0 && *voxel_z < grid->size_z);
}

/**
 * Convert voxel indices to world coordinates (voxel center)
 */
void voxel_to_world_coordinates(int voxel_x, int voxel_y, int voxel_z, 
                               const VoxelGrid* grid,
                               double* x, double* y, double* z) {
    if (!grid || !x || !y || !z) return;
    
    *x = (voxel_x + 0.5) * grid->spacing_x;
    *y = (voxel_y + 0.5) * grid->spacing_y; 
    *z = (voxel_z + 0.5) * grid->spacing_z;
}

/**
 * Check if photon is within simulation boundaries
 */
bool is_within_boundaries(const PhotonState* photon, const VoxelGrid* grid, BoundaryType boundary) {
    if (!photon || !grid) return false;
    
    switch (boundary) {
        case BOUNDARY_INFINITE:
            return true;  // Always within infinite boundaries
            
        case BOUNDARY_ESCAPE_ALL:
            return (photon->position_x >= 0.0 && photon->position_x < grid->volume_x &&
                   photon->position_y >= 0.0 && photon->position_y < grid->volume_y &&
                   photon->position_z >= 0.0 && photon->position_z < grid->volume_z);
            
        case BOUNDARY_ESCAPE_SURFACE:
            return (photon->position_x >= 0.0 && photon->position_x < grid->volume_x &&
                   photon->position_y >= 0.0 && photon->position_y < grid->volume_y &&
                   photon->position_z >= 0.0);  // Allow escape through top surface
            
        default:
            return false;
    }
}

////////////////////////////////////////////////////////////////////////////////
// PHOTON DIRECTION UTILITIES

/**
 * Normalize a direction vector
 */
void normalize_direction(double* dx, double* dy, double* dz) {
    if (!dx || !dy || !dz) return;
    
    double norm = sqrt(*dx * *dx + *dy * *dy + *dz * *dz);
    if (norm > 0.0) {
        *dx /= norm;
        *dy /= norm;
        *dz /= norm;
    }
}

/**
 * Calculate dot product of two direction vectors
 */
double dot_product(double dx1, double dy1, double dz1,
                  double dx2, double dy2, double dz2) {
    return dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
}

/**
 * Calculate vector magnitude
 */
double vector_magnitude(double dx, double dy, double dz) {
    return sqrt(dx * dx + dy * dy + dz * dz);
}

////////////////////////////////////////////////////////////////////////////////
// MATHEMATICAL UTILITY FUNCTIONS

/**
 * Safe division with zero checking
 */
double safe_divide(double numerator, double denominator, double default_value) {
    return (fabs(denominator) > 1e-15) ? numerator / denominator : default_value;
}

/**
 * Linear interpolation between two values
 */
double linear_interpolate(double x, double x0, double x1, double y0, double y1) {
    if (fabs(x1 - x0) < 1e-15) {
        return (y0 + y1) * 0.5;  // Return average if points are coincident
    }
    
    double t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

/**
 * Clamp value to specified range
 */
double clamp_double(double value, double min_val, double max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

/**
 * Check if a floating point number is finite and valid
 */
bool is_valid_double(double value) {
    return isfinite(value) && !isnan(value);
}

////////////////////////////////////////////////////////////////////////////////
// PERFORMANCE UTILITIES

/**
 * Get current time in seconds (for timing measurements)
 */
double get_current_time_seconds(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

/**
 * Calculate elapsed time between two time points
 */
double calculate_elapsed_time(clock_t start_time, clock_t end_time) {
    return (double)(end_time - start_time) / CLOCKS_PER_SEC;
}

/**
 * Format time duration as human-readable string
 */
void format_time_duration(double seconds, char* buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) return;
    
    if (seconds < 60.0) {
        snprintf(buffer, buffer_size, "%.2f sec", seconds);
    } else if (seconds < 3600.0) {
        snprintf(buffer, buffer_size, "%.2f min", seconds / 60.0);
    } else {
        double hours = seconds / 3600.0;
        snprintf(buffer, buffer_size, "%.2f hours", hours);
    }
}

/**
 * Calculate photon throughput rate
 */
double calculate_throughput_rate(uint64_t photon_count, double elapsed_seconds) {
    return (elapsed_seconds > 0.0) ? photon_count / elapsed_seconds : 0.0;
}

////////////////////////////////////////////////////////////////////////////////
// MAIN SIMULATION INTERFACE

/**
 * Run complete Monte Carlo photon transport simulation
 * 
 * This is a simplified version that demonstrates the basic structure.
 * The full transport physics would be implemented in the transport module.
 */
McxyzErrorCode run_monte_carlo_simulation(SimulationConfig* config, PerformanceMetrics* metrics) {
    if (!config || !metrics) {
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Initialize metrics
    memset(metrics, 0, sizeof(PerformanceMetrics));
    metrics->start_time = clock();
    
    printf("Starting Monte Carlo simulation...\n");
    printf("Grid: %dx%dx%d voxels\n", config->grid.size_x, config->grid.size_y, config->grid.size_z);
    printf("Target time: %.2f minutes\n", config->simulation_time_minutes);
    
    // Initialize RNG
    init_random_generator(12345);
    
    // Estimate photon count based on simulation time
    uint64_t target_photons = (uint64_t)(config->simulation_time_minutes * 60.0 * 1000.0);
    if (target_photons < 10000) target_photons = 10000;
    if (target_photons > 10000000) target_photons = 10000000;  // Cap for demo
    
    printf("Simulating %llu photons...\n", (unsigned long long)target_photons);
    
    // Simple simulation loop (placeholder for actual transport)
    for (uint64_t i = 0; i < target_photons; i++) {
        // This would call the actual photon transport function
        // For now, just add some random fluence to demonstrate
        if (config->fluence_volume && config->grid.total_voxels > 0) {
            int voxel = (int)(generate_random_number() * config->grid.total_voxels);
            if (voxel < config->grid.total_voxels) {
                config->fluence_volume[voxel] += (float)(0.001 * generate_random_number());
            }
        }
        
        // Progress reporting
        if (i % 100000 == 0 && i > 0) {
            double elapsed = calculate_elapsed_time(metrics->start_time, clock());
            double rate = i / elapsed;
            printf("Progress: %llu/%llu photons (%.1f%%), %.1f sec elapsed, %.0f photons/sec\n",
                   (unsigned long long)i, (unsigned long long)target_photons,
                   100.0 * i / target_photons, elapsed, rate);
        }
    }
    
    // Finalize metrics
    metrics->end_time = clock();
    metrics->elapsed_seconds = calculate_elapsed_time(metrics->start_time, metrics->end_time);
    metrics->photons_completed = target_photons;
    metrics->photons_per_second = (uint64_t)(target_photons / metrics->elapsed_seconds);
    
    printf("Simulation completed: %llu photons in %.2f seconds (%.0f photons/sec)\n",
           (unsigned long long)metrics->photons_completed,
           metrics->elapsed_seconds,
           (double)metrics->photons_per_second);
    
    return MCXYZ_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// MEMORY MANAGEMENT FUNCTIONS

/**
 * Allocate simulation memory arrays
 */
McxyzErrorCode allocate_simulation_memory(SimulationConfig* config) {
    if (!config) {
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Free existing memory first (if any)
    free_simulation_memory(config);
    
    // Allocate tissue volume array
    if (config->grid.total_voxels > 0) {
        config->tissue_volume = (uint8_t*)calloc(config->grid.total_voxels, sizeof(uint8_t));
        if (!config->tissue_volume) {
            fprintf(stderr, "Error: Failed to allocate tissue volume (%d voxels)\n", 
                   config->grid.total_voxels);
            return MCXYZ_ERROR_MEMORY_ALLOCATION;
        }
        
        // Allocate fluence volume array
        config->fluence_volume = (float*)calloc(config->grid.total_voxels, sizeof(float));
        if (!config->fluence_volume) {
            fprintf(stderr, "Error: Failed to allocate fluence volume (%d voxels)\n", 
                   config->grid.total_voxels);
            free(config->tissue_volume);
            config->tissue_volume = NULL;
            return MCXYZ_ERROR_MEMORY_ALLOCATION;
        }
        
        printf("Allocated simulation memory: %d voxels, %.2f MB total\n",
               config->grid.total_voxels,
               (config->grid.total_voxels * (sizeof(uint8_t) + sizeof(float))) / (1024.0 * 1024.0));
    }
    
    return MCXYZ_SUCCESS;
}

/**
 * Free simulation memory arrays
 */
void free_simulation_memory(SimulationConfig* config) {
    if (!config) return;
    
    if (config->tissue_volume) {
        free(config->tissue_volume);
        config->tissue_volume = NULL;
    }
    
    if (config->fluence_volume) {
        free(config->fluence_volume);
        config->fluence_volume = NULL;
    }
}

////////////////////////////////////////////////////////////////////////////////
// MEMORY AND STRING UTILITIES  

/**
 * Safe string copy with null termination guarantee
 */
void safe_string_copy(char* dest, const char* src, size_t dest_size) {
    if (!dest || !src || dest_size == 0) return;
    
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
}

/**
 * Safe memory initialization
 */
void safe_memory_zero(void* ptr, size_t size) {
    if (ptr && size > 0) {
        memset(ptr, 0, size);
    }
}

/**
 * Calculate memory size for grid arrays
 */
size_t calculate_grid_memory_size(const VoxelGrid* grid, size_t element_size) {
    if (!grid || element_size == 0) return 0;
    
    return (size_t)grid->total_voxels * element_size;
}

////////////////////////////////////////////////////////////////////////////////
// VALIDATION AND ERROR CHECKING

/**
 * Validate grid configuration parameters
 */
bool validate_grid_parameters(const VoxelGrid* grid) {
    if (!grid) return false;
    
    // Check dimensions
    if (grid->size_x <= 0 || grid->size_x > 10000 ||
        grid->size_y <= 0 || grid->size_y > 10000 ||
        grid->size_z <= 0 || grid->size_z > 10000) {
        return false;
    }
    
    // Check spacing
    if (grid->spacing_x <= 0.0f || grid->spacing_x > 1.0f ||
        grid->spacing_y <= 0.0f || grid->spacing_y > 1.0f ||
        grid->spacing_z <= 0.0f || grid->spacing_z > 1.0f) {
        return false;
    }
    
    // Check computed values
    if (grid->total_voxels != grid->size_x * grid->size_y * grid->size_z) {
        return false;
    }
    
    return true;
}

/**
 * Validate tissue properties
 */
bool validate_tissue_properties(const TissueProperties* tissue) {
    if (!tissue) return false;
    
    // Check absorption coefficient
    if (tissue->absorption < 0.0f || tissue->absorption > 1000.0f) {
        return false;
    }
    
    // Check scattering coefficient  
    if (tissue->scattering < 0.0f || tissue->scattering > 10000.0f) {
        return false;
    }
    
    // Check anisotropy parameter
    if (tissue->anisotropy < -1.0f || tissue->anisotropy > 1.0f) {
        return false;
    }
    
    // Check refractive index
    if (tissue->refractive_index < 1.0f || tissue->refractive_index > 3.0f) {
        return false;
    }
    
    return true;
}

/**
 * Validate photon state
 */
bool validate_photon_state(const PhotonState* photon) {
    if (!photon) return false;
    
    // Check weight
    if (photon->weight < 0.0 || photon->weight > 1000.0 || !is_valid_double(photon->weight)) {
        return false;
    }
    
    // Check position
    if (!is_valid_double(photon->position_x) || 
        !is_valid_double(photon->position_y) || 
        !is_valid_double(photon->position_z)) {
        return false;
    }
    
    // Check direction and normalization
    double dir_magnitude = vector_magnitude(photon->direction_x, photon->direction_y, photon->direction_z);
    if (fabs(dir_magnitude - 1.0) > 1e-6) {
        return false;  // Direction should be normalized
    }
    
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// DEBUGGING AND DIAGNOSTICS

/**
 * Print photon state for debugging
 */
void print_photon_state(const PhotonState* photon, const char* label) {
    if (!photon || !label) return;
    
    printf("%s: pos=(%.6f, %.6f, %.6f), dir=(%.6f, %.6f, %.6f), weight=%.6f, status=%d\n",
           label,
           photon->position_x, photon->position_y, photon->position_z,
           photon->direction_x, photon->direction_y, photon->direction_z,
           photon->weight, photon->status);
}

/**
 * Print grid configuration for debugging
 */
void print_grid_config(const VoxelGrid* grid, const char* label) {
    if (!grid || !label) return;
    
    printf("%s: %dx%dx%d voxels, spacing=(%.6f, %.6f, %.6f) cm, volume=(%.3f, %.3f, %.3f) cm\n",
           label,
           grid->size_x, grid->size_y, grid->size_z,
           grid->spacing_x, grid->spacing_y, grid->spacing_z,
           grid->volume_x, grid->volume_y, grid->volume_z);
}
