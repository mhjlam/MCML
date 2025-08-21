/*==============================================================================
 * MCXYZ - Monte Carlo simulation of photon transport in 3D voxelized media
 * 
 * Input/Output Operations Module
 * 
 * This module handles all file I/O operations including reading simulation
 * configuration files, loading tissue geometry data, and writing simulation
 * results. Provides safe file handling with comprehensive error checking.
 *
 * FEATURES:
 * - Safe file I/O with bounds checking and validation
 * - Comprehensive error reporting with context
 * - Binary file handling for tissue and fluence data
 * - MATLAB-compatible output format generation
 * - Input parameter validation and sanity checking
 *
 * COPYRIGHT:
 * ----------
 * Original work (2010-2017): Steven L. Jacques, Ting Li (Oregon Health & Science University)
 * Modernization (2025): Upgraded to C17 standards with modular architecture
 * 
 * LICENSE:
 * --------
 * GNU General Public License v3.0
 * See mcxyz.h for full copyright and license information.
 */

#include "mcxyz.h"
#include <sys/stat.h>

////////////////////////////////////////////////////////////////////////////////
// FORWARD DECLARATIONS

static McxyzErrorCode load_tissue_volume(const char* basename, SimulationConfig* config);
static McxyzErrorCode write_fluence_data(const SimulationConfig* config);
static McxyzErrorCode write_optical_properties(const SimulationConfig* config);

////////////////////////////////////////////////////////////////////////////////
// FILE I/O HELPER FUNCTIONS

/**
 * Safe string copy with bounds checking
 */
static void safe_strncpy(char* dest, const char* src, size_t dest_size) {
    if (dest && src && dest_size > 0) {
        strncpy(dest, src, dest_size - 1);
        dest[dest_size - 1] = '\0';
    }
}

/**
 * Read a single line from file and parse as float
 */
static McxyzErrorCode read_float_line(FILE* file, float* value, const char* param_name) {
    char buffer[256];
    
    if (!fgets(buffer, sizeof(buffer), file)) {
        fprintf(stderr, "Error: Failed to read %s from input file\n", param_name);
        return MCXYZ_ERROR_FILE_FORMAT;
    }
    
    char* endptr;
    *value = strtof(buffer, &endptr);
    
    if (endptr == buffer) {
        fprintf(stderr, "Error: Invalid format for %s: '%s'\n", param_name, buffer);
        return MCXYZ_ERROR_FILE_FORMAT;
    }
    
    return MCXYZ_SUCCESS;
}

/**
 * Read a single line from file and parse as integer
 */
static McxyzErrorCode read_int_line(FILE* file, int* value, const char* param_name) {
    char buffer[256];
    
    if (!fgets(buffer, sizeof(buffer), file)) {
        fprintf(stderr, "Error: Failed to read %s from input file\n", param_name);
        return MCXYZ_ERROR_FILE_FORMAT;
    }
    
    char* endptr;
    *value = (int)strtol(buffer, &endptr, 10);
    
    if (endptr == buffer) {
        fprintf(stderr, "Error: Invalid format for %s: '%s'\n", param_name, buffer);
        return MCXYZ_ERROR_FILE_FORMAT;
    }
    
    return MCXYZ_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// SIMULATION CONFIGURATION LOADING

/**
 * Load simulation configuration from header file
 */
McxyzErrorCode load_simulation_config(const char* basename, SimulationConfig* config) {
    char filename[MAX_FILENAME_LENGTH];
    FILE* file;
    McxyzErrorCode result;
    
    if (!basename || !config) {
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Initialize configuration
    memset(config, 0, sizeof(SimulationConfig));
    safe_strncpy(config->name, basename, sizeof(config->name));
    
    // Construct header filename
    snprintf(filename, sizeof(filename), "%s_H.mci", basename);
    
    // Open header file
    file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open header file '%s'\n", filename);
        return MCXYZ_ERROR_FILE_NOT_FOUND;
    }
    
    // Read runtime parameters
    result = read_float_line(file, &config->simulation_time_minutes, "simulation time");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    // Read grid dimensions
    result = read_int_line(file, &config->grid.size_x, "Nx");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    result = read_int_line(file, &config->grid.size_y, "Ny");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    result = read_int_line(file, &config->grid.size_z, "Nz");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    // Read voxel spacing
    result = read_float_line(file, &config->grid.spacing_x, "dx");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    result = read_float_line(file, &config->grid.spacing_y, "dy");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    result = read_float_line(file, &config->grid.spacing_z, "dz");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    // Read source configuration
    int source_type_int;
    result = read_int_line(file, &source_type_int, "source type");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    config->source.type = (SourceType)source_type_int;
    
    int launch_flag;
    result = read_int_line(file, &launch_flag, "launch flag");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    config->source.manual_trajectory = (launch_flag == 1);
    
    int boundary_type_int;
    result = read_int_line(file, &boundary_type_int, "boundary type");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    config->source.boundary = (BoundaryType)boundary_type_int;
    
    // Read source position
    result = read_float_line(file, &config->source.position_x, "source position X");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    result = read_float_line(file, &config->source.position_y, "source position Y");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    result = read_float_line(file, &config->source.position_z, "source position Z");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    // Read focus position
    result = read_float_line(file, &config->source.focus_x, "focus X");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    result = read_float_line(file, &config->source.focus_y, "focus Y");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    result = read_float_line(file, &config->source.focus_z, "focus Z");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    // Read manual trajectory (if applicable)
    result = read_float_line(file, &config->source.direction_x, "direction X");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    result = read_float_line(file, &config->source.direction_y, "direction Y");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    result = read_float_line(file, &config->source.direction_z, "direction Z");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    // Read beam parameters
    result = read_float_line(file, &config->source.radius, "radius");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    result = read_float_line(file, &config->source.waist, "waist");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    // Read tissue count
    result = read_int_line(file, &config->tissue_count, "tissue count");
    if (result != MCXYZ_SUCCESS) goto cleanup;
    
    // Validate tissue count
    if (config->tissue_count < 1 || config->tissue_count > MAX_TISSUE_TYPES) {
        fprintf(stderr, "Error: Invalid tissue count %d (must be 1-%d)\n", 
               config->tissue_count, MAX_TISSUE_TYPES);
        result = MCXYZ_ERROR_FILE_FORMAT;
        goto cleanup;
    }
    
    // Read tissue properties
    for (int i = 1; i <= config->tissue_count; i++) {
        result = read_float_line(file, &config->tissues[i].absorption, "tissue absorption");
        if (result != MCXYZ_SUCCESS) goto cleanup;
        
        result = read_float_line(file, &config->tissues[i].scattering, "tissue scattering");
        if (result != MCXYZ_SUCCESS) goto cleanup;
        
        result = read_float_line(file, &config->tissues[i].anisotropy, "tissue anisotropy");
        if (result != MCXYZ_SUCCESS) goto cleanup;
        
        // Set default tissue name
        snprintf(config->tissues[i].name, sizeof(config->tissues[i].name), "Tissue_%d", i);
        
        // Set default refractive index
        config->tissues[i].refractive_index = 1.0f;
    }
    
    // Calculate derived grid properties
    config->grid.total_voxels = config->grid.size_x * config->grid.size_y * config->grid.size_z;
    config->grid.volume_x = config->grid.size_x * config->grid.spacing_x;
    config->grid.volume_y = config->grid.size_y * config->grid.spacing_y;
    config->grid.volume_z = config->grid.size_z * config->grid.spacing_z;
    
    // Set initial target photon count (will be updated during simulation)
    config->target_photon_count = 1000000;
    
    result = MCXYZ_SUCCESS;

cleanup:
    fclose(file);
    
    if (result == MCXYZ_SUCCESS) {
        // Load tissue volume data
        result = load_tissue_volume(basename, config);
    }
    
    return result;
}

/**
 * Load binary tissue volume data
 */
static McxyzErrorCode load_tissue_volume(const char* basename, SimulationConfig* config) {
    char filename[MAX_FILENAME_LENGTH];
    FILE* file;
    size_t bytes_read;
    
    if (!basename || !config) {
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Construct tissue filename
    snprintf(filename, sizeof(filename), "%s_T.bin", basename);
    
    // Allocate tissue volume array
    size_t volume_size = config->grid.total_voxels * sizeof(uint8_t);
    config->tissue_volume = (uint8_t*)calloc(config->grid.total_voxels, sizeof(uint8_t));
    if (!config->tissue_volume) {
        fprintf(stderr, "Error: Failed to allocate memory for tissue volume (%zu bytes)\n", volume_size);
        return MCXYZ_ERROR_MEMORY_ALLOCATION;
    }
    
    // Open tissue file
    file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open tissue file '%s'\n", filename);
        free(config->tissue_volume);
        config->tissue_volume = NULL;
        return MCXYZ_ERROR_FILE_NOT_FOUND;
    }
    
    // Read tissue data
    bytes_read = fread(config->tissue_volume, sizeof(uint8_t), config->grid.total_voxels, file);
    fclose(file);
    
    if (bytes_read != (size_t)config->grid.total_voxels) {
        fprintf(stderr, "Error: Failed to read complete tissue file (expected %d bytes, got %zu)\n", 
               config->grid.total_voxels, bytes_read);
        free(config->tissue_volume);
        config->tissue_volume = NULL;
        return MCXYZ_ERROR_FILE_FORMAT;
    }
    
    return MCXYZ_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// SIMULATION RESULT WRITING

/**
 * Write simulation results to output files
 */
McxyzErrorCode write_simulation_results(const SimulationConfig* config, const PerformanceMetrics* metrics) {
    McxyzErrorCode result;
    
    if (!config || !metrics) {
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Write fluence data
    result = write_fluence_data(config);
    if (result != MCXYZ_SUCCESS) {
        return result;
    }
    
    // Write optical properties
    result = write_optical_properties(config);
    if (result != MCXYZ_SUCCESS) {
        return result;
    }
    
    return MCXYZ_SUCCESS;
}

/**
 * Write binary fluence data file
 */
static McxyzErrorCode write_fluence_data(const SimulationConfig* config) {
    char filename[MAX_FILENAME_LENGTH];
    FILE* file;
    size_t bytes_written;
    
    if (!config || !config->fluence_volume) {
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Construct fluence filename
    snprintf(filename, sizeof(filename), "%s_F.bin", config->name);
    
    // Open output file
    file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot create fluence file '%s'\n", filename);
        return MCXYZ_ERROR_OUTPUT_WRITE;
    }
    
    // Write fluence data
    bytes_written = fwrite(config->fluence_volume, sizeof(float), config->grid.total_voxels, file);
    fclose(file);
    
    if (bytes_written != (size_t)config->grid.total_voxels) {
        fprintf(stderr, "Error: Failed to write complete fluence file (expected %d floats, wrote %zu)\n", 
               config->grid.total_voxels, bytes_written);
        return MCXYZ_ERROR_OUTPUT_WRITE;
    }
    
    printf("Saved: %s (%.1f MB)\n", filename, 
           (config->grid.total_voxels * sizeof(float)) / (1024.0 * 1024.0));
    
    return MCXYZ_SUCCESS;
}

/**
 * Write tissue optical properties in MATLAB format
 */
static McxyzErrorCode write_optical_properties(const SimulationConfig* config) {
    char filename[MAX_FILENAME_LENGTH];
    FILE* file;
    
    if (!config) {
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Construct properties filename
    snprintf(filename, sizeof(filename), "%s_props.m", config->name);
    
    // Open output file
    file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Cannot create properties file '%s'\n", filename);
        return MCXYZ_ERROR_OUTPUT_WRITE;
    }
    
    // Write MATLAB header
    fprintf(file, "%% Tissue optical properties for %s\n", config->name);
    fprintf(file, "%% Generated by MCXYZ v2.0.0\n");
    fprintf(file, "%% Tissue properties: [absorption, scattering, anisotropy]\n\n");
    
    // Write tissue properties
    for (int i = 1; i <= config->tissue_count; i++) {
        fprintf(file, "muav(%d) = %.4f;\n", i, config->tissues[i].absorption);
        fprintf(file, "musv(%d) = %.4f;\n", i, config->tissues[i].scattering);
        fprintf(file, "gv(%d) = %.4f;\n\n", i, config->tissues[i].anisotropy);
    }
    
    // Write grid information
    fprintf(file, "%% Grid information\n");
    fprintf(file, "Nx = %d;\n", config->grid.size_x);
    fprintf(file, "Ny = %d;\n", config->grid.size_y);
    fprintf(file, "Nz = %d;\n", config->grid.size_z);
    fprintf(file, "dx = %.6f;\n", config->grid.spacing_x);
    fprintf(file, "dy = %.6f;\n", config->grid.spacing_y);
    fprintf(file, "dz = %.6f;\n", config->grid.spacing_z);
    
    fclose(file);
    
    // Get file size for reporting
    struct stat st;
    long file_size = 0;
    if (stat(filename, &st) == 0) {
        file_size = st.st_size;
    }
    
    printf("Saved: %s (%.1f KB)\n", filename, file_size / 1024.0);
    
    return MCXYZ_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// INPUT VALIDATION FUNCTIONS

/**
 * Validate simulation configuration parameters
 */
McxyzErrorCode validate_simulation_config(const SimulationConfig* config) {
    if (!config) {
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Validate simulation time
    if (config->simulation_time_minutes <= 0.0f || config->simulation_time_minutes > 1440.0f) {
        fprintf(stderr, "Error: Invalid simulation time %.2f minutes (must be 0-1440)\n", 
               config->simulation_time_minutes);
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Validate grid dimensions
    if (config->grid.size_x <= 0 || config->grid.size_x > 10000 ||
        config->grid.size_y <= 0 || config->grid.size_y > 10000 ||
        config->grid.size_z <= 0 || config->grid.size_z > 10000) {
        fprintf(stderr, "Error: Invalid grid dimensions %d×%d×%d (must be 1-10000)\n",
               config->grid.size_x, config->grid.size_y, config->grid.size_z);
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Validate voxel spacing
    if (config->grid.spacing_x <= 0.0f || config->grid.spacing_x > 1.0f ||
        config->grid.spacing_y <= 0.0f || config->grid.spacing_y > 1.0f ||
        config->grid.spacing_z <= 0.0f || config->grid.spacing_z > 1.0f) {
        fprintf(stderr, "Error: Invalid voxel spacing %.6f×%.6f×%.6f cm (must be 0-1)\n",
               config->grid.spacing_x, config->grid.spacing_y, config->grid.spacing_z);
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Validate source type
    if (config->source.type < SOURCE_UNIFORM || config->source.type > SOURCE_RECTANGULAR) {
        fprintf(stderr, "Error: Invalid source type %d (must be 0-3)\n", (int)config->source.type);
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Validate boundary type
    if (config->source.boundary < BOUNDARY_INFINITE || config->source.boundary > BOUNDARY_ESCAPE_SURFACE) {
        fprintf(stderr, "Error: Invalid boundary type %d (must be 0-2)\n", (int)config->source.boundary);
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Validate tissue count
    if (config->tissue_count < 1 || config->tissue_count > MAX_TISSUE_TYPES) {
        fprintf(stderr, "Error: Invalid tissue count %d (must be 1-%d)\n", 
               config->tissue_count, MAX_TISSUE_TYPES);
        return MCXYZ_ERROR_INVALID_PARAMETER;
    }
    
    // Validate tissue properties
    for (int i = 1; i <= config->tissue_count; i++) {
        if (config->tissues[i].absorption < 0.0f || config->tissues[i].absorption > 1000.0f) {
            fprintf(stderr, "Error: Invalid absorption coefficient %.4f for tissue %d (must be 0-1000)\n",
                   config->tissues[i].absorption, i);
            return MCXYZ_ERROR_INVALID_PARAMETER;
        }
        
        if (config->tissues[i].scattering < 0.0f || config->tissues[i].scattering > 10000.0f) {
            fprintf(stderr, "Error: Invalid scattering coefficient %.4f for tissue %d (must be 0-10000)\n",
                   config->tissues[i].scattering, i);
            return MCXYZ_ERROR_INVALID_PARAMETER;
        }
        
        if (config->tissues[i].anisotropy < -1.0f || config->tissues[i].anisotropy > 1.0f) {
            fprintf(stderr, "Error: Invalid anisotropy %.4f for tissue %d (must be -1 to 1)\n",
                   config->tissues[i].anisotropy, i);
            return MCXYZ_ERROR_INVALID_PARAMETER;
        }
    }
    
    return MCXYZ_SUCCESS;
}

/**
 * Print tissue profile along central z-axis for visualization
 */
void print_tissue_profile(const SimulationConfig* config) {
    if (!config || !config->tissue_volume) {
        return;
    }
    
    int center_x = config->grid.size_x / 2;
    int center_y = config->grid.size_y / 2;
    
    printf("Central tissue profile (z-axis):\n");
    
    for (int z = 0; z < config->grid.size_z && z < 100; z++) {  // Limit to 100 chars
        int index = get_voxel_index(center_x, center_y, z, &config->grid);
        printf("%d", config->tissue_volume[index]);
    }
    
    if (config->grid.size_z > 100) {
        printf("...");
    }
    
    printf("\n\n");
}
