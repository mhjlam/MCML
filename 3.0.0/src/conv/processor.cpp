/*******************************************************************************
 *	Basic implementation for processor class
 *  Copyright M.H.J. Lam, 2025.
 ****/

#include "processor.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <numbers>

namespace conv {

// ProcessorConfig implementation
bool ProcessorConfig::is_valid() const noexcept {
    if (mode == ProcessingMode::Batch && input_filename.empty()) {
        return false;
    }
    return beam_params.is_valid() && conv_params.is_valid() && grid_params.is_valid();
}

// ProcessingStats implementation
void ProcessingStats::reset() {
    total_time = std::chrono::duration<double>::zero();
    reading_time = std::chrono::duration<double>::zero();
    convolution_time = std::chrono::duration<double>::zero();
    writing_time = std::chrono::duration<double>::zero();
    bytes_read = 0;
    bytes_written = 0;
    cache_hits = 0;
    cache_misses = 0;
}

std::string ProcessingStats::to_string() const {
    std::ostringstream oss;
    oss << "Processing Statistics:\n";
    oss << "  Total time: " << total_time.count() << " seconds\n";
    oss << "  Reading time: " << reading_time.count() << " seconds\n";
    oss << "  Convolution time: " << convolution_time.count() << " seconds\n";
    oss << "  Writing time: " << writing_time.count() << " seconds\n";
    oss << "  Bytes read: " << bytes_read << "\n";
    oss << "  Bytes written: " << bytes_written << "\n";
    oss << "  Cache hits: " << cache_hits << "\n";
    oss << "  Cache misses: " << cache_misses << "\n";
    return oss.str();
}

// ConvProcessor implementation
ConvProcessor::ConvProcessor()
    : m_reader(std::make_unique<McmlDataReader>())
    , m_beam(std::make_unique<BeamProfile>())
    , m_engine(std::make_unique<ConvolutionEngine>()) {
}

int ConvProcessor::run(const ProcessorConfig& config) {
    try {
        m_current_config = config;
        
        switch (config.mode) {
            case ProcessingMode::Interactive:
                return run_interactive();
            case ProcessingMode::Batch:
                return process_file(config) ? 0 : 1;
            case ProcessingMode::Extract:
                // Handle extraction mode
                return 0;
        }
        
        return 0;
    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
}

int ConvProcessor::run_interactive() {
    show_about();
    
    std::string input;
    bool continue_running = true;
    
    while (continue_running) {
        show_main_menu();
        std::cout << "\nEnter command: ";
        
        if (!std::getline(std::cin, input)) {
            // Handle EOF (Ctrl+C or pipe closure)
            break;
        }
        
        if (!input.empty()) {
            continue_running = handle_menu_command(input[0]);
        }
    }
    
    return 0;
}

int ConvProcessor::run_batch(int argc, char* argv[]) {
    try {
        ProcessorConfig config = parse_arguments(argc, argv);
        return run(config);
    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
}

void ConvProcessor::show_about() const {
    std::cout << "CONV 3.0.0 - Convolution Program for MCML\n";
    std::cout << "Copyright (c) 1992-1996, 2025\n";
    std::cout << "Modern C++20 implementation\n\n";
}

void ConvProcessor::show_main_menu() const {
    std::cout << "\nMain Menu:\n";
    std::cout << "  a = About CONV\n";
    std::cout << "  i = Input MCML output file\n";
    std::cout << "  r = Show reflectance, absorption, and transmittance\n";
    std::cout << "  o = Extract original data\n";
    std::cout << "  b = Specify laser beam\n";
    std::cout << "  e = Specify convolution error\n";
    std::cout << "  c = Extract convolved data\n";
    std::cout << "  q = Quit program\n";
    std::cout << "Commands are not case-sensitive.\n";
}

bool ConvProcessor::handle_menu_command(char command) {
    command = static_cast<char>(std::tolower(static_cast<unsigned char>(command)));
    
    switch (command) {
        case 'a':
            show_about();
            break;
        case 'i':
            input_file_interactive();
            break;
        case 'r':
            show_rat_totals();
            break;
        case 'o':
            extract_original_data();
            break;
        case 'b':
            configure_beam_interactive();
            break;
        case 'e':
            configure_error_interactive();
            break;
        case 'c':
            extract_convolved_data();
            break;
        case 'q':
            if (get_confirmation("Do you really want to exit CONV?")) {
                return false;
            }
            break;
        default:
            std::cout << "Unknown command. Please try again.\n";
    }
    
    return true;
}

bool ConvProcessor::input_file_interactive() {
    std::string filename = get_user_input("Enter MCML output filename (.mco): ");
    
    try {
        m_reader->read_file(filename);
        m_data_loaded = true;
        std::cout << "File loaded successfully.\n";
        return true;
    } catch (const std::exception& e) {
        log_error("Failed to load file: " + std::string(e.what()));
        return true;
    }
}

bool ConvProcessor::configure_beam_interactive() {
    if (!m_data_loaded) {
        std::cout << "No MCML data loaded. Please load a file first (option 'i').\n";
        return true;
    }

    std::cout << "\nBeam Configuration:\n";
    std::cout << "Current beam type: ";
    
    switch (m_current_config.beam_params.type) {
        case conv::BeamType::Original:  std::cout << "Original (point source)\n"; break;
        case conv::BeamType::Flat:      std::cout << "Flat beam (R=" << m_current_config.beam_params.radius << " cm)\n"; break;
        case conv::BeamType::Gaussian:  std::cout << "Gaussian beam (R=" << m_current_config.beam_params.radius << " cm)\n"; break;
        case conv::BeamType::Arbitrary: std::cout << "Arbitrary beam\n"; break;
    }
    
    std::cout << "\nBeam profile options:\n";
    std::cout << "  f = Flat beam\n";
    std::cout << "  g = Gaussian beam\n";
    std::cout << "  a = Arbitrary beam (from file)\n";
    std::cout << "  o = Original (point source)\n";
    std::cout << "  q = Quit beam configuration\n";
    
    std::string choice = get_user_input("Select beam type: ");
    if (choice.empty()) return true;
    
    char beam_choice = static_cast<char>(std::tolower(static_cast<unsigned char>(choice[0])));
    
    switch (beam_choice) {
        case 'f':
            configure_flat_beam();
            break;
        case 'g':
            configure_gaussian_beam();
            break;
        case 'a':
            configure_arbitrary_beam();
            break;
        case 'o':
            m_current_config.beam_params.type = conv::BeamType::Original;
            m_current_config.beam_params.radius = 0.0;
            std::cout << "Set to original (point source) beam.\n";
            break;
        case 'q':
            break;
        default:
            std::cout << "Invalid choice.\n";
    }
    
    return true;
}

bool ConvProcessor::configure_error_interactive() {
    std::cout << "\nConvolution Error Configuration:\n";
    std::cout << "Current relative error tolerance: " << m_current_config.conv_params.epsilon << "\n";
    std::cout << "Recommended range: 0.001 - 0.1\n";
    
    std::string input = get_user_input("Enter new error tolerance (or press Enter to keep current): ");
    if (!input.empty()) {
        try {
            double new_epsilon = std::stod(input);
            if (new_epsilon > 0.0 && new_epsilon <= 1.0) {
                m_current_config.conv_params.epsilon = new_epsilon;
                std::cout << "Error tolerance set to " << new_epsilon << "\n";
            } else {
                std::cout << "Error: Value must be between 0 and 1.\n";
            }
        } catch (const std::exception&) {
            std::cout << "Error: Invalid numeric value.\n";
        }
    }
    
    return true;
}

bool ConvProcessor::show_rat_totals() {
    if (!m_data_loaded) {
        std::cout << "No MCML data loaded. Please load a file first (option 'i').\n";
        return true;
    }

    const auto& output = m_reader->output_data();
    const auto& input = m_reader->input_data();
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "RAT - Reflectance, Absorption, and Transmittance Totals\n";
    std::cout << "File: " << input.filename << "\n";
    std::cout << std::string(60, '=') << "\n";
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Specular reflectance (Rsp): " << std::setw(12) << output.specular_reflectance << "\n";
    std::cout << "Total reflectance    (Rd):  " << std::setw(12) << output.total_reflectance << "\n";
    std::cout << "Total absorption     (A):   " << std::setw(12) << output.total_absorption << "\n";
    std::cout << "Total transmittance  (Td):  " << std::setw(12) << output.total_transmittance << "\n";
    
    // Calculate and show total
    double total = output.specular_reflectance + output.total_reflectance + 
                   output.total_absorption + output.total_transmittance;
    std::cout << std::string(35, '-') << "\n";
    std::cout << "Total:                      " << std::setw(12) << total << "\n";
    
    if (std::abs(total - 1.0) > 0.01) {
        std::cout << "\nWarning: Total does not equal 1.0 (difference: " << (total - 1.0) << ")\n";
    }
    
    std::cout << "\nSimulation parameters:\n";
    std::cout << "  Photons simulated: " << input.num_photons << "\n";
    std::cout << "  Grid spacing: dz=" << input.dz << " cm, dr=" << input.dr << " cm\n";
    std::cout << "  Grid size: nz=" << input.nz << ", nr=" << input.nr << ", na=" << input.na << "\n";
    
    return true;
}

bool ConvProcessor::process_file(const ProcessorConfig& config) {
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Read MCML file
        std::cout << "Reading MCML file: " << config.input_filename << std::endl;
        auto read_start = std::chrono::high_resolution_clock::now();
        
        m_reader->read_file(config.input_filename);
        m_data_loaded = true;
        
        auto read_end = std::chrono::high_resolution_clock::now();
        m_stats.reading_time = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start);
        
        // Display file information
        const auto& input_data = m_reader->input_data();
        const auto& output_data = m_reader->output_data();
        
        std::cout << "File loaded successfully!" << std::endl;
        std::cout << "Grid parameters: nr=" << input_data.nr << ", nz=" << input_data.nz 
                  << ", dr=" << input_data.dr << ", dz=" << input_data.dz << std::endl;
        
        // Setup beam profile
        BeamParameters beam_params = config.beam_params;
        if (beam_params.type == BeamType::Original) {
            // Default to flat beam for batch processing
            beam_params.type = BeamType::Flat;
            beam_params.radius = 0.1; // 1mm default
            beam_params.total_power = 1.0;
        }
        
        m_beam->set_parameters(beam_params);
        std::cout << "Beam configuration: ";
        switch (beam_params.type) {
            case BeamType::Original:  std::cout << "Original"; break;
            case BeamType::Flat:      std::cout << "Flat"; break;
            case BeamType::Gaussian:  std::cout << "Gaussian"; break;
            case BeamType::Arbitrary: std::cout << "Arbitrary"; break;
        }
        std::cout << " (R=" << beam_params.radius << " cm, P=" << beam_params.total_power << " J)" << std::endl;
        
        // Setup grid parameters for output
        GridParameters grid_params = config.grid_params;
        if (grid_params.nr == 0) {
            // Use input grid as default
            grid_params.nr = input_data.nr;
            grid_params.dr = input_data.dr;
        }
        
        // Perform convolution
        std::cout << "Performing convolution..." << std::endl;
        auto conv_start = std::chrono::high_resolution_clock::now();
        
        // Convolve reflectance data
        ConvolutionResult rd_result = m_engine->convolve_reflectance(input_data, output_data, *m_beam, grid_params);
        
        // Convolve transmittance data if available
        ConvolutionResult td_result;
        if (!output_data.td_ra.empty()) {
            td_result = m_engine->convolve_transmittance(input_data, output_data, *m_beam, grid_params);
        }
        
        auto conv_end = std::chrono::high_resolution_clock::now();
        m_stats.convolution_time = std::chrono::duration_cast<std::chrono::microseconds>(conv_end - conv_start);
        
        std::cout << "Convolution completed in " << m_stats.convolution_time.count() / 1000.0 << " ms" << std::endl;
        
        // Write results
        auto write_start = std::chrono::high_resolution_clock::now();
        
        std::string output_filename = config.output_filename;
        if (output_filename.empty()) {
            output_filename = generate_output_filename(config.input_filename, "_convolved");
        }
        
        bool write_success = write_convolution_results(rd_result, output_filename);
        if (!td_result.convolved_data.empty()) {
            std::string td_filename = generate_output_filename(config.input_filename, "_convolved_td");
            write_success &= write_convolution_results(td_result, td_filename);
        }
        
        auto write_end = std::chrono::high_resolution_clock::now();
        m_stats.writing_time = std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_start);
        
        // Update total time
        auto end_time = std::chrono::high_resolution_clock::now();
        m_stats.total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Processing completed successfully!" << std::endl;
        std::cout << "Results written to: " << output_filename << std::endl;
        std::cout << "Total processing time: " << m_stats.total_time.count() / 1000.0 << " ms" << std::endl;
        
        return write_success;
        
    } catch (const std::exception& e) {
        log_error("File processing failed: " + std::string(e.what()));
        return false;
    }
}

ProcessorConfig ConvProcessor::parse_arguments(int argc, char* argv[]) const {
    ProcessorConfig config;
    config.mode = ProcessingMode::Batch;
    
    // Initialize default beam parameters
    config.beam_params.type = BeamType::Original;
    config.beam_params.total_power = 1.0;
    config.beam_params.radius = 0.1; // 1mm default
    
    // Initialize default convolution parameters  
    config.conv_params.epsilon = DEFAULT_EPSILON;
    config.conv_params.max_iterations = 10000;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        }
        else if (arg == "-f" || arg == "--force") {
            config.overwrite_output = true;
        }
        else if (arg == "-i" || arg == "--interactive") {
            config.mode = ProcessingMode::Interactive;
        }
        else if (arg == "-b" || arg == "--batch") {
            config.mode = ProcessingMode::Batch;
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                config.output_filename = argv[++i];
            } else {
                throw ConvException("--output requires a filename argument");
            }
        }
        else if (arg == "--beam-type") {
            if (i + 1 < argc) {
                std::string beam_type = argv[++i];
                if (beam_type == "original") {
                    config.beam_params.type = BeamType::Original;
                } else if (beam_type == "flat") {
                    config.beam_params.type = BeamType::Flat;
                } else if (beam_type == "gaussian") {
                    config.beam_params.type = BeamType::Gaussian;
                } else if (beam_type == "arbitrary") {
                    config.beam_params.type = BeamType::Arbitrary;
                } else {
                    throw ConvException("Invalid beam type: " + beam_type);
                }
            } else {
                throw ConvException("--beam-type requires an argument");
            }
        }
        else if (arg == "--beam-radius") {
            if (i + 1 < argc) {
                config.beam_params.radius = std::stod(argv[++i]);
                if (config.beam_params.radius <= 0.0) {
                    throw ConvException("Beam radius must be positive");
                }
            } else {
                throw ConvException("--beam-radius requires a numeric argument");
            }
        }
        else if (arg == "--beam-power") {
            if (i + 1 < argc) {
                config.beam_params.total_power = std::stod(argv[++i]);
                if (config.beam_params.total_power <= 0.0) {
                    throw ConvException("Beam power must be positive");
                }
            } else {
                throw ConvException("--beam-power requires a numeric argument");
            }
        }
        else if (arg == "--epsilon") {
            if (i + 1 < argc) {
                config.conv_params.epsilon = std::stod(argv[++i]);
                if (config.conv_params.epsilon <= 0.0 || config.conv_params.epsilon >= 1.0) {
                    throw ConvException("Epsilon must be between 0 and 1");
                }
            } else {
                throw ConvException("--epsilon requires a numeric argument");
            }
        }
        else if (arg[0] == '-') {
            throw ConvException("Unknown option: " + arg);
        }
        else {
            // Non-option argument - treat as input filename
            if (config.input_filename.empty()) {
                config.input_filename = arg;
            } else {
                throw ConvException("Multiple input files specified");
            }
        }
    }
    
    // Validate configuration
    if (config.mode == ProcessingMode::Batch && config.input_filename.empty()) {
        throw ConvException("Input filename required for batch mode");
    }
    
    return config;
}

std::string ConvProcessor::get_user_input(const std::string& prompt) const {
    std::cout << prompt;
    std::string input;
    std::getline(std::cin, input);
    return input;
}

bool ConvProcessor::get_confirmation(const std::string& prompt) const {
    std::string response = get_user_input(prompt + " (y/n): ");
    return !response.empty() && (std::tolower(response[0]) == 'y');
}

void ConvProcessor::log_verbose(const std::string& message) const {
    if (m_verbose) {
        std::cout << "INFO: " << message << std::endl;
    }
}

void ConvProcessor::log_error(const std::string& message) const {
    std::cerr << "ERROR: " << message << std::endl;
}

void ConvProcessor::configure_flat_beam() {
    std::cout << "\nFlat Beam Configuration:\n";
    std::cout << "A flat beam has uniform intensity within radius R.\n";
    
    std::string input = get_user_input("Enter beam radius in cm: ");
    try {
        double radius = std::stod(input);
        if (radius > 0.0) {
            m_current_config.beam_params.type = conv::BeamType::Flat;
            m_current_config.beam_params.radius = radius;
            m_current_config.beam_params.total_power = 1.0; // Default power
            std::cout << "Flat beam configured with radius " << radius << " cm.\n";
        } else {
            std::cout << "Error: Radius must be positive.\n";
        }
    } catch (const std::exception&) {
        std::cout << "Error: Invalid radius value.\n";
    }
}

void ConvProcessor::configure_gaussian_beam() {
    std::cout << "\nGaussian Beam Configuration:\n";
    std::cout << "A Gaussian beam has 1/e² radius R (contains ~86.5% of power).\n";
    
    std::string input = get_user_input("Enter 1/e² radius in cm: ");
    try {
        double radius = std::stod(input);
        if (radius > 0.0) {
            m_current_config.beam_params.type = conv::BeamType::Gaussian;
            m_current_config.beam_params.radius = radius;
            m_current_config.beam_params.total_power = 1.0; // Default power
            std::cout << "Gaussian beam configured with 1/e² radius " << radius << " cm.\n";
        } else {
            std::cout << "Error: Radius must be positive.\n";
        }
    } catch (const std::exception&) {
        std::cout << "Error: Invalid radius value.\n";
    }
}

void ConvProcessor::configure_arbitrary_beam() {
    std::cout << "\nArbitrary Beam Configuration:\n";
    std::cout << "Load beam profile from a text file with format:\n";
    std::cout << "  # radius(cm)  power_density(J/cm²)\n";
    std::cout << "  0.0           1.0\n";
    std::cout << "  0.1           0.8\n";
    std::cout << "  ...\n";
    
    std::string filename = get_user_input("Enter profile filename: ");
    if (!filename.empty()) {
        try {
            m_beam->load_arbitrary_profile(filename);
            m_current_config.beam_params.type = conv::BeamType::Arbitrary;
            m_current_config.beam_params.profile_filename = filename;
            std::cout << "Arbitrary beam profile loaded from " << filename << ".\n";
        } catch (const std::exception& e) {
            std::cout << "Error loading profile: " << e.what() << "\n";
        }
    }
}

bool ConvProcessor::extract_original_data() {
    if (!m_data_loaded) {
        std::cout << "No MCML data loaded. Please load a file first (option 'i').\n";
        return true;
    }

    const auto& input = m_reader->input_data();
    const auto& output = m_reader->output_data();
    
    std::cout << "\n=== Original Data Extraction ===\n";
    std::cout << "Available quantities from " << input.filename << ":\n\n";
    
    // Build a comprehensive menu of all available data quantities
    struct DataQuantity {
        std::string code;
        std::string description;
        bool available;
    };
    
    std::vector<DataQuantity> quantities = {
        {"RAT", "Reflectance, Absorption, Transmittance totals", true},
        {"RD_A", "Diffuse reflectance vs angle [1/sr]", !output.rd_a.empty()},
        {"RD_R", "Diffuse reflectance vs radius [1/cm²]", !output.rd_r.empty()},
        {"RD_T", "Diffuse reflectance vs time [1/ps]", !output.rd_t.empty()},
        {"TD_A", "Diffuse transmittance vs angle [1/sr]", !output.td_a.empty()},
        {"TD_R", "Diffuse transmittance vs radius [1/cm²]", !output.td_r.empty()},
        {"TD_T", "Diffuse transmittance vs time [1/ps]", !output.td_t.empty()},
        {"A_Z", "Absorption vs depth [1/cm]", !output.a_z.empty()},
        {"A_T", "Absorption vs time [1/ps]", !output.a_t.empty()},
        {"A_RZ", "Absorption vs (radius,depth) [1/cm³]", !output.a_rz.empty()},
        {"RD_RA", "Diffuse reflectance vs (radius,angle) [1/(sr·cm²)]", !output.rd_ra.empty()},
        {"RD_RT", "Diffuse reflectance vs (radius,time) [1/(cm²·ps)]", !output.rd_rt.empty()},
        {"RD_AT", "Diffuse reflectance vs (angle,time) [1/(sr·ps)]", !output.rd_at.empty()},
        {"TD_RA", "Diffuse transmittance vs (radius,angle) [1/(sr·cm²)]", !output.td_ra.empty()},
        {"TD_RT", "Diffuse transmittance vs (radius,time) [1/(cm²·ps)]", !output.td_rt.empty()},
        {"TD_AT", "Diffuse transmittance vs (angle,time) [1/(sr·ps)]", !output.td_at.empty()}
    };
    
    // Display available quantities
    std::vector<size_t> available_indices;
    int display_num = 1;
    for (size_t i = 0; i < quantities.size(); ++i) {
        if (quantities[i].available) {
            std::cout << std::setw(2) << display_num << ". " 
                      << std::setw(8) << std::left << quantities[i].code 
                      << " - " << quantities[i].description << "\n";
            available_indices.push_back(i);
            display_num++;
        }
    }
    
    if (available_indices.empty()) {
        std::cout << "No extractable data found in the loaded file.\n";
        return true;
    }
    
    std::cout << "\nSelect quantity to extract (number, 'all' for all data, or 'q' to quit): ";
    std::string choice = get_user_input("");
    
    if (choice == "q" || choice.empty()) return true;
    
    if (choice == "all") {
        // Extract all available quantities
        std::cout << "\nExtracting all available data quantities...\n";
        for (size_t idx : available_indices) {
            std::string quantity_code = quantities[idx].code;
            std::string output_filename = generate_output_filename(input.filename, "_" + quantity_code);
            
            if (extract_quantity_to_file(quantity_code, output_filename)) {
                std::cout << "✓ " << quantity_code << " → " << output_filename << "\n";
            } else {
                std::cout << "✗ Failed to extract " << quantity_code << "\n";
            }
        }
        std::cout << "\nData extraction complete.\n";
    } else {
        try {
            int selection = std::stoi(choice);
            if (selection >= 1 && selection <= static_cast<int>(available_indices.size())) {
                size_t quantity_idx = available_indices[static_cast<size_t>(selection - 1)];
                std::string quantity_code = quantities[quantity_idx].code;
                
                std::string output_filename = generate_output_filename(input.filename, "_" + quantity_code);
                
                if (extract_quantity_to_file(quantity_code, output_filename)) {
                    std::cout << "\n✓ Data extracted to: " << output_filename << "\n";
                    std::cout << "  Quantity: " << quantities[quantity_idx].description << "\n";
                } else {
                    std::cout << "\n✗ Failed to extract data.\n";
                }
            } else {
                std::cout << "Invalid selection. Please choose 1-" << available_indices.size() << ".\n";
            }
        } catch (const std::exception&) {
            std::cout << "Invalid input. Please enter a number, 'all', or 'q'.\n";
        }
    }
    
    return true;
}

bool ConvProcessor::extract_convolved_data() {
    if (!m_data_loaded) {
        std::cout << "No MCML data loaded. Please load a file first (option 'i').\n";
        return true;
    }

    if (m_current_config.beam_params.type == conv::BeamType::Original) {
        std::cout << "No incident beam specified. Please configure a beam first (option 'b').\n";
        return true;
    }

    std::cout << "\n=== Convolved Data Extraction ===\n";
    std::cout << "This feature requires Phase 4 (Convolution Engine) implementation.\n";
    std::cout << "\nCurrent beam configuration: ";
    
    switch (m_current_config.beam_params.type) {
        case conv::BeamType::Flat:
            std::cout << "Flat (R=" << m_current_config.beam_params.radius << " cm)\n";
            break;
        case conv::BeamType::Gaussian:
            std::cout << "Gaussian (R=" << m_current_config.beam_params.radius << " cm)\n";
            break;
        case conv::BeamType::Arbitrary:
            std::cout << "Arbitrary (from " << m_current_config.beam_params.profile_filename << ")\n";
            break;
        default:
            std::cout << "Original (no convolution needed)\n";
    }
    
    std::cout << "Error tolerance: " << m_current_config.conv_params.epsilon << "\n";
    
    if (m_current_config.beam_params.type == conv::BeamType::Original) {
        std::cout << "\nNo convolution needed for original beam.\n";
        return true;
    }
    
    std::cout << "\nAvailable convolution operations (Phase 4 required):\n";
    std::cout << "  1. RD_R_CONV - Convolved diffuse reflectance vs radius\n";
    std::cout << "  2. TD_R_CONV - Convolved diffuse transmittance vs radius\n";
    std::cout << "  3. A_RZ_CONV - Convolved absorption vs (radius,depth)\n";
    
    std::string choice = get_user_input("\nSelect operation (1-3 or 'q' to quit): ");
    if (choice == "q") return true;
    
    std::cout << "\nPhase 4 convolution engine implementation needed.\n";
    
    if (get_confirmation("Proceed with convolution")) {
        try {
            perform_convolution_extraction();
        } catch (const std::exception& e) {
            std::cout << "Error during convolution: " << e.what() << "\n";
        }
    }
    
    return true;
}

bool ConvProcessor::extract_quantity_to_file(const std::string& quantity, const std::string& filename) {
    const auto& input = m_reader->input_data();
    const auto& output = m_reader->output_data();
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: Cannot create output file " << filename << "\n";
        return false;
    }
    
    // Write standard header
    file << "# " << quantity << " extracted from " << input.filename << "\n";
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    file << "# Generated by CONV 3.0.0 on " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\n";
    file << "# Grid parameters: dz=" << input.dz << ", dr=" << input.dr << ", dt=" << input.dt << "\n";
    file << "# Dimensions: nz=" << input.nz << ", nr=" << input.nr << ", nt=" << input.nt << ", na=" << input.na << "\n";
    file << "#\n";
    
    // Set precision for scientific output
    file << std::scientific << std::setprecision(6);
    
    if (quantity == "RAT") {
        file << "# RAT -- Reflectance, absorption, transmittance.\n";
        file << "# Average values from Monte Carlo simulation\n";
        file << "#\n";
        file << std::setw(14) << output.specular_reflectance << " \t#Rsp: Specular reflectance.\n";
        file << std::setw(14) << output.total_reflectance << " \t#Rd:  Diffuse reflectance.\n";
        file << std::setw(14) << output.total_absorption << " \t#A:   Absorption.\n";
        file << std::setw(14) << output.total_transmittance << " \t#Td:  Diffuse transmittance.\n";
        
        // Calculate and show totals
        double total = output.specular_reflectance + output.total_reflectance + 
                      output.total_absorption + output.total_transmittance;
        file << "#\n# Total: " << std::setw(14) << total << " (should be ~1.0)\n";
    }
    else if (quantity == "RD_A" && !output.rd_a.empty()) {
        file << "# Rd_a[0], [1],..Rd_a[na-1]. [1/sr]\n";
        file << "# na = " << output.rd_a.size() << "\n";
        file << "# angle grid: da = " << (std::numbers::pi / (2.0 * output.rd_a.size())) << " radians\n";
        file << "#\n";
        for (size_t i = 0; i < output.rd_a.size(); ++i) {
            file << std::setw(4) << i << "  " << output.rd_a[i] << "\n";
        }
    }
    else if (quantity == "RD_R" && !output.rd_r.empty()) {
        file << "# Rd_r[0], [1],..Rd_r[nr-1]. [1/cm²]\n";
        file << "# nr = " << output.rd_r.size() << "\n";
        file << "# radial grid: dr = " << input.dr << " cm\n";
        file << "#\n";
        for (size_t i = 0; i < output.rd_r.size(); ++i) {
            double r = (i + 0.5) * input.dr;
            file << std::setw(4) << i << "  " << std::setw(10) << r << "  " << output.rd_r[i] << "\n";
        }
    }
    else if (quantity == "RD_T" && !output.rd_t.empty()) {
        file << "# Rd_t[0], [1],..Rd_t[nt-1]. [1/ps]\n";
        file << "# nt = " << output.rd_t.size() << "\n";
        file << "# time grid: dt = " << input.dt << " ps\n";
        file << "#\n";
        for (size_t i = 0; i < output.rd_t.size(); ++i) {
            double t = (i + 0.5) * input.dt;
            file << std::setw(4) << i << "  " << std::setw(10) << t << "  " << output.rd_t[i] << "\n";
        }
    }
    else if (quantity == "TD_A" && !output.td_a.empty()) {
        file << "# Td_a[0], [1],..Td_a[na-1]. [1/sr]\n";
        file << "# na = " << output.td_a.size() << "\n";
        file << "# angle grid: da = " << (std::numbers::pi / (2.0 * output.td_a.size())) << " radians\n";
        file << "#\n";
        for (size_t i = 0; i < output.td_a.size(); ++i) {
            file << std::setw(4) << i << "  " << output.td_a[i] << "\n";
        }
    }
    else if (quantity == "TD_R" && !output.td_r.empty()) {
        file << "# Td_r[0], [1],..Td_r[nr-1]. [1/cm²]\n";
        file << "# nr = " << output.td_r.size() << "\n";
        file << "# radial grid: dr = " << input.dr << " cm\n";
        file << "#\n";
        for (size_t i = 0; i < output.td_r.size(); ++i) {
            double r = (i + 0.5) * input.dr;
            file << std::setw(4) << i << "  " << std::setw(10) << r << "  " << output.td_r[i] << "\n";
        }
    }
    else if (quantity == "TD_T" && !output.td_t.empty()) {
        file << "# Td_t[0], [1],..Td_t[nt-1]. [1/ps]\n";
        file << "# nt = " << output.td_t.size() << "\n";
        file << "# time grid: dt = " << input.dt << " ps\n";
        file << "#\n";
        for (size_t i = 0; i < output.td_t.size(); ++i) {
            double t = (i + 0.5) * input.dt;
            file << std::setw(4) << i << "  " << std::setw(10) << t << "  " << output.td_t[i] << "\n";
        }
    }
    else if (quantity == "A_Z" && !output.a_z.empty()) {
        file << "# A_z[0], [1],..A_z[nz-1]. [1/cm]\n";
        file << "# nz = " << output.a_z.size() << "\n";
        file << "# depth grid: dz = " << input.dz << " cm\n";
        file << "#\n";
        for (size_t i = 0; i < output.a_z.size(); ++i) {
            double z = (i + 0.5) * input.dz;
            file << std::setw(4) << i << "  " << std::setw(10) << z << "  " << output.a_z[i] << "\n";
        }
    }
    else if (quantity == "A_T" && !output.a_t.empty()) {
        file << "# A_t[0], [1],..A_t[nt-1]. [1/ps]\n";
        file << "# nt = " << output.a_t.size() << "\n";
        file << "# time grid: dt = " << input.dt << " ps\n";
        file << "#\n";
        for (size_t i = 0; i < output.a_t.size(); ++i) {
            double t = (i + 0.5) * input.dt;
            file << std::setw(4) << i << "  " << std::setw(10) << t << "  " << output.a_t[i] << "\n";
        }
    }
    else if (quantity == "A_RZ" && !output.a_rz.empty()) {
        file << "# A_rz[ir][iz] in 3-column format: r(cm), z(cm), A_rz [1/cm³]\n";
        file << "# nr = " << output.a_rz.rows() << ", nz = " << output.a_rz.cols() << "\n";
        file << "# Grid: dr = " << input.dr << " cm, dz = " << input.dz << " cm\n";
        file << "#\n# r(cm)      z(cm)      A_rz[1/cm³]\n";
        
        for (size_t ir = 0; ir < output.a_rz.rows(); ++ir) {
            for (size_t iz = 0; iz < output.a_rz.cols(); ++iz) {
                double r = (ir + 0.5) * input.dr;
                double z = (iz + 0.5) * input.dz;
                file << std::setw(10) << r << "  " << std::setw(10) << z << "  " 
                     << output.a_rz(ir, iz) << "\n";
            }
        }
    }
    else if (quantity == "RD_RA" && !output.rd_ra.empty()) {
        file << "# Rd_ra[ir][ia] in 3-column format: r(cm), angle(rad), Rd_ra [1/(sr·cm²)]\n";
        file << "# nr = " << output.rd_ra.rows() << ", na = " << output.rd_ra.cols() << "\n";
        file << "# Grid: dr = " << input.dr << " cm, da = " << (std::numbers::pi / (2.0 * output.rd_ra.cols())) << " rad\n";
        file << "#\n# r(cm)      angle(rad)  Rd_ra[1/(sr·cm²)]\n";
        
        double da = std::numbers::pi / (2.0 * output.rd_ra.cols());
        for (size_t ir = 0; ir < output.rd_ra.rows(); ++ir) {
            for (size_t ia = 0; ia < output.rd_ra.cols(); ++ia) {
                double r = (ir + 0.5) * input.dr;
                double angle = (ia + 0.5) * da;
                file << std::setw(10) << r << "  " << std::setw(10) << angle << "  " 
                     << output.rd_ra(ir, ia) << "\n";
            }
        }
    }
    else {
        file << "# Data extraction for " << quantity << " not yet implemented\n";
        file << "# This is a placeholder for quantity: " << quantity << "\n";
        file.close();
        return false;
    }
    
    file.close();
    return true;
}

void ConvProcessor::perform_convolution_extraction() {
    std::cout << "Convolution extraction not yet fully implemented.\n";
    std::cout << "(This would perform convolution and save results)\n";
}

bool ConvProcessor::write_convolution_results(const conv::ConvolutionResult& result, const std::string& filename) {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            log_error("Cannot open output file: " + filename);
            return false;
        }
        
        // Write header information
        file << "# CONV 3.0.0 - Convolved MCML Results\n";
        
        // Get current time
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        file << "# Generated: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\n";
        
        file << "#\n";
        file << "# Matrix dimensions: " << result.convolved_data.rows() << " x " << result.convolved_data.cols() << "\n";
        file << "# Computation time: " << result.computation_time << " seconds\n";
        file << "# Estimated error: " << result.estimated_error << "\n";
        file << "# Converged: " << (result.converged ? "Yes" : "No") << "\n";
        file << "# Iterations used: " << result.iterations_used << "\n";
        file << "#\n";
        file << "# Data format: row-major order (r-coordinate varies fastest)\n";
        file << "#\n";
        
        // Write matrix data
        const auto& matrix = result.convolved_data;
        for (size_t iz = 0; iz < matrix.cols(); ++iz) {
            for (size_t ir = 0; ir < matrix.rows(); ++ir) {
                file << std::scientific << std::setprecision(6) << matrix(ir, iz);
                if (ir < matrix.rows() - 1) {
                    file << "\t";
                }
            }
            file << "\n";
        }
        
        file.close();
        
        // Update statistics
        std::filesystem::path filepath(filename);
        m_stats.bytes_written += std::filesystem::file_size(filepath);
        
        std::cout << "Results written to: " << filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        log_error("Error writing convolution results: " + std::string(e.what()));
        return false;
    }
}

std::string ConvProcessor::generate_output_filename(const std::string& input_filename, const std::string& suffix) const {
    std::filesystem::path input_path(input_filename);
    std::string base = input_path.stem().string();
    std::string ext = input_path.extension().string();
    
    return base + suffix + ".txt";
}

// Stub implementations for other classes
bool ConvolutionParams::is_valid() const noexcept {
    return epsilon > 0.0 && epsilon < 1.0 && max_iterations > 0;
}

} // namespace conv

#ifdef TEST_PROCESSOR
int main(int argc, char* argv[]) {
    using namespace conv;
    
    try {
        ConvProcessor processor;
        
        if (argc < 2) {
            std::cout << "Testing interactive mode...\n";
            return processor.run_interactive() ? 0 : 1;
        } else {
            std::cout << "Testing batch mode with file: " << argv[1] << "\n";
            // Create a simple config for testing
            ProcessorConfig config;
            config.mode = ProcessingMode::Batch;
            config.input_filename = argv[1];
            return processor.run(config) ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
#endif
