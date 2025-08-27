/*******************************************************************************
 *	Basic implementation for processor class
 *  Copyright M.H.J. Lam, 2025.
 ****/

#include "processor.hpp"
#include <iostream>
#include <sstream>

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
    command = std::tolower(command);
    
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
    std::cout << "Beam configuration not yet implemented.\n";
    return true;
}

bool ConvProcessor::configure_error_interactive() {
    std::cout << "Error configuration not yet implemented.\n";
    return true;
}

bool ConvProcessor::extract_original_data() {
    std::cout << "Original data extraction not yet implemented.\n";
    return true;
}

bool ConvProcessor::extract_convolved_data() {
    std::cout << "Convolved data extraction not yet implemented.\n";
    return true;
}

bool ConvProcessor::show_rat_totals() {
    std::cout << "RAT totals display not yet implemented.\n";
    return true;
}

bool ConvProcessor::process_file(const ProcessorConfig& config) {
    std::cout << "File processing not yet implemented.\n";
    return true;
}

ProcessorConfig ConvProcessor::parse_arguments(int argc, char* argv[]) const {
    ProcessorConfig config;
    config.mode = ProcessingMode::Batch;
    
    // Basic argument parsing - just take first non-option as input file
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] != '-' && config.input_filename.empty()) {
            config.input_filename = arg;
            break;
        }
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

// Stub implementations for other classes
bool ConvolutionParams::is_valid() const noexcept {
    return epsilon > 0.0 && epsilon < 1.0 && max_iterations > 0;
}

} // namespace conv
