/*******************************************************************************
 *	Main CONV processor class for C++20 version
 *  Copyright M.H.J. Lam, 2025.
 ****/

#pragma once

#include "beam_profile.hpp"
#include "conv.hpp"
#include "convolution_engine.hpp"
#include "mcml_reader.hpp"

#include <chrono>
#include <memory>

namespace conv
{

/**
 * @brief Processing mode for CONV operations
 */
enum class ProcessingMode {
	Interactive, // Interactive command-line interface
	Batch,       // Batch processing from command-line arguments
	Extract      // Extract original data without convolution
};

/**
 * @brief Configuration for CONV processor
 */
struct ProcessorConfig {
	ProcessingMode mode = ProcessingMode::Interactive;
	std::string input_filename;
	std::string output_filename;
	BeamParameters beam_params;
	ConvolutionParams conv_params;
	GridParameters grid_params;
	ExtractableQuantities extract_quantities;

	bool verbose = false;          // Enable verbose output
	bool overwrite_output = false; // Allow output file overwriting

	[[nodiscard]] bool is_valid() const noexcept;
};

/**
 * @brief Statistics for processing operations
 */
struct ProcessingStats {
	std::chrono::duration<double> total_time {0};
	std::chrono::duration<double> reading_time {0};
	std::chrono::duration<double> convolution_time {0};
	std::chrono::duration<double> writing_time {0};

	size_t bytes_read = 0;
	size_t bytes_written = 0;
	size_t cache_hits = 0;
	size_t cache_misses = 0;

	void reset();
	[[nodiscard]] std::string to_string() const;
};

/**
 * @brief Main processor for CONV operations
 */
class ConvProcessor
{
public:
	ConvProcessor();
	~ConvProcessor() = default;

	// Non-copyable but movable
	ConvProcessor(const ConvProcessor&) = delete;
	ConvProcessor& operator=(const ConvProcessor&) = delete;
	ConvProcessor(ConvProcessor&&) = default;
	ConvProcessor& operator=(ConvProcessor&&) = default;

	/**
	 * @brief Run CONV with given configuration
	 * @param config Processing configuration
	 * @return 0 on success, error code on failure
	 */
	int run(const ProcessorConfig& config);

	/**
	 * @brief Run in interactive mode
	 * @return 0 on success, error code on failure
	 */
	int run_interactive();

	/**
	 * @brief Run in batch mode with command-line arguments
	 * @param argc Argument count
	 * @param argv Argument vector
	 * @return 0 on success, error code on failure
	 */
	int run_batch(int argc, char* argv[]);

	/**
	 * @brief Get current processing statistics
	 * @return Processing statistics
	 */
	[[nodiscard]] const ProcessingStats& stats() const noexcept { return m_stats; }

	/**
	 * @brief Set verbose mode
	 * @param verbose Enable/disable verbose output
	 */
	void set_verbose(bool verbose) noexcept { m_verbose = verbose; }

private:
	/**
	 * @brief Display program information and credits
	 */
	void show_about() const;

	/**
	 * @brief Show main menu for interactive mode
	 */
	void show_main_menu() const;

	/**
	 * @brief Show help for command-line usage
	 */
	void show_help() const;

	/**
	 * @brief Parse command-line arguments
	 * @param argc Argument count
	 * @param argv Argument vector
	 * @return Parsed configuration
	 */
	[[nodiscard]] ProcessorConfig parse_arguments(int argc, char* argv[]) const;

	/**
	 * @brief Handle interactive menu command
	 * @param command Menu command character
	 * @return true to continue, false to quit
	 */
	bool handle_menu_command(char command);

	/**
	 * @brief Input MCML file interactively
	 * @return true if successful
	 */
	bool input_file_interactive();

	/**
	 * @brief Configure beam parameters interactively
	 * @return true if successful
	 */
	bool configure_beam_interactive();

	/**
	 * @brief Configure convolution error interactively
	 * @return true if successful
	 */
	bool configure_error_interactive();

	/**
	 * @brief Extract original data
	 * @return true if successful
	 */
	bool extract_original_data();

	/**
	 * @brief Extract convolved data
	 * @return true if successful
	 */
	bool extract_convolved_data();

	/**
	 * @brief Show reflectance, absorption, and transmittance totals
	 * @return true if successful
	 */
	bool show_rat_totals();

	/**
	 * @brief Process a single file with given parameters
	 * @param config Processing configuration
	 * @return true if successful
	 */
	bool process_file(const ProcessorConfig& config);

	/**
	 * @brief Write extraction results to file
	 * @param filename Output filename
	 * @param quantities Quantities to extract
	 * @return true if successful
	 */
	bool write_extraction_results(const std::string& filename, const ExtractableQuantities& quantities);

	/**
	 * @brief Get user input with prompt
	 * @param prompt Prompt string
	 * @return User input string
	 */
	[[nodiscard]] std::string get_user_input(const std::string& prompt) const;

	/**
	 * @brief Get yes/no confirmation from user
	 * @param prompt Prompt string
	 * @return true for yes, false for no
	 */
	[[nodiscard]] bool get_confirmation(const std::string& prompt) const;

	/**
	 * @brief Validate output filename
	 * @param filename Proposed output filename
	 * @param allow_overwrite Allow overwriting existing files
	 * @return true if valid
	 */
	[[nodiscard]] bool validate_output_filename(const std::string& filename, bool allow_overwrite) const;

	/**
	 * @brief Generate default output filename based on input
	 * @param input_filename Input filename
	 * @param suffix Additional suffix
	 * @return Generated output filename
	 */
	[[nodiscard]] std::string generate_output_filename(const std::string& input_filename,
													   const std::string& suffix = "") const;

	/**
	 * @brief Log message if verbose mode is enabled
	 * @param message Message to log
	 */
	void log_verbose(const std::string& message) const;

	/**
	 * @brief Log error message
	 * @param message Error message
	 */
	void log_error(const std::string& message) const;

	// Data members
	std::unique_ptr<McmlDataReader> m_reader;
	std::unique_ptr<BeamProfile> m_beam;
	std::unique_ptr<ConvolutionEngine> m_engine;

	ProcessingStats m_stats;
	bool m_verbose = false;
	bool m_data_loaded = false;

	// Current session state
	ProcessorConfig m_current_config;

	static constexpr std::string_view VERSION = "3.0.0";
	static constexpr std::string_view BUILD_DATE = "2025";
};

} // namespace conv
