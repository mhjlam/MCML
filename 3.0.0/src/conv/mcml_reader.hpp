/*******************************************************************************
 *	Data reader for MCML output files in CONV 3.0
 *  Copyright M.H.J. Lam, 2025.
 ****/

#pragma once

#include "conv.hpp"
#include "matrix.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

namespace conv
{

// Forward declarations
struct McmlInputData;
struct McmlOutputData;

/**
 * @brief Data structures for MCML input parameters
 */
struct McmlInputData {
	std::string filename;
	size_t num_photons = 0;
	double dz = 0.0, dr = 0.0;     // Grid spacing [cm]
	double dt = 0.0;               // Time spacing [ps]
	size_t nz = 0, nr = 0, na = 0; // Grid dimensions
	size_t nt = 0;                 // Time bins
	size_t num_layers = 0;

	// Layer properties
	std::vector<double> layer_thickness;  // [cm]
	std::vector<double> refractive_index;
	std::vector<double> absorption_coeff; // [1/cm]
	std::vector<double> scattering_coeff; // [1/cm]
	std::vector<double> anisotropy;

	[[nodiscard]] bool is_valid() const noexcept;
};

/**
 * @brief Data structures for MCML output results
 */
struct McmlOutputData {
	// Scalars
	double specular_reflectance = 0.0;
	double total_transmittance = 0.0;
	double total_reflectance = 0.0;
	double total_absorption = 0.0;

	// 1D arrays
	std::vector<double> rd_r; // Diffuse reflectance vs r
	std::vector<double> rd_a; // Diffuse reflectance vs angle
	std::vector<double> rd_t; // Diffuse reflectance vs time
	std::vector<double> td_r; // Diffuse transmittance vs r
	std::vector<double> td_a; // Diffuse transmittance vs angle
	std::vector<double> td_t; // Diffuse transmittance vs time
	std::vector<double> a_z;  // Absorption vs z
	std::vector<double> a_t;  // Absorption vs time

	// 2D arrays
	Matrix2D<double> rd_ra; // Diffuse reflectance vs (r,a)
	Matrix2D<double> rd_rt; // Diffuse reflectance vs (r,t)
	Matrix2D<double> rd_at; // Diffuse reflectance vs (a,t)
	Matrix2D<double> td_ra; // Diffuse transmittance vs (r,a)
	Matrix2D<double> td_rt; // Diffuse transmittance vs (r,t)
	Matrix2D<double> td_at; // Diffuse transmittance vs (a,t)
	Matrix2D<double> a_rz;  // Absorption vs (r,z)
	Matrix2D<double> a_zt;  // Absorption vs (z,t)

	// 3D arrays
	Matrix3D<double> rd_rat; // Diffuse reflectance vs (r,a,t)
	Matrix3D<double> td_rat; // Diffuse transmittance vs (r,a,t)
	Matrix3D<double> a_rzt;  // Absorption vs (r,z,t)

	[[nodiscard]] bool is_valid() const noexcept;
	void clear();
};

/**
 * @brief Reader for MCML output (.mco) files
 */
class McmlDataReader
{
public:
	McmlDataReader() = default;
	~McmlDataReader() = default;

	// Non-copyable but movable
	McmlDataReader(const McmlDataReader&) = delete;
	McmlDataReader& operator=(const McmlDataReader&) = delete;
	McmlDataReader(McmlDataReader&&) = default;
	McmlDataReader& operator=(McmlDataReader&&) = default;

	/**
	 * @brief Read MCML output file
	 * @param filename Path to .mco file
	 * @throws FileException if file cannot be read or is invalid
	 */
	void read_file(const std::string& filename);

	/**
	 * @brief Get input parameters
	 * @return Reference to input data
	 */
	[[nodiscard]] const McmlInputData& input_data() const { return m_input_data; }

	/**
	 * @brief Get output results
	 * @return Reference to output data
	 */
	[[nodiscard]] const McmlOutputData& output_data() const { return m_output_data; }

	/**
	 * @brief Get available quantities for extraction
	 * @return Bitset of extractable quantities
	 */
	[[nodiscard]] ExtractableQuantities get_extractable_quantities() const;

	/**
	 * @brief Check if data is loaded
	 * @return true if valid data is loaded
	 */
	[[nodiscard]] bool has_data() const noexcept { return m_data_loaded; }

	/**
	 * @brief Get file version
	 * @return File version character
	 */
	[[nodiscard]] char file_version() const noexcept { return m_file_version; }

	/**
	 * @brief Clear all loaded data
	 */
	void clear();

private:
	/**
	 * @brief Read file header and determine version
	 * @param file Input file stream
	 */
	void read_header(std::ifstream& file);

	/**
	 * @brief Read input parameters section
	 * @param file Input file stream
	 */
	void read_input_parameters(std::ifstream& file);

	/**
	 * @brief Read output results section
	 * @param file Input file stream
	 */
	void read_output_results(std::ifstream& file);

	/**
	 * @brief Read 1D array from file
	 * @param file Input file stream
	 * @param data Vector to store data
	 * @param expected_size Expected array size
	 */
	void read_1d_array(std::ifstream& file, std::vector<double>& data, size_t expected_size);

	/**
	 * @brief Read 2D array from file
	 * @param file Input file stream
	 * @param data Matrix to store data
	 * @param rows Number of rows
	 * @param cols Number of columns
	 */
	void read_2d_array(std::ifstream& file, Matrix2D<double>& data, size_t rows, size_t cols);

	/**
	 * @brief Read 3D array from file
	 * @param file Input file stream
	 * @param data Matrix to store data
	 * @param dim1 First dimension
	 * @param dim2 Second dimension
	 * @param dim3 Third dimension
	 */
	void read_3d_array(std::ifstream& file, Matrix3D<double>& data, size_t dim1, size_t dim2, size_t dim3);

	/**
	 * @brief Skip whitespace and comments
	 * @param file Input file stream
	 * @return Next non-whitespace character
	 */
	char skip_whitespace_and_comments(std::ifstream& file);

	/**
	 * @brief Parse complete MCML file
	 * @param file Input file stream
	 * @param filename Original filename
	 */
	void parse_mcml_file(std::ifstream& file, const std::string& filename);
	
	/**
	 * @brief Find specific section in file
	 * @param file Input file stream
	 * @param section_name Section to find
	 * @return true if section found
	 */
	bool find_section(std::ifstream& file, const std::string& section_name);
	
	/**
	 * @brief Read numeric values from line
	 * @param file Input file stream
	 * @param pattern Pattern to search for
	 * @return Vector of values if found
	 */
	std::optional<std::vector<double>> read_line_values(std::ifstream& file, const std::string& pattern);
	
	/**
	 * @brief Parse output data section
	 * @param file Input file stream
	 */
	void parse_output_data(std::ifstream& file);
	
	/**
	 * @brief Read array sections from file
	 * @param file Input file stream
	 */
	void read_array_sections(std::ifstream& file);

	/**
	 * @brief Validate input data consistency
	 * @return true if data is consistent
	 */
	[[nodiscard]] bool validate_data_consistency() const noexcept;	McmlInputData m_input_data;
	McmlOutputData m_output_data;
	bool m_data_loaded = false;
	char m_file_version = '3';
	std::string m_current_filename;
};

// Validation helper functions
[[nodiscard]] bool is_valid_filename(const std::string& filename) noexcept;
[[nodiscard]] bool is_mcml_output_file(const std::string& filename) noexcept;

} // namespace conv
