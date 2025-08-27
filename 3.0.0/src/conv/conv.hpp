/*******************************************************************************
 *	Copyright Univ. of Texas M.D. Anderson Cancer Center, 1992-1996.
 *  Copyright M.H.J. Lam, 2025.
 *	Convolution program for MCML data processing in C++20.
 ****
 *	Version 1.x:    10/1991.
 *	Version 2.0:    02/1996.
 *  Version 3.0:    03/2025.
 *
 *	@authors
 *	Lihong Wang, Ph.D.
 *	Bioengineering Program
 *	Texas A&M University
 *	College Station, Texas
 *
 *	Steven L. Jacques, Ph.D.
 *	M.D. Anderson Cancer Center
 *	University of Texas
 *	Houston, Texas
 *
 *	Liqiong Zheng, B.S.
 *	Department of Computer Science
 *	University of Houston
 *	Houston, Texas
 *
 *  M.H.J. Lam, MSc.
 *  Graduate School of Natural Sciences
 *  Utrecht University
 *  Utrecht, Netherlands
 *
 *	@brief Convolution Program for MCML Output Processing
 *
 *	This program processes data from MCML - A Monte Carlo simulation of light
 *	transport in multi-layered turbid media.
 *
 *	Major modifications in version 3.0:
 *		- Complete conversion to modern C++20
 *		- Object-oriented design following MCML 3.0 patterns
 *		- RAII and smart pointer usage
 *		- Strongly typed enums and modern containers
 *		- Exception-safe design
 *		- Header-only templates for generic algorithms
 *
 ****/

#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <concepts>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace conv
{

// Constants
constexpr double DEFAULT_EPSILON = 0.1;  // Default relative error in convolution
constexpr int COLUMN_WIDTH = 80;         // Column width for printing
constexpr int MAX_FILENAME_LENGTH = 256; // Maximum filename length

// Forward declarations
class ConvProcessor;
class BeamProfile;
class ConvolutionEngine;
class DataExtractor;
template<typename T>
class Matrix2D;
template<typename T>
class Matrix3D;
template<typename Key, typename Value>
class BinaryTree;

/**
 * @brief Beam types for photon sources
 */
enum class BeamType {
	Original, // Infinitely narrow beam (default from MCML output)
	Flat,     // Flat beam with radius R
	Gaussian, // Gaussian with 1/e² radius R
	Arbitrary // General beam described by interpolation points
};

/**
 * @brief Available data quantities for extraction
 * Using modern bitset approach instead of bit fields
 */
struct ExtractableQuantities {
	std::bitset<32> quantities;

	// Transmittance quantities
	static constexpr size_t TD_R_A_T = 0; // Td_r@a@t
	static constexpr size_t TD_R_A = 1;   // Td_r@a*t
	static constexpr size_t TD_R_T = 2;   // Td_r*a@t
	static constexpr size_t TD_R = 3;     // Td_r*a*t

	static constexpr size_t TD_A_R_T = 4; // Td_a@r@t
	static constexpr size_t TD_A_R = 5;   // Td_a@r*t
	static constexpr size_t TD_A_T = 6;   // Td_a*r@t
	static constexpr size_t TD_A = 7;     // Td_a*r*t

	static constexpr size_t TD_T_R_A = 8; // Td_t@r@a
	static constexpr size_t TD_T_R = 9;   // Td_t@a*a
	static constexpr size_t TD_T_A = 10;  // Td_t*r@a
	static constexpr size_t TD_T = 11;    // Td_t*r*a

	// Reflectance quantities
	static constexpr size_t RD_R_A_T = 12; // Rd_r@a@t
	static constexpr size_t RD_R_A = 13;   // Rd_r@a*t
	static constexpr size_t RD_R_T = 14;   // Rd_r*a@t
	static constexpr size_t RD_R = 15;     // Rd_r*a*t

	static constexpr size_t RD_A_R_T = 16; // Rd_a@r@t
	static constexpr size_t RD_A_R = 17;   // Rd_a@r*t
	static constexpr size_t RD_A_T = 18;   // Rd_a*r@t
	static constexpr size_t RD_A = 19;     // Rd_a*r*t

	static constexpr size_t RD_T_R_A = 20; // Rd_t@r@a
	static constexpr size_t RD_T_R = 21;   // Rd_t@a*a
	static constexpr size_t RD_T_A = 22;   // Rd_t*r@a
	static constexpr size_t RD_T = 23;     // Rd_t*r*a

	// Absorption quantities
	static constexpr size_t A_RZ_T = 24;  // A_rz@t
	static constexpr size_t A_RZ = 25;    // A_rz*t
	static constexpr size_t A_Z_T = 26;   // A_z*r@t
	static constexpr size_t A_Z = 27;     // A_z*r*t
	static constexpr size_t A_T_R_Z = 28; // A_t@r@z
	static constexpr size_t A_T_Z = 29;   // A_t*r@z
	static constexpr size_t A_T = 30;     // A_t*r*z
	static constexpr size_t A_L = 31;     // A_layer

	// Helper methods
	bool has(size_t quantity) const { return quantities.test(quantity); }
	void set(size_t quantity, bool value = true) { quantities.set(quantity, value); }
	void clear() { quantities.reset(); }
};

/**
 * @brief Parameters describing a photon beam profile
 */
struct BeamParameters {
	BeamType type = BeamType::Original;
	double total_power = 1.0;            // Total power [J]
	double radius = 0.0;                 // Beam radius [cm]
	std::string profile_filename;        // Profile file for arbitrary beam
	std::vector<double> radii;           // r values for beam profile [cm]
	std::vector<double> power_densities; // Power density values [J/cm²]

	// Validation
	[[nodiscard]] bool is_valid() const noexcept;
};

/**
 * @brief Grid parameters for convolution
 */
struct GridParameters {
	size_t nr = 0;    // Number of r gridlines from input
	double dr = 0.0;  // r grid spacing from input [cm]
	size_t nrc = 0;   // Number of r gridlines for convolution
	double drc = 0.0; // r grid spacing for convolution [cm]
	size_t nxc = 0;   // Number of x gridlines for convolution
	double dxc = 0.0; // x grid spacing for convolution [cm]

	[[nodiscard]] bool is_valid() const noexcept;
};

/**
 * @brief Configuration for convolution processing
 */
struct ConvolutionConfig {
	bool data_available = false;       // MCO data is available
	char file_version = '3';           // MCO file version
	ExtractableQuantities extractable; // Available quantities
	BeamParameters beam;               // Incident beam parameters
	double epsilon = DEFAULT_EPSILON;  // Relative error in convolution
	GridParameters grid;               // Grid configuration

	[[nodiscard]] bool is_valid() const noexcept;
};

/**
 * @brief Exception types for CONV processing
 */
class ConvException : public std::exception
{
public:
	explicit ConvException(std::string message) : m_message(std::move(message)) {}
	[[nodiscard]] const char* what() const noexcept override { return m_message.c_str(); }

private:
	std::string m_message;
};

class FileException : public ConvException
{
public:
	explicit FileException(const std::string& message) : ConvException("File error: " + message) {}
};

class ConvolutionException : public ConvException
{
public:
	explicit ConvolutionException(const std::string& message) : ConvException("Convolution error: " + message) {}
};

/**
 * @brief Concepts for type constraints
 */
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<typename T>
concept ConvolvableData = requires(T t) {
	{ t.data() } -> std::convertible_to<const double*>;
	{ t.size() } -> std::convertible_to<size_t>;
};

} // namespace conv
