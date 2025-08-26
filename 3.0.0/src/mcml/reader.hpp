#pragma once

#include "mcml.hpp"
#include "reader_util.hpp"

#include <iostream>
#include <istream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using OutputFile = std::pair<std::string, FileFormat>;

class Random;

class Reader
{
public:
	Reader(std::string filename, std::string_view version = MCI_VERSION);
	~Reader() = default;

	/** @brief Read input parameters for all runs and count number of runs */
	virtual bool read_params(std::istream& input, RunParams& params);

	/** @brief Read the mediums list */
	virtual bool read_mediums(std::istream& input, vec1<Layer>& out);

	/** @brief Read the output filename and format */
	virtual bool read_output(std::istream& input, std::string& out);

	/** @brief Read the parameters of all layers */
	virtual bool read_layers(std::istream& input, RunParams& params, vec1<Layer>& out);

	/** @brief Read beam source type and starting position */
	virtual bool read_source(std::istream& input, RunParams& params, LightSource& out);

	/** @brief Read grid separation parameters and number of grid lines */
	virtual bool read_grid(std::istream& in, Grid& out);

	/** @brief Read which quantity is to be scored */
	virtual bool read_record(std::istream& input, RunParams& params, Record& out);

	/** @brief Read number of photons and computation time limit */
	virtual bool read_target(std::istream& input, RunParams& params, Target& out, bool add = false);

	/** @brief Read the weight threshold */
	virtual bool read_weight(std::istream& input, double& out);

	/** @brief Read random number generator seed (unused) */
	bool read_seed(std::istream& input, long& out);

	/** @brief Check whether input version matches expected version */
	bool read_version(std::istream& input, const std::string_view& version);

	/** @brief Read input parameters for one run */
	bool read_run_params(std::istream& input, RunParams& params);

	/** @brief Read and restore random number generator state from previous output */
	bool read_randomizer(std::istream& input, std::shared_ptr<Random>& random);

	/** @brief Read simulation results back from output file */
	bool read_radiance(std::istream& input, RunParams& params, std::shared_ptr<Random>& random, Radiance& out);

	/** @brief Skip specified number of lines in input stream */
	void skip_line(std::istream& input, std::size_t num_lines = 1);

protected:
	// Skip space or comment lines and return a data line.
	std::string next_dataline(std::istream& in);

	// Check consistancy of input parameters.
	bool check_input_params(RunParams& params);

private:
	// Diffuse reflectance per unit area, per unit solid angle, per unit time [1/(cm� sr ps)]
	vec3<double> read_r_rat(std::istream& input, std::size_t Nr, std::size_t Na, std::size_t Nt);

	// Diffuse reflectance per unit area, per unit solid angle [1/(cm� sr)]
	vec2<double> read_r_ra(std::istream& input, std::size_t Nr, std::size_t Na);

	// Diffuse reflectance per unit solid angle, per unit time [1/sr ps]
	vec2<double> read_r_rt(std::istream& input, std::size_t Nr, std::size_t Nt);

	// Diffuse reflectance per unit area, per unit time [1/cm� ps]
	vec2<double> read_r_at(std::istream& input, std::size_t Na, std::size_t Nt);

	// Diffuse reflectance distribution per unit area [1/cm�]
	vec1<double> read_r_r(std::istream& input, std::size_t Nr);

	// Diffuse reflectance per unit solid angle [1/sr]
	vec1<double> read_r_a(std::istream& input, std::size_t Na);

	// Diffuse reflectance per unit time [1/ps]
	vec1<double> read_r_t(std::istream& input, std::size_t Nt);

	// Diffuse transmittance per unit area, per unit solid angle, per unit time [1/(cm� sr ps)]
	vec3<double> read_t_rat(std::istream& input, std::size_t Nr, std::size_t Na, std::size_t Nt);

	// Diffuse transmittance per unit area, per unit solid angle [1/(cm� sr)]
	vec2<double> read_t_ra(std::istream& input, std::size_t Nr, std::size_t Na);

	// Diffuse transmittance per unit solid angle, per unit time [1/sr ps]
	vec2<double> read_t_rt(std::istream& input, std::size_t Nr, std::size_t Nt);

	// Diffuse transmittance per unit area, per unit time [1/cm� ps]
	vec2<double> read_t_at(std::istream& input, std::size_t Na, std::size_t Nt);

	// Diffuse reflectance per unit area [1/cm�]
	vec1<double> read_t_r(std::istream& input, std::size_t Nr);

	// Diffuse reflectance per unit solid angle [1/sr]
	vec1<double> read_t_a(std::istream& input, std::size_t Na);

	// Diffuse reflectance per unit time [1/ps]
	vec1<double> read_t_t(std::istream& input, std::size_t Nt);

	// Rate of absorption per unit volume, per unit time [1/(cm� ps]
	vec3<double> read_a_rzt(std::istream& input, std::size_t Nr, std::size_t Nz, std::size_t Nt);

	// Rate of absorption per unit volume [1/cm�]
	vec2<double> read_a_rz(std::istream& input, std::size_t Nr, std::size_t Nz);

	// Rate of absorption per unit time [1/(cm ps)]
	vec2<double> read_a_zt(std::istream& input, std::size_t Nz, std::size_t Nt);

	// Absorption per unit depth [1/cm]
	vec1<double> read_a_z(std::istream& input, std::size_t Nz);

	// Absorption per unit time [1/ps]
	vec1<double> read_a_t(std::istream& input, std::size_t Nt);

	// Ballistic absorption per unit depth, per unit time [1/(cm ps)]
	vec2<double> read_ab_zt(std::istream& input, std::size_t Nz, std::size_t Nt);

	// Ballistic absorption per unit depth [1/cm]
	vec1<double> read_ab_z(std::istream& input, std::size_t Nz);

public:
	operator std::istream&() { return *m_input; }

protected:
	std::string m_filename;
	std::unique_ptr<std::istream> m_input;
};
