#pragma once

#include <iostream>
#include <memory>

class Random;

struct Radiance;
struct RunParams;

class Writer
{
public:
	Writer(std::string filename = {});
	~Writer();

	/** @brief Write medium list to output stream */
	void write_mediums(std::ostream& output, RunParams& params);
	
	/** @brief Write output filename to stream */
	void write_filename(std::ostream& output, RunParams& params);
	
	/** @brief Write grid parameters to stream */
	void write_grid_params(std::ostream& output, RunParams& params);
	
	/** @brief Write grid size parameters to stream */
	void write_grid_size(std::ostream& output, RunParams& params);
	
	/** @brief Write record configuration to stream */
	void write_record(std::ostream& output, RunParams& params);
	
	/** @brief Write weight threshold to stream */
	void write_weight(std::ostream& output, RunParams& params);
	
	/** @brief Write random seed to stream */
	void write_random_seed(std::ostream& output, RunParams& params);
	
	/** @brief Write layer parameters to stream */
	void write_layers(std::ostream& output, RunParams& params);
	
	/** @brief Write termination criteria to stream */
	void write_end_criteria(std::ostream& output, RunParams& params);
	
	/** @brief Write source type to stream */
	void write_source_type(std::ostream& output, RunParams& params);
	
	/** @brief Write photon source parameters to stream */
	void write_photon_source(std::ostream& output, RunParams& params);
	
	/** @brief Write all simulation parameters to stream */
	void write_params(std::ostream& output, RunParams& params);
	
	/** @brief Write version information to stream */
	void write_version(std::ostream& output, const std::string_view& version);
	
	/** @brief Write random number generator state to stream */
	void write_randomizer(std::ostream& output, std::shared_ptr<Random> random);

	/** @brief Write simulation results to output file */
	void write_results(std::ostream& output, RunParams& params, Radiance& radiance, std::shared_ptr<Random> random);

	/** @brief Write radiance totals to output file */
	void write_radiance(std::ostream& output, Radiance& radiance);

	// Ballistic absorption per unit depth, per unit time [1/(cm ps)]
	void write_ab_zt(std::ostream& output, std::size_t Nz, std::size_t Nt, Radiance& radiance);

	// Rate of absorption per unit volume, per unit time [1/(cm� ps]
	void write_a_rzt(std::ostream& output, std::size_t Nr, std::size_t Nz, std::size_t Nt, Radiance& radiance);

	// Ballistic absorption per unit depth [1/cm]
	void write_ab_z(std::ostream& output, std::size_t Nz, Radiance& radiance);

	// Rate of absorption per unit volume [1/cm�]
	void write_a_rz(std::ostream& output, std::size_t Nr, std::size_t Nz, Radiance& radiance);

	// Rate of absorption per unit time [1/(cm ps)]
	void write_a_zt(std::ostream& output, std::size_t Nz, std::size_t Nt, Radiance& radiance);

	// Absorption per unit depth [1/cm]
	void write_a_z(std::ostream& output, std::size_t Nz, Radiance& radiance);

	// Absorption per unit time [1/ps]
	void write_a_t(std::ostream& output, std::size_t Nt, Radiance& radiance);

	// Diffuse reflectance per unit area, per unit solid angle, per unit time [1/(cm� sr ps)]
	void write_r_rat(std::ostream& output, std::size_t Nr, std::size_t Na, std::size_t Nt, Radiance& radiance);

	// Diffuse reflectance per unit area, per unit solid angle [1/(cm� sr)]
	void write_r_ra(std::ostream& output, std::size_t Nr, std::size_t Na, Radiance& radiance);

	// Diffuse reflectance per unit solid angle, per unit time [1/sr ps]
	void write_r_rt(std::ostream& output, std::size_t Nr, std::size_t Nt, Radiance& radiance);

	// Diffuse reflectance per unit area, per unit time [1/cm� ps]
	void write_r_at(std::ostream& output, std::size_t Na, std::size_t Nt, Radiance& radiance);

	// Diffuse reflectance distribution per unit area [1/cm�]
	void write_r_r(std::ostream& output, std::size_t Nr, Radiance& radiance);

	// Diffuse reflectance per unit solid angle [1/sr]
	void write_r_a(std::ostream& output, std::size_t Na, Radiance& radiance);

	// Diffuse reflectance per unit time [1/ps]
	void write_r_t(std::ostream& output, std::size_t Nt, Radiance& radiance);

	// Diffuse transmittance per unit area, per unit solid angle, per unit time [1/(cm� sr ps)]
	void write_t_rat(std::ostream& output, std::size_t Nr, std::size_t Na, std::size_t Nt, Radiance& radiance);

	// Diffuse transmittance per unit area, per unit solid angle [1/(cm� sr)]
	void write_t_ra(std::ostream& output, std::size_t Nr, std::size_t Na, Radiance& radiance);

	// Diffuse transmittance per unit solid angle, per unit time [1/sr ps]
	void write_t_rt(std::ostream& output, std::size_t Nr, std::size_t Nt, Radiance& radiance);

	// Diffuse transmittance per unit area, per unit time [1/cm� ps]
	void write_t_at(std::ostream& output, std::size_t Na, std::size_t Nt, Radiance& radiance);

	// Diffuse reflectance per unit area [1/cm�]
	void write_t_r(std::ostream& output, std::size_t Nr, Radiance& radiance);

	// Diffuse reflectance per unit solid angle [1/sr]
	void write_t_a(std::ostream& output, std::size_t Na, Radiance& radiance);

	// Diffuse reflectance per unit time [1/ps]
	void write_t_t(std::ostream& output, std::size_t Nt, Radiance& radiance);

public:
	operator std::ostream&() { return *m_output; }

protected:
	std::unique_ptr<std::ostream> m_output;
};
