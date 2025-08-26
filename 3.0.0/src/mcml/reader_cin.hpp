#pragma once

#include "mcml.hpp"
#include "reader.hpp"

#include <istream>
#include <string>

/**
 * @brief Interactive console input reader for simulation parameters
 * 
 * Extends Reader to provide interactive console input functionality,
 * prompting the user for simulation parameters through stdin.
 */
class ReaderCin : public Reader
{
public:
	ReaderCin() : Reader {{}} {}
	~ReaderCin() = default;

	/** @brief Read simulation parameters interactively from console */
	bool read_params(std::istream& in, RunParams& params) override;

	/** @brief Read mediums list interactively from console */
	bool read_mediums(std::istream& in, vec1<Layer>& out) override;

	/** @brief Read output filename and format interactively from console */
	bool read_output(std::istream& in, std::string& out) override;

	/** @brief Read layer parameters interactively from console */
	bool read_layers(std::istream& in, RunParams& params, vec1<Layer>& out) override;

	/** @brief Read beam source type and position interactively from console */
	bool read_source(std::istream& in, RunParams& params, LightSource& out) override;

	/** @brief Read grid parameters interactively from console */
	bool read_grid(std::istream& in, Grid& out) override;

	/** @brief Read only grid spacing parameters (dz, dr, dt) from console */
	bool ReadGridSpacing(std::istream& in, Grid& out);

	/** @brief Read only grid size parameters (nz, nr, nt, na) from console */
	bool ReadGridSize(std::istream& in, Grid& out);

	/** @brief Read recording options interactively from console */
	bool read_record(std::istream& in, RunParams& params, Record& out) override;

	/** @brief Read target photons/time limit interactively from console */
	bool read_target(std::istream& in, RunParams& params, Target& out, bool add = false) override;

	/** @brief Read weight threshold interactively from console */
	bool read_weight(std::istream& in, double& out) override;
};
