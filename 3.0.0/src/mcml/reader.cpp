#include "reader.hpp"

#include "exception.hpp"
#include "mcml.hpp"
#include "random.hpp"
#include "reader.tpp"

#include <algorithm>
#include <format>
#include <fstream>
#include <limits>
#include <numbers>
#include <ranges>
#include <sstream>
#include <string_view>

using namespace std::literals;

Reader::Reader(std::string filename, std::string_view version) :
	m_filename(filename), m_input {std::make_unique<std::istream>(nullptr)} {
	if (filename.empty()) {
		m_input = std::make_unique<std::istream>(std::cin.rdbuf());
	}
	else {
		auto file_stream = std::make_unique<std::ifstream>(filename);

		if (!file_stream->is_open()) {
			throw Exception("Failed to open file: " + filename, true);
		}
		m_input = std::move(file_stream);

		if (!read_version(*m_input, version)) {
			throw Exception("Invalid file version.");
		}
	}
}

bool Reader::read_params(std::istream& input, RunParams& params) {
	std::cout << std::format("Reading parameters from {}.", m_filename) << std::endl;

	auto endOfRuns = [&]() {
		// Found end of runs
		bool end_found = false;

		// Record input position
		std::streampos file_pos = input.tellg();

		std::string buf = next_dataline(input);

		if (buf.empty()) {
			end_found = true;
			std::cout << "Missing end." << std::endl;
		}
		else if (buf.find("end") != std::string::npos) {
			end_found = true;
		}

		// Restore postion
		input.seekg(file_pos, std::ios::beg);
		return end_found;
	};

	// Read list of mediums
	if (!read_mediums(input, params.mediums)) {
		return false;
	}

	// Save current position in input file.
	std::streampos file_pos = input.tellg();

	short run_index = 0;
	do {
		++run_index;
		std::cout << "Checking input data for run " << run_index << std::endl;

		// Read the input parameters for one run
		if (!read_run_params(input, params)) {
			std::cerr << "Error reading input parameters." << std::endl;
			return false;
		}

		// Attempt insert and detect duplicate output filenames
		if (!params.unique_output_filenames.insert(params.output_filename).second) {
			std::cout << "File name " + params.output_filename + " duplicated." << std::endl;
			return false;
		}
	}
	while (!endOfRuns());

	params.num_runs = static_cast<std::size_t>(run_index);

	// Restore file position
	input.seekg(file_pos, std::ios::beg);
	return true;
}

bool Reader::read_version(std::istream& input, const std::string_view& version) {
	// Find next data line
	std::string line = next_dataline(input);

	if (line.find(version) == std::string::npos) {
		std::cerr << "Invalid file version.";
		return false;
	}
	return true;
}

bool Reader::read_mediums(std::istream& input, vec1<Layer>& out) {
	vec1<Layer> mediums;

	// Get current output position
	std::streampos file_pos = input.tellg();

	// Find number of mediums
	std::size_t num_mediums = 0;

	// Keep reading lines until the end of the mediums section
	for (std::string buf = next_dataline(input); buf.find("end") == std::string::npos; buf = next_dataline(input)) {
		if (buf.empty()) {
			std::cerr << "Error: Missing end." << std::endl;
			return false;
		}

		std::string error = "Bad optical parameters in medium " + std::to_string(num_mediums);
		auto [success, name, eta, mu_a, mu_s, g] = read_line<std::string, double, double, double, double>(
			buf, error, [](const std::tuple<std::string, double, double, double, double>& t) {
				return (std::get<1>(t) > 0.0 && std::get<2>(t) >= 0.0 && std::get<3>(t) >= 0.0 && std::get<4>(t) >= -1.0
						&& std::get<4>(t) <= 1.0);
			});

		mediums.push_back(Layer {.index = num_mediums, .name = name, .eta = eta, .mu_a = mu_a, .mu_s = mu_s, .g = g});
		num_mediums++;
	}

	if (num_mediums < 1) {
		std::cerr << "Error: No mediums found." << std::endl;
		return false;
	}

	std::for_each(mediums.begin(), mediums.end(), [&](Layer& medium) { out.push_back(medium); });

	return true;
}

bool Reader::read_output(std::istream& in, std::string& out) {
	std::string buf = next_dataline(in);

	std::vector<alpha_num> extracted;
	extract(buf, extracted, {std::string {}}, "Error reading file name.");
	out = std::get<std::string>(extracted[0]);

	return true;
}

bool Reader::read_grid(std::istream& in, Grid& out) {
	using namespace std;

	auto [s1, dz, dr, dt] = read<double, double, double>(
		in, "Invalid or non-positive dz, dr, dt", [](const std::tuple<double, double, double>& t) {
			return (std::get<0>(t) > 0.0 && std::get<1>(t) > 0.0 && std::get<2>(t) > 0.0);
		});
	if (!s1) {
		return false;
	}

	auto [s2, nz, nr, nt, na] = read<int, int, int, int>(
		in, "Invalid or non-positive nz, nr, nt, na", [](const std::tuple<int, int, int, int>& t) {
			return (std::get<0>(t) > 0 && std::get<1>(t) > 0 && std::get<2>(t) > 0 && std::get<3>(t) > 0);
		});
	if (!s2) {
		return false;
	}

	double da = 0.5 * std::numbers::pi / na;

	out = Grid {.step_z = dz,
				.step_r = dr,
				.step_a = da,
				.step_t = dt,
				.num_z = static_cast<size_t>(nz),
				.num_r = static_cast<size_t>(nr),
				.num_t = static_cast<size_t>(nt),
				.num_a = static_cast<size_t>(na),
				.max_z = dz * nz,
				.max_r = dr * nr,
				.max_a = da * na,
				.max_t = dt * nt};

	return true;
}

bool Reader::read_record(std::istream& input, [[maybe_unused]] RunParams& params, Record& record) {
	std::string buf = next_dataline(input);
	if (buf.empty()) {
		std::cerr << "Error: No scored quantities found." << std::endl;
		return false;
	}

	std::string string;
	std::stringstream iss(buf);

	do {
		iss >> string;

		// Stop when comment is found
		if (string.starts_with("#")) {
			break;
		}

		// Trim and uppercase
		string = std::format("{:}", string);
		string = uppercase(string);

		if (string == "R_R"sv) {
			record.R_r = true;
		}
		else if (string == "R_A"sv) {
			record.R_a = true;
		}
		else if (string == "R_RA"sv) {
			record.R_ra = true;
		}
		else if (string == "R_T"sv) {
			record.R_t = true;
		}
		else if (string == "R_RT"sv) {
			record.R_rt = true;
		}
		else if (string == "R_AT"sv) {
			record.R_at = true;
		}
		else if (string == "R_RAT"sv) {
			record.R_rat = true;
		}
		else if (string == "T_R"sv) {
			record.T_r = true;
		}
		else if (string == "T_A"sv) {
			record.T_a = true;
		}
		else if (string == "T_RA"sv) {
			record.T_ra = true;
		}
		else if (string == "T_T"sv) {
			record.T_t = true;
		}
		else if (string == "T_RT"sv) {
			record.T_rt = true;
		}
		else if (string == "T_AT"sv) {
			record.T_at = true;
		}
		else if (string == "T_RAT"sv) {
			record.T_rat = true;
		}
		else if (string == "A_Z"sv) {
			record.A_z = true;
		}
		else if (string == "A_RZ"sv) {
			record.A_rz = true;
		}
		else if (string == "A_T"sv) {
			record.A_t = true;
		}
		else if (string == "A_ZT"sv) {
			record.A_zt = true;
		}
		else if (string == "A_RZT"sv) {
			record.A_rzt = true;
		}
		else {
			std::cerr << "Unknown quantity: "s + string << std::endl;
			return false;
		}
	}
	while (!iss.fail() && !string.empty());

	// Check for conflicting records
	if (record.R_rat) {
		record.R_ra = record.R_rt = record.R_at = record.R_r = record.R_a = record.R_t = false;
	}
	if (record.R_ra) {
		record.R_r = record.R_a = false;
	}
	if (record.R_rt) {
		record.R_r = record.R_t = false;
	}
	if (record.R_at) {
		record.R_a = record.R_t = false;
	}
	if (record.T_rat) {
		record.T_ra = record.T_rt = record.T_at = record.T_r = record.T_a = record.T_t = false;
	}
	if (record.T_ra) {
		record.T_r = record.T_a = false;
	}
	if (record.T_rt) {
		record.T_r = record.T_t = false;
	}
	if (record.T_at) {
		record.T_a = record.T_t = false;
	}
	if (record.A_rzt) {
		record.A_rz = record.A_zt = record.A_z = record.A_t = false;
	}
	if (record.A_rz) {
		record.A_z = false;
	}
	if (record.A_zt) {
		record.A_z = record.A_t = false;
	}
	if (record.A_zt) {
		record.A_z = record.A_t = false;
	}

	return true;
}

bool Reader::read_weight(std::istream& in, double& out) {
	auto [success, weight] =
		read_single<double>(in, "Invalid weight threshold", [](const double& w) { return (w > 0.0 && w < 1.0); });
	if (!success) {
		return false;
	}

	out = weight;
	return true;
}

bool Reader::read_seed(std::istream& in, long& out) {
	auto [success, seed] = read_single<long>(in, "Invalid random number seed", [](const long& s) {
		return (s > 0 && s < std::numeric_limits<long>::max());
	});
	if (!success) {
		return false;
	}

	out = seed;
	return true;
}

bool Reader::read_layers(std::istream& input, RunParams& params, vec1<Layer>& out) {
	std::string name;
	double thickness = 0.0;

	// Z coordinate of the current layer
	double z = 0.0;

	// Read layers line by line until "end" is found
	for (std::string buf = next_dataline(input); buf.find("end") == std::string::npos; buf = next_dataline(input)) {
		// Extract name and optionally thickness
		std::vector<alpha_num> extracted;
		if (!extract(buf, extracted, {std::string {}, double {}}, true)) {
			std::cerr << "Error reading layer specifications." << std::endl;
			return false;
		}

		// Assign name and thickness
		name = std::get<std::string>(extracted[0]);
		if (extracted.size() > 1) {
			thickness = std::get<double>(extracted[1]);
			if (thickness <= 0.0) {
				std::cerr << "Nonpositive layer thickness." << std::endl;
				return false;
			}
		}
		else {
			thickness = 0.0; // Ambient layers have no thickness
		}

		// Find the medium by name in params.mediums
		auto it = std::find_if(params.mediums.begin(), params.mediums.end(),
							   [&](const Layer& medium) { return medium.name == name; });

		if (it == params.mediums.end()) {
			std::cerr << "Invalid medium name: " << name << std::endl;
			return false;
		}

		// Create a new layer and assign parameters from the medium
		Layer layer = Layer(*it);

		// Assign z0 and z1 based on thickness
		layer.z0 = z;
		z += thickness; // if thickness is 0.0, nothing happens
		layer.z1 = z;

		// Add the layer to the output vector
		out.push_back(layer);
	}

	// Update the number of layers in params (excluding ambient layers)
	params.num_layers = out.size() - 2;

	return true;
}

bool Reader::read_target(std::istream& input, RunParams& params, Target& out, bool add) {
	Target target = params.target;

	std::vector<alpha_num> extracted;
	std::string buf = next_dataline(input);
	if (!extract(buf, extracted, {double {}, std::string {}}, true)) {
		std::cerr << "Error reading target." << std::endl;
		return false;
	}

	if (extracted.size() == 1 && std::holds_alternative<double>(extracted[0])) {
		int num_photons = static_cast<int>(std::get<double>(extracted[0]));

		if (num_photons > 0) {
			target.control_bit = ControlBit::NumPhotons;

			if (add) {
				target.photons_remaining += static_cast<std::size_t>(num_photons);
			}
			else {
				target.photons_limit = static_cast<std::size_t>(num_photons);
			}
		}
		else {
			std::cerr << "Nonpositive number of photons." << std::endl;
			return false;
		}
	}
	else if (extracted.size() == 1 && std::holds_alternative<std::string>(extracted[0])) {
		std::string time = std::get<std::string>(extracted[0]);

		int hours = 0;
		int minutes = 0;
		char separator;

		std::istringstream iss(time);
		iss >> hours >> separator >> minutes;

		if (!iss || separator != ':') {
			std::cerr << "Invalid time limit format." << std::endl;
			return false;
		}

		if ((hours * 3600 + minutes * 60) > 0) {
			target.control_bit = ControlBit::TimeLimit;

			if (add) {
				target.time_remaining += hours * 3600 + minutes * 60;
			}
			else {
				target.time_limit = hours * 3600 + minutes * 60;
			}
		}
		else {
			std::cerr << "Nonpositive time limit." << std::endl;
			return false;
		}
	}
	else if (extracted.size() == 2 && std::holds_alternative<double>(extracted[0])
			 && std::holds_alternative<std::string>(extracted[1])) {
		int num_photons = static_cast<int>(std::get<double>(extracted[0]));
		std::string time = std::get<std::string>(extracted[1]);

		int hours = 0;
		int minutes = 0;
		char separator;

		std::istringstream iss(time);
		iss >> hours >> separator >> minutes;

		if (!iss || separator != ':') {
			std::cerr << "Invalid time limit format." << std::endl;
			return false;
		}

		if (num_photons > 0 && (hours * 3600 + minutes * 60) >= 0) {
			target.control_bit = ControlBit::Both;

			if (add) {
				target.photons_remaining += static_cast<std::size_t>(num_photons);
				target.time_remaining += hours * 3600 + minutes * 60;
			}
			else {
				target.photons_limit = static_cast<std::size_t>(num_photons);
				target.time_limit = hours * 3600 + minutes * 60;
			}
		}
		else {
			std::cerr << "Nonpositive number of photons or time limit." << std::endl;
		}
	}
	else {
		std::cerr << "Invalid number of photons or time limit." << std::endl;
	}

	if (!add) {
		target.photons_remaining = target.photons_limit;
		target.time_remaining = target.time_limit;
	}

	// Set output values
	out = Target {.control_bit = target.control_bit,
				  .photons_limit = target.photons_limit,
				  .time_limit = target.time_limit,
				  .photons_remaining = target.photons_remaining,
				  .time_remaining = target.time_remaining};

	return true;
}

bool Reader::read_source([[maybe_unused]] std::istream& input, RunParams& params, [[maybe_unused]] LightSource& out) {
	// Compute the index to layer according to the z coordinate.
	// If the z is on an interface between layers, the returned index will point to the upper layer.
	// Layer 0 is the top ambient and layer num_layers+1 is the bottom ambient layer.
	auto layer_index = [&](double z, RunParams& params) -> std::size_t {
		for (std::size_t i = 1; i <= params.num_layers; i++) {
			if (z >= params.layers[i].z0 && z <= params.layers[i].z1) {
				return i;
			}
		}
		return std::numeric_limits<std::size_t>::max();
	};

	LightSource source {};

	std::string buf = next_dataline(input);

	std::vector<alpha_num> extracted_source;
	if (!extract(buf, extracted_source, {std::string {}})) {
		std::cerr << "Error reading photon source type." << std::endl;
		return false;
	}

	std::string source_type = std::get<std::string>(extracted_source[0]);
	if (uppercase(source_type) == "PENCIL"sv) {
		source.beam = BeamType::Pencil;
	}
	else if (uppercase(source_type) == "ISOTROPIC"sv) {
		source.beam = BeamType::Isotropic;
	}
	else {
		std::cerr << "Unknown photon source type." << std::endl;
		return false;
	}

	buf = next_dataline(input);
	std::vector<alpha_num> extracted_start;
	if (!extract(buf, extracted_start, {double {}, std::string {}}, true)) {
		std::cerr << "Error reading starting position of photon source." << std::endl;
		return false;
	}

	if (extracted_start.size() == 1) {
		source.z = std::get<double>(extracted_start[0]);
		source.layer_index = layer_index(source.z, params);

		if (source.layer_index == std::numeric_limits<std::size_t>::max()) {
			std::cerr << "Invalid starting position of photon source." << std::endl;
			return false;
		}
	}
	else if (extracted_start.size() == 2) {
		source.z = std::get<double>(extracted_start[0]);
		source.medium_name = std::get<std::string>(extracted_start[1]);

		if (source.medium_name[0] != '#' && source.medium_name[0] != '\n') {
			source.layer_index = layer_index(source.z, params);

			if (source.layer_index == std::numeric_limits<std::size_t>::max()) {
				std::cerr << "Invalid starting position of photon source." << std::endl;
				return false;
			}

			if (params.layers[source.layer_index].name == source.medium_name) {
				if ((std::fabs(source.z - params.layers[source.layer_index].z1)
					 < std::numeric_limits<double>::epsilon())
					&& (params.layers[source.layer_index + 1].name == source.medium_name)) {
					source.layer_index++;
					if (source.layer_index > params.num_layers) {
						std::cerr << "Source is outside of the last layer." << std::endl;
						return false;
					}
				}
				else {
					std::cerr << "Medium name and z coordinate do not match." << std::endl;
					return false;
				}
			}
		}
	}

	if (source.beam == BeamType::Isotropic && source.z == 0.0) {
		std::cerr << "Can not put an isotropic source in upper ambient medium." << std::endl;
		return false;
	}

	return true;
}

bool Reader::read_run_params(std::istream& input, RunParams& params) {
	if (!read_output(input, params.output_filename)) {
		std::cerr << "Error reading output file name." << std::endl;
		return false;
	}

	if (!read_layers(input, params, params.layers)) {
		std::cerr << "Error reading layers." << std::endl;
		return false;
	}

	if (!read_source(input, params, params.source)) {
		std::cerr << "Error reading source." << std::endl;
		return false;
	}

	if (!read_grid(input, params.grid)) {
		std::cerr << "Error reading grid." << std::endl;
		return false;
	}

	if (!read_record(input, params, params.record)) {
		std::cerr << "Error reading record." << std::endl;
		return false;
	}

	if (!read_target(input, params, params.target)) {
		std::cerr << "Error reading target." << std::endl;
		return false;
	}

	if (!read_weight(input, params.weight_threshold)) {
		std::cerr << "Error reading weight threshold." << std::endl;
		return false;
	}

	if (!read_seed(input, params.seed)) {
		std::cerr << "Error reading random number seed." << std::endl;
		return false;
	}

	// Compute the critical angles for total internal reflection according to the
	// relative refractive index of the layer.
	for (std::size_t i = 1; i <= params.num_layers; i++) {
		double eta_0 = params.layers[i - 1].eta;
		double eta_1 = params.layers[i].eta;
		double eta_2 = params.layers[i + 1].eta;

		params.layers[i].cos_theta_c0 = eta_1 > eta_0 ? std::sqrt(1.0 - eta_0 * eta_0 / (eta_1 * eta_1)) : 0.0;
		params.layers[i].cos_theta_c1 = eta_1 > eta_2 ? std::sqrt(1.0 - eta_2 * eta_2 / (eta_1 * eta_1)) : 0.0;
	}

	return true;
}

bool Reader::read_randomizer(std::istream& input, std::shared_ptr<Random>& random) {
	std::string buf;
	std::vector<std::mt19937::result_type> status(624);

	do {
		std::getline(input, buf);
	}
	while (buf[0] != '#');

	for (std::size_t i = 0; i < status.size(); i++) {
		input >> status[i];
	}

	// Restore the status
	random->restore_state(status);

	return true;
}

bool Reader::read_radiance(std::istream& input, RunParams& params, std::shared_ptr<Random>& random,
						   Radiance& radiance) {
	read_randomizer(input, random);

	// Skip comment line
	std::string buf = next_dataline(input);

	buf = next_dataline(input);
	std::istringstream iss(buf);
	iss >> radiance.R_spec;

	buf = next_dataline(input);
	iss = std::istringstream(buf);
	iss >> radiance.Rb_total >> radiance.Rb_error;

	buf = next_dataline(input);
	iss = std::istringstream(buf);
	iss >> radiance.R_total >> radiance.R_error;

	buf = next_dataline(input);
	iss = std::istringstream(buf);
	iss >> radiance.A_total >> radiance.A_error;

	buf = next_dataline(input);
	iss = std::istringstream(buf);
	iss >> radiance.Tb_total >> radiance.Tb_error;

	buf = next_dataline(input);
	iss = std::istringstream(buf);
	iss >> radiance.T_total >> radiance.T_error;

	// Reflectance
	if (params.record.R_rat) {
		radiance.R_rat = read_r_rat(input, params.grid.num_r, params.grid.num_a, params.grid.num_t);
	}
	if (params.record.R_ra) {
		radiance.R_ra = read_r_ra(input, params.grid.num_r, params.grid.num_a);
	}
	if (params.record.R_rt) {
		radiance.R_rt = read_r_rt(input, params.grid.num_r, params.grid.num_t);
	}
	if (params.record.R_at) {
		radiance.R_at = read_r_at(input, params.grid.num_a, params.grid.num_t);
	}
	if (params.record.R_r) {
		radiance.R_r = read_r_r(input, params.grid.num_r);
	}
	if (params.record.R_a) {
		radiance.R_a = read_r_a(input, params.grid.num_a);
	}
	if (params.record.R_t) {
		radiance.R_t = read_r_t(input, params.grid.num_t);
	}

	// Transmittance
	if (params.record.T_rat) {
		radiance.T_rat = read_t_rat(input, params.grid.num_r, params.grid.num_a, params.grid.num_t);
	}
	if (params.record.T_ra) {
		radiance.T_ra = read_t_ra(input, params.grid.num_r, params.grid.num_a);
	}
	if (params.record.T_rt) {
		radiance.T_rt = read_t_rt(input, params.grid.num_r, params.grid.num_t);
	}
	if (params.record.T_at) {
		radiance.T_at = read_t_at(input, params.grid.num_a, params.grid.num_t);
	}
	if (params.record.T_r) {
		radiance.T_r = read_t_r(input, params.grid.num_r);
	}
	if (params.record.T_a) {
		radiance.T_a = read_t_a(input, params.grid.num_a);
	}
	if (params.record.T_t) {
		radiance.T_t = read_t_t(input, params.grid.num_t);
	}

	// Absorption
	if (params.record.A_rzt) {
		radiance.Ab_zt = read_ab_zt(input, params.grid.num_z, params.grid.num_t);
		radiance.A_rzt = read_a_rzt(input, params.grid.num_r, params.grid.num_z, params.grid.num_t);
	}
	if (params.record.A_rz) {
		radiance.Ab_z = read_ab_z(input, params.grid.num_z);
		radiance.A_rz = read_a_rz(input, params.grid.num_r, params.grid.num_z);
	}
	if (params.record.A_zt) {
		radiance.A_zt = read_a_zt(input, params.grid.num_z, params.grid.num_t);
	}
	if (params.record.A_z) {
		radiance.A_z = read_a_z(input, params.grid.num_z);
	}
	if (params.record.A_t) {
		radiance.A_t = read_a_t(input, params.grid.num_t);
	}

	return true;
}

void Reader::skip_line(std::istream& input, std::size_t num_lines) {
	for (std::size_t i = 0; i < num_lines; i++) {
		auto line = next_dataline(input);
		if (line.empty()) {
			break;
		}
	}
}

std::string Reader::next_dataline(std::istream& in) {
	std::string line;
	while (std::getline(in, line)) {
		// Find first non-whitespace character
		auto it = std::ranges::find_if(line, [](char c) { return !std::isspace(c); });

		// Skip whitespace-only lines
		if (it == line.end()) {
			continue;
		}

		// Skip comment lines
		if (*it == '#') {
			continue;
		}

		return {line, false};
	}

	// Return empty string if no valid lines found
	std::cerr << "Error: No valid data line found." << std::endl;
	return {{}, false};
}

bool Reader::check_input_params(RunParams& params) {
	for (std::size_t i = 0; i <= params.num_layers + 1; i++) {
		// Find index of the medium name in the medium list.
		auto it = std::ranges::find_if(params.mediums, [&](const Layer& m) {
			return std::ranges::any_of(params.layers, [&](const Layer& l) { return l.name == m.name; });
		});

		if (it == params.mediums.end()) {
			return std::cerr << "Invalid medium name of layer " << i << ".\n", 0;
		}

		std::size_t index = static_cast<std::size_t>(std::distance(params.mediums.begin(), it));
		params.layers[i].eta = params.mediums[index].eta;
		params.layers[i].mu_a = params.mediums[index].mu_a;
		params.layers[i].mu_s = params.mediums[index].mu_s;
		params.layers[i].g = params.mediums[index].g;
	}

	if ((params.source.beam == BeamType::Isotropic) && (params.source.z == 0.0)) {
		return std::cerr << "Can not put isotropic source in upper ambient medium.\n", 0;
	}

	// Find the index of the layer according to the z coordinate.
	if (params.source.z < 0.0) {
		return std::cerr << "Nonpositive z coordinate.\n", false;
	}
	else if (params.source.z > params.layers.back().z1) {
		return std::cerr << "Source is outside of the last layer.\n", false;
	}
	else {
		params.source.layer_index = static_cast<std::size_t>(
			std::ranges::lower_bound(params.layers, params.source.z, {}, &Layer::z1) - params.layers.begin());
	}

	// Check the medium name and z coordinate of the source.
	if (params.source.medium_name[0] != '\0') {
		if (params.layers[params.source.layer_index].name == params.source.medium_name) {
			if ((std::abs(params.source.z - params.layers[params.source.layer_index].z1)
				 < std::numeric_limits<double>::epsilon())
				&& (params.layers[params.source.layer_index + 1].name == params.source.medium_name)) {
				params.source.layer_index++;
			}
			else {
				std::cerr << "Medium name and z coordinate do not match." << std::endl;
				return false;
			}
		}
	}
	return true;
}

vec3<double> Reader::read_r_rat(std::istream& input, std::size_t Nr, std::size_t Na, std::size_t Nt) {
	next_dataline(input);
	vec3<double> R_rat(Nr, vec2<double>(Na, vec1<double>(Nt)));

	for (std::size_t ir = 0; ir < Nr; ir++) {
		for (std::size_t ia = 0; ia < Na; ia++) {
			for (std::size_t it = 0; it < Nt; it++) {
				input >> R_rat[ir][ia][it];
			}
		}
	}
	return R_rat;
}

vec2<double> Reader::read_r_ra(std::istream& input, std::size_t Nr, std::size_t Na) {
	next_dataline(input);
	vec2<double> R_ra(Nr, vec1<double>(Na));

	for (std::size_t ir = 0; ir < Nr; ir++) {
		for (std::size_t ia = 0; ia < Na; ia++) {
			input >> R_ra[ir][ia];
		}
	}
	return R_ra;
}

vec2<double> Reader::read_r_rt(std::istream& input, std::size_t Nr, std::size_t Nt) {
	next_dataline(input);
	vec2<double> R_rt(Nr, vec1<double>(Nt));

	for (std::size_t ir = 0; ir < Nr; ir++) {
		for (std::size_t it = 0; it < Nt; it++) {
			input >> R_rt[ir][it];
		}
	}
	return R_rt;
}

vec2<double> Reader::read_r_at(std::istream& input, std::size_t Na, std::size_t Nt) {
	next_dataline(input);
	vec2<double> R_at(Na, vec1<double>(Nt));

	for (std::size_t ia = 0; ia < Na; ia++) {
		for (std::size_t it = 0; it < Nt; it++) {
			input >> R_at[ia][it];
		}
	}
	return R_at;
}

vec1<double> Reader::read_r_r(std::istream& input, std::size_t Nr) {
	next_dataline(input);
	vec1<double> R_r(Nr);

	for (std::size_t ir = 0; ir < Nr; ir++) {
		input >> R_r[ir];
	}
	return R_r;
}

vec1<double> Reader::read_r_a(std::istream& input, std::size_t Na) {
	next_dataline(input);
	vec1<double> R_a(Na);

	for (std::size_t ia = 0; ia < Na; ia++) {
		input >> R_a[ia];
	}
	return R_a;
}

vec1<double> Reader::read_r_t(std::istream& input, std::size_t Nt) {
	next_dataline(input);
	vec1<double> R_t(Nt);

	for (std::size_t it = 0; it < Nt; it++) {
		input >> R_t[it];
	}
	return R_t;
}

vec3<double> Reader::read_t_rat(std::istream& input, std::size_t Nr, std::size_t Na, std::size_t Nt) {
	next_dataline(input);
	vec3<double> T_rat(Nr, vec2<double>(Na, vec1<double>(Nt)));

	for (std::size_t ir = 0; ir < Nr; ir++) {
		for (std::size_t ia = 0; ia < Na; ia++) {
			for (std::size_t it = 0; it < Nt; it++) {
				input >> T_rat[ir][ia][it];
			}
		}
	}
	return T_rat;
}

vec2<double> Reader::read_t_ra(std::istream& input, std::size_t Nr, std::size_t Na) {
	next_dataline(input);
	vec2<double> T_ra(Nr, vec1<double>(Na));

	for (std::size_t ir = 0; ir < Nr; ir++) {
		for (std::size_t ia = 0; ia < Na; ia++) {
			input >> T_ra[ir][ia];
		}
	}
	return T_ra;
}

vec2<double> Reader::read_t_rt(std::istream& input, std::size_t Nr, std::size_t Nt) {
	next_dataline(input);
	vec2<double> T_rt(Nr, vec1<double>(Nt));

	for (std::size_t ir = 0; ir < Nr; ir++) {
		for (std::size_t it = 0; it < Nt; it++) {
			input >> T_rt[ir][it];
		}
	}
	return T_rt;
}

vec2<double> Reader::read_t_at(std::istream& input, std::size_t Na, std::size_t Nt) {
	next_dataline(input);
	vec2<double> T_at(Na, vec1<double>(Nt));

	for (std::size_t ia = 0; ia < Na; ia++) {
		for (std::size_t it = 0; it < Nt; it++) {
			input >> T_at[ia][it];
		}
	}
	return T_at;
}

vec1<double> Reader::read_t_r(std::istream& input, std::size_t Nr) {
	next_dataline(input);
	vec1<double> T_r(Nr);

	for (std::size_t ir = 0; ir < Nr; ir++) {
		input >> T_r[ir];
	}
	return T_r;
}

vec1<double> Reader::read_t_a(std::istream& input, std::size_t Na) {
	next_dataline(input);
	vec1<double> T_a(Na);

	for (std::size_t ia = 0; ia < Na; ia++) {
		input >> T_a[ia];
	}
	return T_a;
}

vec1<double> Reader::read_t_t(std::istream& input, std::size_t Nt) {
	next_dataline(input);
	vec1<double> T_t(Nt);

	for (std::size_t it = 0; it < Nt; it++) {
		input >> T_t[it];
	}
	return T_t;
}

vec3<double> Reader::read_a_rzt(std::istream& input, std::size_t Nr, std::size_t Nz, std::size_t Nt) {
	next_dataline(input);
	vec3<double> A_rzt(Nr, vec2<double>(Nz, vec1<double>(Nt)));

	for (std::size_t ir = 0; ir < Nr; ir++) {
		for (std::size_t iz = 0; iz < Nz; iz++) {
			for (std::size_t it = 0; it < Nt; it++) {
				input >> A_rzt[ir][iz][it];
			}
		}
	}
	return A_rzt;
}

vec2<double> Reader::read_a_rz(std::istream& input, std::size_t Nr, std::size_t Nz) {
	next_dataline(input);
	vec2<double> A_rz(Nr, vec1<double>(Nz));

	for (std::size_t ir = 0; ir < Nr; ir++) {
		for (std::size_t iz = 0; iz < Nz; iz++) {
			input >> A_rz[ir][iz];
		}
	}
	return A_rz;
}

vec2<double> Reader::read_a_zt(std::istream& input, std::size_t Nz, std::size_t Nt) {
	next_dataline(input);
	vec2<double> A_zt(Nz, vec1<double>(Nt));

	for (std::size_t iz = 0; iz < Nz; iz++) {
		for (std::size_t it = 0; it < Nt; it++) {
			input >> A_zt[iz][it];
		}
	}
	return A_zt;
}

vec1<double> Reader::read_a_z(std::istream& input, std::size_t Nz) {
	next_dataline(input);
	vec1<double> A_z(Nz);

	for (std::size_t iz = 0; iz < Nz; iz++) {
		input >> A_z[iz];
	}
	return A_z;
}

vec1<double> Reader::read_a_t(std::istream& input, std::size_t Nt) {
	next_dataline(input);
	vec1<double> A_t(Nt);

	for (std::size_t it = 0; it < Nt; it++) {
		input >> A_t[it];
	}
	return A_t;
}

vec2<double> Reader::read_ab_zt(std::istream& input, std::size_t Nz, std::size_t Nt) {
	next_dataline(input);
	vec2<double> Ab_zt(Nz, vec1<double>(Nt));

	for (std::size_t iz = 0; iz < Nz; iz++) {
		for (std::size_t it = 0; it < Nt; it++) {
			input >> Ab_zt[iz][it];
		}
	}
	return Ab_zt;
}

vec1<double> Reader::read_ab_z(std::istream& input, std::size_t Nz) {
	next_dataline(input);
	vec1<double> Ab_z(Nz);

	for (std::size_t iz = 0; iz < Nz; iz++) {
		input >> Ab_z[iz];
	}
	return Ab_z;
}
