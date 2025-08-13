#include "cin_reader.hpp"
#include "cin_reader.tpp"

#include <regex>
#include <format>
#include <fstream>
#include <numbers>
#include <iostream>
#include <algorithm>
#include <filesystem>


bool CinReader::ReadParams(std::istream& input, RunParams& params)
{
    if (!ReadMediums(*m_input, params.mediums)) {
        std::cerr << "Error: Failed to read mediums." << std::endl;
        return false;
    }
    
    if (!ReadOutput(*m_input, params.output_filename)) {
        std::cerr << "Error: Failed to read output filename." << std::endl;
        return false;
    }
    
    if (!ReadLayers(*m_input, params, params.layers)) {
        std::cerr << "Error: Failed to read layers." << std::endl;
        return false;
    }
    
    if (!ReadSource(*m_input, params, params.source)) {
        std::cerr << "Error: Failed to read source." << std::endl;
        return false;
    }
    
    if (!ReadGrid(*m_input, params.grid)) {
        std::cerr << "Error: Failed to read grid parameters." << std::endl;
        return false;
    }
    
    if (!ReadRecord(*m_input, params, params.record)) {
        std::cerr << "Error: Failed to read record." << std::endl;
        return false;
    }
    
    if (!ReadTarget(*m_input, params, params.target)) {
        std::cerr << "Error: Failed to read target." << std::endl;
        return false;
    }
    
    if (!ReadWeight(*m_input, params.weight_threshold)) {
        std::cerr << "Error: Failed to read weight threshold." << std::endl;
        return false;
    }

    return true;
}

bool CinReader::ReadMediums(std::istream& in, vec1<Layer>& out)
{
    std::string prompt = "Specify number of mediums (>= 1)";
    std::string error = "Invalid number of mediums";

    auto [success, num_media] = read_in<int>(in, prompt, error, false, [](const int& num_media) {
        return num_media >= 1;
    });
    if (!success) return false;

    // Allocate space for the layer parameters.
    vec1<Layer> mediums;
    mediums.resize(num_media);

    for (std::size_t i = 0; i < num_media; i++) {
        std::cout << std::format("Specify medium {}.\n", i + 1);

        auto [s1, name] = read_in<std::string>(in, "  Name of medium", "Invalid medium name or duplicate", false, [&mediums](const std::string& name) {
            return !name.empty() && std::ranges::none_of(mediums, [&](const Layer& l) { return l.name == name; });
        });
        if (!s1) return false;

        auto [s2, eta] = read_in<double>(in, "  Refractive index n (>= 1.0): ", {}, false, [&mediums](const double& value) { 
            return value >= 1.0;
        });
        if (!s2) return false;

        auto [s3, mua] = read_in<double>(in, "  Absorption coefficient mua (>= 0.0 /cm): ", {}, false, [&mediums](const double& value) { 
            return value >= 0.0;
        });
        if (!s3) return false;

        auto [s4, mus] = read_in<double>(in, "  Scattering coefficient mus (>= 0.0 /cm): ", {}, false, [&mediums](const double& value) { 
            return value >= 0.0;
        });
        if (!s4) return false;

        auto [s5, g] = read_in<double>(in, "  Anisotropy factor g (0.0 - 1.0): ", {}, false, [&mediums](const double& value) { 
            return value >= 0.0 && value <= 1.0;
        });
        if (!s5) return false;

        mediums[i] = { i, name, eta, mua, mus, g };
        std::cout << std::endl;
    }

    std::for_each(mediums.begin(), mediums.end(), [&](Layer& mediums) {
        out.push_back(mediums);
    });

    return true;
}

bool CinReader::ReadOutput(std::istream& in, std::string& out)
{
    std::string file_name;
    do {
        std::cout << "Specify output filename with extension .mco: ";
        std::getline(std::cin, file_name);

        // Check if the extension is .mco
        if (file_name.size() < 4 || file_name.substr(file_name.size() - 4) != ".mco") {
            std::cerr << "Error: File must have a .mco extension." << std::endl;
            file_name.clear();
            continue;
        }

        // Check file existence and permissions
        std::ifstream file(file_name, std::ios::in);

        if (file.is_open()) { // File exists
            std::cout << "File " << file_name << " exists, w=overwrite, n=new filename: ";

            // Avoid null line and get valid input
            char file_mode;
            do {
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cin.get(file_mode);
            } while (file_mode != 'n' && file_mode != 'w');
            
            file.close();

            if (file_mode == 'n') {
                file_name.clear();
                continue;
            }
        }

        // Check if path is valid
        std::filesystem::path file_path(file_name);
        if (!file_path.has_parent_path()) {
            std::cerr << "Error: Invalid file path." << std::endl;
            file_name.clear();
            continue;
        }

        // Create file and report if it is writable
        std::ofstream test_create(file_name, std::ios::app);
        if (!test_create.is_open()) {
            std::cerr << "Error: Unable to create or write to the file." << std::endl;
            test_create.close();
            file_name.clear();
            continue;
        }
        test_create.close();
    } while (file_name.empty());

    std::cout << std::endl;

    out = file_name;
    return true;
}

bool CinReader::ReadLayers(std::istream& in, RunParams& params, vec1<Layer>& out)
{
    auto make_layer = [&](std::size_t i, double z0, double z1) -> Layer {
        bool name_exists{ false };

        auto [success, name] = read_in<std::string>(in, {}, "Invalid or duplicate medium name", false, [&params](const std::string& name) {
            return !name.empty() && std::ranges::none_of(params.mediums, [&](const Layer& l) { return l.name == name; });
        });
        if (!success) { throw std::runtime_error({}); }

        return Layer{ i,
            params.mediums[i].name,
            params.mediums[i].eta,
            params.mediums[i].mu_a,
            params.mediums[i].mu_s,
            params.mediums[i].g,
            z0, z1, 0.0, 0.0
        };
    };

    std::cout << std::endl << "Specify layer list. Available medium types:" << std::endl;
    for (int i = 0, j = 1; i < params.mediums.size(); i++, j++) {
        std::cout << std::format("{:<16}", params.mediums[i].name);
        if (j % 4 == 0) {
            std::cout << std::endl;
        }
    }

    auto [success, num_layers] = read_in<int>(in, "  Total number of layers: ", "Invalid layer number", false, [](const int& num_layers) {
        return num_layers >= 1;
    });
    if (!success) { return false; }

    vec1<Layer> layers;
    layers.resize(num_layers);

    // z coordinate of the current layer
    double z{ 0.0 };

    // First and last layers are for ambient
    for (std::size_t i = 0; i < num_layers; ++i) {
        Layer layer{};

        if (i == 0) {
            std::cout << std::endl << "  Name of upper ambient medium: ";
            try { layer = make_layer(i, 0.0, 0.0); }
            catch (const std::exception&) { return false; }
        }
        else if (i == num_layers - 1) {
            std::cout << std::endl << "  Name of lower ambient medium: ";
            try { layer = make_layer(i, z, z); }
            catch (const std::exception&) { return false; }
        }
        else {
            auto [success, thickness] = read_in<double>(in, "  Thickness of layer (thickness > 0.0 cm): ", {}, false, [](const double& thickness) {
                return thickness > 0.0;
            });
            if (!success) { return false; }

            std::cout << std::endl << "  Name of layer " << i << ": ";
            layer = make_layer(i, z, z + thickness);

            z += thickness;
        }

        layers[i] = layer;
    }

    std::for_each(layers.begin(), layers.end(), [&](Layer& layer) {
        out.push_back(layer);
    });

    std::cout << std::endl;
    return true;
}

bool CinReader::ReadSource(std::istream& input, RunParams& params, LightSource& out)
{
    // Compute the index to layer according to the z coordinate.
    // If the z is on an interface between layers, the returned index will point to the upper layer.
    auto layerIndex = [&](double z, RunParams& params) -> std::size_t {
        for (std::size_t i = 1; i <= params.num_layers; i++) {
            if (z >= params.layers[i].z0 && z <= params.layers[i].z1) {
                return i;
            }
        }
        return std::numeric_limits<std::size_t>::max();
    };

    std::size_t layer_index{};

    std::string prompt = "Input source type (P = pencil / I = isotropic)";
    auto [s1, source_type] = read_in<char>(input, prompt, "Invalid source type", false, [](char source_type) {
        return std::regex_match(std::string(1, source_type), std::regex("[PpIi]")); // P or I
    });
    if (!s1) { return false; }

    BeamType beam_type = (std::toupper(source_type) == 'P') ? BeamType::Pencil : BeamType::Isotropic;

    prompt = std::format("Input source z-coordinate (0.0 - {} cm. ", params.layers[params.num_layers + 1].z1);
    auto [s2, source_z] = read_in<double>(input, prompt, "Invalid starting position of photon source", true, 
                                            [&params, &layer_index, layerIndex](const double& z) {
        // Check if the source is on an interface
        layer_index = layerIndex(z, params);
        if (layer_index == std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        return z >= 0.0 && z <= params.layers[params.num_layers + 1].z1;
    });
    if (!s2) { return false; }

    // Check if the source is isotropic and in upper ambient medium
    if (beam_type == BeamType::Isotropic && source_z == 0.0) {
        std::cerr << "Can not put an isotropic source in upper ambient medium." << std::endl;
        return false;
    }

    out.z = source_z;
    out.beam = beam_type;
    out.layer_index = layer_index;
    out.medium_name = params.layers[layer_index].name;

    std::cout << "Saved." << std::endl;
    return true;
}

bool CinReader::ReadGrid(std::istream& in, Grid& out)
{
    using namespace std;

    std::string prompt = "Specify dz, dr, dt in one line (all > 0.0 cm, e.g., 0.1 0.1 0.1)";
    std::string error = "";

    auto [s1, dz, dr, dt] = read_in<double, double, double>(in, prompt, error, false, [](const std::tuple<double, double, double>& t) {
        return std::get<0>(t) > 0.0 && std::get<1>(t) > 0.0 && std::get<2>(t) > 0.0;
    });
    if (!s1) { return false; }

    prompt = "Specify nz, nr, nt, na in one line (all > 0, e.g., 100 100 100 100)";

    auto [s2, nz, nr, nt, na] = read_in<int, int, int, int>(in, prompt, error, false, [](const std::tuple<int, int, int, int>& t) {
        return std::get<0>(t) > 0 && std::get<1>(t) > 0 && std::get<2>(t) > 0 && std::get<3>(t) > 0;
    });
    if (!s2) { return false; }

    double da = 0.5 * std::numbers::pi / na;

    out = Grid{
        .step_z = dz,
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
        .max_t = dt * nt
    };

    return true;
}

bool CinReader::ReadRecord(std::istream& input, RunParams& params, Record& out)
{
    std::cout << "Select scored quantities from the following data categories:" << std::endl;
    std::cout << "\tR_rat\t\t\tT_rat\t\t\tA_rzt" << std::endl;
    std::cout << "\tR_ra\tR_rt\tR_at\tT_ra\tT_rt\tR_at\tA_rz\tA_zt" << std::endl;
    std::cout << "\tR_r\tR_a\tR_t\tT_r\tT_a\tT_t\tA_z\tA_t" << std::endl;

    if (!Reader::ReadRecord(input, params, out)) {
        return false;
    }

    std::cout << std::endl;
    return true;
}

bool CinReader::ReadTarget(std::istream& input, RunParams& params, Target& out, bool add)
{
    if (add) {
        std::cout << params.target.photons_limit - params.target.photons_remaining;
        std::cout << " photons have been traced in the previous simulation.\n";
        std::cout << "Specify additional photons or compution time in hh:mm format,\n";
        std::cout << "or both in one line (e.g. 10000 5:30): ";
    }
    else {
        std::cout << "Specify number of photons or time in hh:mm format,\n";
        std::cout << "or both in one line (e.g. 10000 5:30): ";
    }

    if (Reader::ReadTarget(input, params, out, add)) {
        return false;
    }

    std::cout << std::endl;
    return true;
}

bool CinReader::ReadWeight(std::istream& input, double& out)
{
    std::string prompt = "Input weight threshold (0 <= W < 1.0, 0.0001 recommended)";
    auto [success, weight] = read_in<double>(input, prompt, {}, false, [](const double& weight) {
        return weight >= 0.0 && weight < 1.0;
    });
    if (!success) { return false; }

    std::cout << std::endl;
    return true;
}
