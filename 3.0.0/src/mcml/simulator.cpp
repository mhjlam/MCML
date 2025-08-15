#include "simulator.hpp"

#include <chrono>
#include <format>
#include <fstream>
#include <numbers>
#include <iostream>
#include <filesystem>

#include "reader.hpp"
#include "cin_reader.hpp"

#include "writer.hpp"
#include "cout_writer.hpp"

#include "timer.hpp"
#include "random.hpp"
#include "tracer.hpp"
#include "exception.hpp"


using namespace std::chrono;


Simulator::Simulator(std::string in_file) :
    m_mci{ in_file }, m_params{},
    m_reader{ std::make_shared<Reader>(in_file) },
    m_cin_reader{ std::make_shared<CinReader>() },
    m_writer{ std::make_shared<Writer>() },
    m_cout_writer{ std::make_shared<CoutWriter>() },
    m_random{ std::make_shared<Random>() },
    m_timer{ std::make_shared<Timer>() },
    m_tracer{ std::make_shared<Tracer>(m_params, m_random) }
{
    m_random->seed(1);
}

void Simulator::Simulate()
{
    bool retry = false;
    do {
        retry = !promptFileName(m_mci);
        if (m_mci == "Q") {
            return;
        }
    } while (retry);
    
    m_reader = std::make_shared<Reader>(m_mci);
    m_reader->ReadParams(*m_reader, m_params);

    std::size_t run_i = 1;
    while (run_i <= m_params.num_runs) {
        m_reader->ReadRunParams(*m_reader, m_params);
        m_tracer = std::make_unique<Tracer>(m_params, m_random);
        run(run_i++);
    }
}

void Simulator::Resume()
{
    std::cout << "Specify the output file name of a previous simulation. " << std::endl;

    std::string filename;
    bool retry = false;
    do {
        retry = promptFileName(filename, ".mco");
        if (filename == "Q") {
            return;
        }
    } while (retry);

    try {
        auto file_reader = std::make_unique<Reader>(filename, MCO_VERSION);

        // Skip file version number
        file_reader->SkipLine(*file_reader);
        file_reader->ReadMediums(*file_reader, m_params.mediums);

        if (m_params.mediums.empty()) {
            throw std::runtime_error("No mediums found in file.");
        }

        file_reader->ReadRunParams(*file_reader, m_params);

        // Add photons / time limit
        m_cin_reader->ReadTarget(*m_cin_reader, m_params, m_params.target, true);

        Radiance radiance;
        file_reader->ReadRadiance(*file_reader, m_params, m_random, radiance);
        m_tracer = std::make_shared<Tracer>(m_params, m_random, radiance);

        scaleReflectance(*m_tracer, ScaleMode::Unscale);
        scaleTransmittance(*m_tracer, ScaleMode::Unscale);
        scaleAbsorption(*m_tracer, ScaleMode::Unscale);

        run();

        std::exit(0);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::exit(1);
    }
}

void Simulator::Interactive()
{
    // Read params interactively
    m_cin_reader->ReadParams(*m_cin_reader, m_params);

    std::cout << "Do you want to save the input to a file? (Y/N) ";
    char c; std::cin.get(c);
    std::cin.ignore(max_size, '\n');

    if (std::toupper(c) == 'Y') {
        std::string string;
        while (string.empty()) {
            std::cout << "Give the file name to save input: ( .mci): ";
            std::getline(std::cin, string);
        }

        std::fstream mci_file(string, std::ios::out);
        if (!mci_file.is_open()) {
            std::cout << "Can not open the file to write." << std::endl;
        }
        else {
            m_writer->WriteParams(mci_file, m_params);
        }
    }

    if (interactiveRun()) {
        m_tracer = std::make_unique<Tracer>(m_params, m_random);
        run();
    }
}

void Simulator::InteractiveEdit()
{
    // Read mci file
    bool retry = false;
    std::string filename;

    do {
        retry = !promptFileName(filename);
        if (filename == "Q") {
            return;
        }
    } while (retry);
    
    // Read mediums and params from input file
    auto file_reader = std::make_unique<Reader>(filename);
    file_reader->ReadMediums(*file_reader, m_params.mediums);
    file_reader->ReadRunParams(*file_reader, m_params);
    std::cout << "The parameters of the first run have been read in.\n\n";

    if (interactiveRun()) {
        m_tracer = std::make_unique<Tracer>(m_params, m_random);
        run();
    }
}


void Simulator::run(std::size_t run_index, bool start_new)
{
    Radiance& radiance = *m_tracer;

    if (start_new) {
        if (m_params.source.layer_index == 0) {
            double eta0 = m_params.layers[0].eta;
            double eta1 = m_params.layers[1].eta;
            double r = (eta0 - eta1) / (eta0 + eta1);
            radiance.R_spec = r * r;
        }

        m_random->seed(1);
    }

    m_timer->reset();

    reportTarget(m_params.num_runs - run_index);

    bool exit = false;
    std::size_t photon_traced = 1;
    std::size_t tens = 10;

    do {
        Photon photon = m_tracer->Launch();
        m_tracer->Trace(photon);

        if (photon_traced == tens) {
            tens *= 10;
            reportProgress(photon_traced);
        }
        photon_traced++;

        switch (m_params.target.control_bit) {
            case ControlBit::NumPhotons:
            {
                exit = photon_traced > m_params.target.photons_limit;
                break;
            }
            case ControlBit::TimeLimit:
            {
                exit = m_timer->punch() >= m_params.target.time_limit;
                break;
            }
            case ControlBit::Both:
            {
                exit = (photon_traced > m_params.target.photons_limit) || (m_timer->punch() >= m_params.target.time_limit);
                break;
            }
        }
    } while (!exit);

    // Update remaining photons and time
    m_params.target.photons_remaining = m_params.target.photons_limit - photon_traced;
    m_params.target.time_remaining = m_params.target.time_limit - static_cast<long>(m_timer->punch());

    reportResult();

    m_tracer.reset();
    m_params.layers.clear();
}

bool Simulator::interactiveRun()
{
    char command{ '\0' };
    std::string string;
    bool start_sim;

    std::cout << "Any changes to the input parameters? (Y/N) ";
    do {
        std::cin.get(command);
        command = std::toupper(command);
    } while (command != 'Y' && command != 'N');

    std::cin.ignore(max_size, '\n');

    // No changes, return true and start simulation
    if (command == 'N') {
        return true;
    }

    bool cont = true;
    while (cont) {
        cont = false;
        do {
            std::cout << std::endl << "> Change menu (h for help) => ";
            do {
                std::cin.get(command);
                command = std::toupper(command);
            } while (command == '\0' || command == '\n');

            std::cin.ignore(max_size, '\n');
            start_sim = editMenu(command);

            // 'X' or 'Q'
            if (start_sim) {
                break;
            }
        } while (true);

        std::cout << "Do you want to save the input to a file? (Y/N) ";
        do {
            std::cin.get(command);
        } while (command == '\0' || command == '\n');

        if (std::toupper(command) == 'Y') {
            std::cout << "Give the file name to save input: ( .mci): ";
            std::cin.ignore(max_size, '\n');
            std::getline(std::cin, string);

            std::ofstream file(string, std::ios::out);
            if (!file.is_open()) {
                throw std::runtime_error("Can not open the file to write.");
            }

            m_writer->WriteParams(file, m_params);
        }

        // Quit change menu and start simulation
        if (start_sim && !validateParams()) {
            std::cout << "Change input or exit to main menu (C/X): ";
            do {
                std::cin.get(command);

                if (std::toupper(command) == 'X') {
                    m_params.mediums.clear();
                    m_params.layers.clear();
                    return false;
                }
            } while (command == '\0' || command == '\n');

            // Continue to change parameters
            cont = true;
        }
    }

    return true;
}

bool Simulator::editMenu(char command)
{
    switch (std::toupper(command)) {
        case 'O': { m_writer->WriteParams(std::cout, m_params); break; }
        case 'M': { editMediums(); break; }
        case 'F': { editOutput(); break; }
        case 'G': { editGrid(); break; }
        case 'C': { editRecord(); break; }
        case 'W': { editWeight(); break; }
        case 'L': { editLayers(); break; }
        case 'P': { editTarget(); break; }
        case 'S': { editSource(); break; }
        case 'H': { showEditMenuHelp(); break; }
        case 'Q': { return true; }
        default: { std::cout << "Unknown command." << std::endl; }
    }
    return false;
}

bool Simulator::validateParams()
{
    auto layer_index = [&](double z, RunParams& params) -> std::size_t {
        for (std::size_t i = 1; i <= params.num_layers; i++) {
            if (z >= params.layers[i].z0 && z <= params.layers[i].z1) {
                return i;
            }
        }
        return std::numeric_limits<std::size_t>::max();
    };

    for (int i = 0; i <= m_params.num_layers + 1; i++) {
        std::size_t index{};
        bool name_exists = std::ranges::any_of(m_params.mediums, [&](const Layer& medium) {
            if (medium.name == m_params.layers[i].name) {
                index = medium.index;
            }
            return medium.name == m_params.layers[i].name;
        });

        if (!name_exists) {
            throw std::runtime_error("Invalid medium name of layer " + std::to_string(i) + ".");
        }

        m_params.layers[i].index = index;
        m_params.layers[i].eta = m_params.mediums[index].eta;
        m_params.layers[i].mu_a = m_params.mediums[index].mu_a;
        m_params.layers[i].mu_s = m_params.mediums[index].mu_s;
        m_params.layers[i].g = m_params.mediums[index].g;
    }

    if (m_params.source.beam == BeamType::Isotropic && m_params.source.z == 0.0) {
        throw std::runtime_error("Can not put isotropic source in upper ambient medium.");
    }

    if (layer_index(m_params.source.z, m_params) == std::numeric_limits<std::size_t>::max()) {
        return false;
    }

    if (!m_params.source.medium_name.empty()) { // TODO: write into a function
        if (m_params.layers[m_params.source.layer_index].name == m_params.source.medium_name) {
            if ((std::fabs(m_params.source.z - m_params.layers[m_params.source.layer_index].z1) < std::numeric_limits<double>::epsilon()) &&
                (m_params.layers[m_params.source.layer_index + 1].name == m_params.source.medium_name)) {
                m_params.source.layer_index++;
            }
            else {
                throw std::runtime_error("Medium name and z coordinate do not match.");
            }
        }
    }
    return true;
}


void Simulator::reportTarget(std::size_t runs_remaining)
{
    using namespace std::chrono;
    auto [h, m, s] = Timer::time_point_hms(m_params.target.time_limit);

    std::cout << std::endl << "Starting run #" << m_params.num_runs - runs_remaining << ". ";
    switch (m_params.target.control_bit) {
        case ControlBit::NumPhotons:
            std::cout << "Tracing " << m_params.target.photons_limit << " photons.\n\n";
            std::cout << "\tPhotons traced\n";
            std::cout << "\t--------------\n";
            break;

        case ControlBit::TimeLimit:
            std::cout << "Tracing photons until the deadline in ";
            std::cout << Timer::format_hms(h, m, s);
            std::cout << Timer::format_datetime(m_params.target.time_limit) << ".\n\n";
            std::cout << "\tPhotons traced\n";
            std::cout << "\t--------------\n";
            break;

        case ControlBit::Both:
            std::cout << "Tracing " << m_params.target.photons_limit << " photons with a deadline in ";
            std::cout << Timer::format_hms(h, m, s);
            std::cout << Timer::format_datetime(m_params.target.time_limit) << ".\n\n";
            std::cout << "\tPhotons traced\n";
            std::cout << "\t--------------\n";
            break;
    }
}

void Simulator::reportProgress(std::size_t photons_done)
{
    if (m_params.target.control_bit == ControlBit::TimeLimit) {
        std::cout << std::format("\t{:<13}\t", photons_done) << std::endl;
    }
    else {
        std::cout << std::format("{:>12} ({:6.2f}%)\t", photons_done, (float)photons_done * 100 / m_params.target.photons_limit) << std::endl;
    }
}

void Simulator::reportResult()
{
    auto [h, m, s] = Timer::time_point_hms(m_timer->punch());

    std::cout << "\nFinished tracing " << m_params.target.photons_limit << " photons. ";
    std::cout << "This took " << Timer::format_hms(h, m, s) << ".\n";

    // Scale results
    Radiance& radiance = *m_tracer;
    scaleReflectance(radiance);
    scaleTransmittance(radiance);
    scaleAbsorption(radiance);

    std::fstream file(m_params.output_filename, std::ios::out);

    if (!file.is_open()) {
        std::cout << "Can not open output file to write." << std::endl;
        std::exit(1);
    }

    // Write results to file
    m_writer->WriteResults(file, m_params, *m_tracer, m_random);
}

bool Simulator::promptFileName(std::string& result, std::string file_type)
{
    std::string buf;

    while (buf.empty()) {
        std::cout << "Specify path to " << file_type << " file (or Q to quit to main menu): ";

        // Read input buffer
        std::getline(std::cin, buf);

        if (!buf.empty()) {
            // Terminate with letter Q
            if (buf.size() == 1 && std::toupper(buf[0]) == 'Q') {
                result = "Q";
                return true;
            }
        }

        // Check if file exists
        if (!std::filesystem::exists(buf)) {
            std::cout << "File does not exist." << std::endl;
            return false;
        }
    }

    result = buf;
    return true;
}


bool Simulator::promptEdit()
{
    char ch{ '\0' };
    std::cout << "Do you want to change them? (Y/N): ";
    do {
        std::cin.get(ch);
        ch = std::toupper(ch);
    } while (ch != 'Y' && ch != 'N');

    std::cin.ignore(max_size, '\n');
    return ch == 'Y';
}

void Simulator::editMediums()
{
    std::cout << "Current medium list: " << std::endl;
    m_cout_writer->WriteMediums(*m_cout_writer, m_params);
    std::cout << std::endl;

    if (promptEdit()) {
        m_params.mediums.clear();
        m_cin_reader->ReadMediums(*m_cin_reader, m_params.mediums);
    }
}

void Simulator::editOutput()
{
    std::cout << "Current output file name and format: " << std::endl;
    m_cout_writer->WriteFilename(*m_cout_writer, m_params);
    std::cout << std::endl;

    m_cin_reader->ReadOutput(*m_cin_reader, m_params.output_filename);
}

void Simulator::editGrid()
{
    std::cout << "Current dz, dr, dt: " << std::endl;
    m_cout_writer->WriteGridParams(*m_cout_writer, m_params);
    std::cout << std::endl;

    std::cout << "Current dz, dr, dt: " << std::endl;
    m_cout_writer->WriteGridParams(*m_cout_writer, m_params);
    std::cout << std::endl;

    m_cin_reader->ReadGrid(*m_cin_reader, m_params.grid);
}

void Simulator::editRecord()
{
    m_cout_writer->WriteRecord(*m_cout_writer, m_params);
    std::cout << std::endl;

    if (promptEdit()) {
        m_cin_reader->ReadRecord(*m_cin_reader, m_params, m_params.record);
    }
}

void Simulator::editWeight()
{
    std::cout << "Current threshold weight: " << std::endl;
    m_cout_writer->WriteWeight(*m_cout_writer, m_params);
    std::cout << std::endl;

    m_cin_reader->ReadWeight(*m_cin_reader, m_params.weight_threshold);
}

void Simulator::editLayers()
{
    std::cout << "Current layer specification: " << std::endl;
    m_cout_writer->WriteLayers(*m_cout_writer, m_params);
    std::cout << std::endl;

    if (promptEdit()) {
        m_cin_reader->ReadLayers(*m_cin_reader, m_params, m_params.layers);
        m_params.num_layers = m_params.layers.size() - 2;
    }
}

void Simulator::editTarget()
{
    std::cout << "Current value: " << std::endl;
    m_cout_writer->WriteEndCriteria(*m_cout_writer, m_params);
    std::cout << std::endl;

    m_cin_reader->ReadTarget(*m_cin_reader, m_params, m_params.target);
}

void Simulator::editSource()
{
    std::cout << "Current source type: " << std::endl;
    m_cout_writer->WriteSourceType(*m_cout_writer, m_params);
    std::cout << std::endl;

    std::cout << "Layer Specification: " << std::endl;
    m_cout_writer->WriteLayers(*m_cout_writer, m_params);
    std::cout << std::endl;

    std::cout << "Current starting position: " << std::endl;
    m_cout_writer->WritePhotonSource(*m_cout_writer, m_params);
    std::cout << std::endl;

    m_cin_reader->ReadSource(*m_cin_reader, m_params, m_params.source);
}

void Simulator::showEditMenuHelp()
{
    std::cout << "  O = Print the input on screen." << std::endl;
    std::cout << "  M = Change media list." << std::endl;
    std::cout << "  F = Change output file name." << std::endl;
    std::cout << "  G = Change grid properties." << std::endl;
    std::cout << "  C = Change scored data categories." << std::endl;
    std::cout << "  W = Change threshold weight." << std::endl;
    std::cout << "  L = Change layer specifications." << std::endl;
    std::cout << "  P = Change photon number and computation time limit." << std::endl;
    std::cout << "  S = Change source properties." << std::endl;
    std::cout << "  Q = Quit from change menu and start simulation." << std::endl;
    std::cout << "  * Commands here are not case-sensitive" << std::endl;
}

void Simulator::scaleReflectance(Radiance& radiance, ScaleMode mode)
{
    /*
    * a is the angle alpha.
    *
    * Scale Rd(r,a) and Td(r,a) by:
    *     area perpendicular to photon direction * solid angle * num_photons
    * or:
    *     [2*PI*r*grid_r*cos(a)] * [2*PI*sin(a)*grid_a] * [num_photons]
    * or:
    *     [2*PI*PI*grid_r*grid_a*r*sin(2a)] * [num_photons]
    *
    * Scale Rd(r) and Td(r) by area on the surface * num_photons.
    * Scale Rd(a) and Td(a) by solid angle * num_photons.
    */

    auto op = (mode == ScaleMode::Scale) ?
        [](double x, double y) { return x * y; } :
        [](double x, double y) { return x / y; };

    std::size_t nr = m_params.grid.num_r;
    std::size_t na = m_params.grid.num_a;
    std::size_t nt = m_params.grid.num_t;

    double dr = m_params.grid.step_r;
    double da = m_params.grid.step_a;
    double dt = m_params.grid.step_t;

    double scale1 = (double)m_params.target.photons_limit;


    if (mode == ScaleMode::Scale) {
        radiance.R_error = 1 / scale1 * sqrt(radiance.R_error - radiance.R_total * radiance.R_total / scale1);
        radiance.Rb_error = 1 / scale1 * sqrt(radiance.Rb_error - radiance.Rb_total * radiance.Rb_total / scale1);

        radiance.R_total /= scale1;
        radiance.Rb_total = radiance.Rb_total / scale1 + radiance.R_spec;
    }
    else {
        radiance.R_total *= scale1;
        radiance.Rb_total = (radiance.Rb_total - radiance.R_spec) * scale1;

        radiance.R_error = (scale1 * radiance.R_error) * (scale1 * radiance.R_error) + 1 / scale1 * radiance.R_total * radiance.R_total;
        radiance.Rb_error = (scale1 * radiance.Rb_error) * (scale1 * radiance.Rb_error) + 1 / scale1 * radiance.Rb_total * radiance.Rb_total;
    }

    scale1 = dt * m_params.target.photons_limit;
    if (m_params.record.R_t) {
        for (std::size_t it = 0; it < nt; it++) {
            radiance.R_t[it] = op(radiance.R_t[it], scale1);
        }
    }

    scale1 = 2.0 * std::numbers::pi * dr * dr * m_params.target.photons_limit;
    // area is 2*PI*[(ir+0.5)*grid_r]*grid_r.  ir + 0.5 to be added.

    if (m_params.record.R_r) {
        for (std::size_t ir = 0; ir < nr; ir++) {
            double scale2 = 1.0 / ((ir + 0.5) * scale1);
            radiance.R_r[ir] = op(radiance.R_r[ir], scale2);
        }
    }

    scale1 *= dt;
    if (m_params.record.R_rt) {
        for (std::size_t ir = 0; ir < nr; ir++) {
            double scale2 = 1.0 / ((ir + 0.5) * scale1);
            for (std::size_t it = 0; it < nt; it++) {
                radiance.R_rt[ir][it] = op(radiance.R_rt[ir][it], scale2);
            }
        }
    }

    scale1 = std::numbers::pi * da * m_params.target.photons_limit;

    // Solid angle times cos(a) is PI*sin(2a)*grid_a. sin(2a) to be added.
    if (m_params.record.R_a) {
        for (std::size_t ia = 0; ia < na; ia++) {
            double scale2 = 1.0 / (sin(2.0 * (ia + 0.5) * da) * scale1);
            radiance.R_a[ia] = op(radiance.R_a[ia], scale2);
        }
    }

    scale1 *= dt;
    if (m_params.record.R_at) {
        for (std::size_t ia = 0; ia < na; ia++) {
            for (std::size_t it = 0; it < nt; it++) {
                double scale2 = 1.0 / (sin(2.0 * (ia + 0.5) * da) * scale1);
                radiance.R_at[ia][it] = op(radiance.R_at[ia][it], scale2);
            }
        }
    }

    scale1 = 2.0 * std::numbers::pi * dr * dr * std::numbers::pi * da * m_params.target.photons_limit;
    if (m_params.record.R_ra) {
        for (std::size_t ir = 0; ir < nr; ir++) {
            for (std::size_t ia = 0; ia < na; ia++) {
                double scale2 = 1.0 / ((ir + 0.5) * sin(2.0 * (ia + 0.5) * da) * scale1);
                radiance.R_ra[ir][ia] = op(radiance.R_ra[ir][ia], scale2);
            }
        }
    }

    scale1 *= dt;
    if (m_params.record.R_rat) {
        for (std::size_t ir = 0; ir < nr; ir++) {
            for (std::size_t ia = 0; ia < na; ia++) {
                double scale2 = 1.0 / ((ir + 0.5) * sin(2.0 * (ia + 0.5) * da) * scale1);
                for (std::size_t it = 0; it < nt; it++) {
                    radiance.R_rat[ir][ia][it] = op(radiance.R_rat[ir][ia][it], scale2);
                }
            }
        }
    }
}

void Simulator::scaleTransmittance(Radiance& radiance, ScaleMode mode)
{
    /*
    * a is the angle alpha.
    *
    * Scale Rd(r,a) and Td(r,a) by:
    *     area perpendicular to photon direction * solid angle * num_photons
    * or:
    *     [2*PI*r*grid_r*cos(a)] * [2*PI*sin(a)*grid_a] * [num_photons]
    * or:
    *     [2*PI*PI*grid_r*grid_a*r*sin(2a)] * [num_photons]
    *
    * Scale Rd(r) and Td(r) by area on the surface * num_photons.
    * Scale Rd(a) and Td(a) by solid angle * num_photons.
    */

    auto op = (mode == ScaleMode::Scale) ?
        [](double x, double y) { return x * y; } :
        [](double x, double y) { return x / y; };

    std::size_t nr = m_params.grid.num_r;
    std::size_t na = m_params.grid.num_a;
    std::size_t nt = m_params.grid.num_t;

    double dr = m_params.grid.step_r;
    double da = m_params.grid.step_a;
    double dt = m_params.grid.step_t;

    double scale1 = (double)m_params.target.photons_limit;

    if (mode == ScaleMode::Scale) {
        radiance.T_error = 1 / scale1 * sqrt(radiance.T_error - radiance.T_total * radiance.T_total / scale1);
        radiance.Tb_error = 1 / scale1 * sqrt(radiance.Tb_error - radiance.Tb_total * radiance.Tb_total / scale1);

        radiance.T_total /= scale1;
        radiance.Tb_total /= scale1;
    }
    else {
        radiance.T_total *= scale1;
        radiance.Tb_total *= scale1;

        radiance.T_error = (scale1 * radiance.T_error) * (scale1 * radiance.T_error) + 1 / scale1 * radiance.T_total * radiance.T_total;
        radiance.Tb_error = (scale1 * radiance.Tb_error) * (scale1 * radiance.Tb_error) + 1 / scale1 * radiance.Tb_total * radiance.Tb_total;
    }

    scale1 = dt * m_params.target.photons_limit;

    if (m_params.record.T_t) {
        for (std::size_t it = 0; it < nt; it++) {
            radiance.T_t[it] = op(radiance.T_t[it], scale1);
        }
    }

    // area is 2*PI*[(ir+0.5)*grid_r]*grid_r. ir + 0.5 to be added.
    scale1 = 2.0 * std::numbers::pi * dr * dr * m_params.target.photons_limit;

    if (m_params.record.T_r) {
        for (std::size_t ir = 0; ir < nr; ir++) {
            double scale2 = 1.0 / ((ir + 0.5) * scale1);
            radiance.T_r[ir] = op(radiance.T_r[ir], scale2);
        }
    }

    scale1 *= dt;

    if (m_params.record.T_rt) {
        for (std::size_t ir = 0; ir < nr; ir++) {
            for (std::size_t it = 0; it < nt; it++) {
                double scale2 = 1.0 / ((ir + 0.5) * scale1);
                radiance.T_rt[ir][it] = op(radiance.T_rt[ir][it], scale2);
            }
        }
    }

    scale1 = std::numbers::pi * da * m_params.target.photons_limit;

    // Solid angle times cos(a) is PI*sin(2a)*grid_a. sin(2a) to be added.
    if (m_params.record.T_a) {
        for (std::size_t ia = 0; ia < na; ia++) {
            double scale2 = 1.0 / (sin(2.0 * (ia + 0.5) * da) * scale1);
            radiance.T_a[ia] = op(radiance.T_a[ia], scale2);
        }
    }

    scale1 *= dt;
    if (m_params.record.T_at) {
        for (std::size_t ia = 0; ia < na; ia++) {
            for (std::size_t it = 0; it < nt; it++) {
                double scale2 = 1.0 / (sin(2.0 * (ia + 0.5) * da) * scale1);
                radiance.T_at[ia][it] = op(radiance.T_at[ia][it], scale2);
            }
        }
    }

    scale1 = 2.0 * std::numbers::pi * dr * dr * std::numbers::pi * da * m_params.target.photons_limit;
    if (m_params.record.T_ra) {
        for (std::size_t ir = 0; ir < nr; ir++) {
            for (std::size_t ia = 0; ia < na; ia++) {
                double scale2 = 1.0 / ((ir + 0.5) * sin(2.0 * (ia + 0.5) * da) * scale1);
                radiance.T_ra[ir][ia] = op(radiance.T_ra[ir][ia], scale2);
            }
        }
    }

    scale1 *= dt;
    if (m_params.record.T_rat) {
        for (std::size_t ir = 0; ir < nr; ir++) {
            for (std::size_t ia = 0; ia < na; ia++) {
                for (std::size_t it = 0; it < nt; it++) {
                    double scale2 = 1.0 / ((ir + 0.5) * sin(2.0 * (ia + 0.5) * da) * scale1);
                    radiance.T_rat[ir][ia][it] = op(radiance.T_rat[ir][ia][it], scale2);
                }
            }
        }
    }
}


void Simulator::scaleAbsorption(Radiance& radiance, ScaleMode mode)
{
    auto op = (mode == ScaleMode::Scale) ?
        [](double x, double y) { return x / y; } :
        [](double x, double y) { return x * y; };

    std::size_t nz = m_params.grid.num_z;
    std::size_t nr = m_params.grid.num_r;
    std::size_t nt = m_params.grid.num_t;

    double dz = m_params.grid.step_z;
    double dr = m_params.grid.step_r;
    double dt = m_params.grid.step_t;

    double scale1 = (double)m_params.target.photons_limit;

    // scale A
    if (mode == ScaleMode::Scale) {
        radiance.A_error = 1 / scale1 * sqrt(radiance.A_error - radiance.A_total * radiance.A_total / scale1);
        radiance.A_total /= scale1;
    }
    // unscale A
    else {
        radiance.A_total *= scale1;
        radiance.A_error = (scale1 * radiance.A_error) * (scale1 * radiance.A_error) + 1 / scale1 * radiance.A_total * radiance.A_total;
    }

    double scale2 = scale1 * dt;
    if (m_params.record.A_t) {
        for (std::size_t it = 0; it < nt; it++) {
            radiance.A_t[it] = op(radiance.A_t[it], scale2);
        }
    }

    scale1 *= dz;
    if (m_params.record.A_z) {
        for (std::size_t iz = 0; iz < nz; iz++) {
            radiance.A_z[iz] = op(radiance.A_z[iz], scale1);
        }
    }

    scale2 = scale1 * dt;
    if (m_params.record.A_zt) {
        for (std::size_t iz = 0; iz < nz; iz++) {
            for (std::size_t it = 0; it < nt; it++) {
                radiance.A_zt[iz][it] = op(radiance.A_zt[iz][it], scale2);
            }
        }
    }

    if (m_params.record.A_rz) {
        for (std::size_t iz = 0; iz < nz; iz++) {
            radiance.Ab_z[iz] = op(radiance.Ab_z[iz], scale1);
        }
    }

    if (m_params.record.A_rzt) {
        for (std::size_t iz = 0; iz < nz; iz++) {
            for (std::size_t it = 0; it < nt; it++) {
                radiance.Ab_zt[iz][it] = op(radiance.Ab_zt[iz][it], scale2);
            }
        }
    }

    scale1 = 2.0 * std::numbers::pi * dr * dr * dz * m_params.target.photons_limit;
    if (m_params.record.A_rz) {
        for (std::size_t ir = 0; ir < nr; ir++) {
            for (std::size_t iz = 0; iz < nz; iz++) {
                radiance.A_rz[ir][iz] = op(radiance.A_rz[ir][iz], (ir + 0.5) * scale1);
            }
        }
    }

    scale2 = scale1 * dt;
    if (m_params.record.A_rzt) {
        for (std::size_t ir = 0; ir < nr; ir++) {
            for (std::size_t iz = 0; iz < nz; iz++) {
                for (std::size_t it = 0; it < nt; it++) {
                    radiance.A_rzt[ir][iz][it] = op(radiance.A_rzt[ir][iz][it], (ir + 0.5) * scale2);
                }
            }
        }
    }
}
