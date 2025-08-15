#include "writer.hpp"

#include <chrono>
#include <format>
#include <string>
#include <fstream>

#include "mcml.hpp"
#include "random.hpp"


Writer::Writer(std::string filename) : m_output{ std::make_unique<std::ostream>(nullptr) }
{
    if (filename.empty()) {
        m_output = std::make_unique<std::ostream>(std::cout.rdbuf());
    }
    else {
        auto file_stream = std::make_unique<std::ofstream>(filename);

        if (!file_stream->is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        m_output = std::move(file_stream);
    }
}

Writer::~Writer()
{
    m_output.reset();
}


void Writer::WriteMediums(std::ostream& output, RunParams& params)
{
    std::string format;

    output << std::format("{:<24} {:>8} {:>8} {:>8} {:>8}", "# Medium name", "eta", "mu_a", "mu_s", "g") << std::endl;

    for (int i = 0; i < params.mediums.size(); i++) {
        Layer s = params.mediums[i];
        output << std::format("{:<4}{:<20} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}",
                              "", s.name, s.eta, s.mu_a, s.mu_s, s.g) << std::endl;
    }
    output << "end #of media\n";
}

void Writer::WriteFilename(std::ostream& output, RunParams& params)
{
    output << "# output file name\n";
    output << params.output_filename << std::endl;
}

void Writer::WriteGridParams(std::ostream& output, RunParams& params)
{
    output << std::format("{:<8} {:<8} {:<8}", "# step_z", "step_r", "step_t") << std::endl;
    output << std::format("{:<8.2f} {:<8.2f} {:<8.2f}", params.grid.step_z, params.grid.step_r, params.grid.step_t) << std::endl;
    output << std::endl;
}

void Writer::WriteGridSize(std::ostream& output, RunParams& params)
{
    output << std::format("{:<8} {:<8} {:<8} {:<8}", "# num_z", "num_r", "num_t", "num_a") << std::endl;
    output << std::format("{:<8} {:<8} {:<8} {:<8}", params.grid.num_z, params.grid.num_r, params.grid.num_t, params.grid.num_a) << std::endl;
    output << std::endl;
}

void Writer::WriteRecord(std::ostream& output, RunParams& params)
{
    output << "# scored quantities\n";

    std::vector<std::string> quantities;

    if (params.record.R_r) { quantities.push_back("R_r"); }
    if (params.record.R_a) { quantities.push_back("R_a"); }
    if (params.record.R_ra) { quantities.push_back("R_ra"); }
    if (params.record.R_t) { quantities.push_back("R_t"); }
    if (params.record.R_rt) { quantities.push_back("R_rt"); }
    if (params.record.R_at) { quantities.push_back("R_at"); }
    if (params.record.R_rat) { quantities.push_back("R_rat"); }

    if (params.record.T_r) { quantities.push_back("T_r"); }
    if (params.record.T_a) { quantities.push_back("T_a"); }
    if (params.record.T_ra) { quantities.push_back("T_ra"); }
    if (params.record.T_t) { quantities.push_back("T_t"); }
    if (params.record.T_rt) { quantities.push_back("T_rt"); }
    if (params.record.T_at) { quantities.push_back("T_at"); }
    if (params.record.T_rat) { quantities.push_back("T_rat"); }

    if (params.record.A_z) { quantities.push_back("A_z"); }
    if (params.record.A_rz) { quantities.push_back("A_rz"); }
    if (params.record.A_t) { quantities.push_back("A_t"); }
    if (params.record.A_zt) { quantities.push_back("A_zt"); }
    if (params.record.A_rzt) { quantities.push_back("A_rzt"); }

    std::string format;
    for (const auto& q : quantities) {
        format += std::format("{}{}", q, (&q == &quantities.back() ? "" : " "));
    }

    output << format << std::endl;
}

void Writer::WriteWeight(std::ostream& output, RunParams& params)
{
    output << "# weight threshold\n";
    output << std::format("{:.6f}", params.weight_threshold) << std::endl;
    output << std::endl;
}

void Writer::WriteRandomSeed(std::ostream& output, RunParams& params)
{
    output << "# random number generator seed\n";
    output << params.seed << std::endl;
    output << std::endl;
}

void Writer::WriteLayers(std::ostream& output, RunParams& params)
{
    std::string format;

    output << std::format("{:<24} {:<8}", "# Layer name", "thickness") << std::endl;

    for (int i = 0; i <= params.num_layers + 1; i++) {
        Layer s;

        s = params.layers[i];
        if (i != 0 && i != params.num_layers + 1) {
            output << std::format("{:<4}{:<20} {:<8.2f}", "", s.name, s.z1 - s.z0) << std::endl;
        }
        else {
            output << "\t" << s.name << std::endl;
        }
    }

    output << "end #of layers\n";
}

void Writer::WriteEndCriteria(std::ostream& output, RunParams& params)
{
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::seconds(params.target.time_limit));
    output << std::format("{:<12} time limit", "# photons") << std::endl;
    output << std::format("{:<12} {:%H:%M:%S}", params.target.photons_limit, duration) << std::endl;
    output << std::endl;
}

void Writer::WriteSourceType(std::ostream& output, RunParams& params)
{
    output << "# beam type\n";
    output << ((params.source.beam == BeamType::Pencil) ? "pencil" : "isotropic") << std::endl;
    output << std::endl;
}

void Writer::WritePhotonSource(std::ostream& output, RunParams& params)
{
    output << "# starting position of source\n";
    output << std::format("{:<8.2f} {:<20}", params.source.z, params.source.medium_name) << std::endl;
}

/*******************************************************************************
 *  Write input parameters to the output file.
 ****/
void Writer::WriteParams(std::ostream& output, RunParams& params)
{
    output << "# input file version\n";
    output << std::format("{:<50}", MCI_VERSION) << std::endl << std::endl;

    WriteMediums(output, params);

    output << "\n# Run parameters\n\n";

    WriteFilename(output, params);

    // Layers
    output << std::endl;
    WriteLayers(output, params);

    // Light source
    output << std::endl;
    WriteSourceType(output, params);
    WritePhotonSource(output, params);

    // Grids
    output << std::endl;
    WriteGridParams(output, params);
    WriteGridSize(output, params);

    // Scored data categories
    output << std::endl;
    WriteRecord(output, params);

    // Simulation control
    output << std::endl;
    WriteEndCriteria(output, params);
    WriteWeight(output, params);
    WriteRandomSeed(output, params);

    output << "end #of runs\n\n";
}

void Writer::WriteVersion(std::ostream& output, const std::string_view& version)
{
    output << "# output file version\n";
    output << version;
}

void Writer::WriteRandomizer(std::ostream& output, std::shared_ptr<Random> random)
{
    auto status = random->state();

    output << "# PRNG state:\n";

    for (int i = 0; i < status.size(); i++) {
        if (i % 5) {
            output << std::format("{:14d}", status[i]);
        }
        else {
            output << std::format("\n{:14d}", status[i]);
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteResults(std::ostream& output, RunParams& params, Radiance& radiance, std::shared_ptr<Random> random)
{
    WriteVersion(output, MCO_VERSION);
    WriteParams(output, params);
    WriteRandomizer(output, random);
    WriteRadiance(output, radiance);

    // Reflectance
    if (params.record.R_rat) { WriteR_rat(output, params.grid.num_r, params.grid.num_a, params.grid.num_t, radiance); }
    if (params.record.R_ra) { WriteR_ra(output, params.grid.num_r, params.grid.num_a, radiance); }
    if (params.record.R_rt) { WriteR_rt(output, params.grid.num_r, params.grid.num_t, radiance); }
    if (params.record.R_at) { WriteR_at(output, params.grid.num_a, params.grid.num_t, radiance); }
    if (params.record.R_r) { WriteR_r(output, params.grid.num_r, radiance); }
    if (params.record.R_a) { WriteR_a(output, params.grid.num_a, radiance); }
    if (params.record.R_t) { WriteR_t(output, params.grid.num_t, radiance); }

    // Transmittance
    if (params.record.T_rat) { WriteT_rat(output, params.grid.num_r, params.grid.num_a, params.grid.num_t, radiance); }
    if (params.record.T_ra) { WriteT_ra(output, params.grid.num_r, params.grid.num_a, radiance); }
    if (params.record.T_rt) { WriteT_rt(output, params.grid.num_r, params.grid.num_t, radiance); }
    if (params.record.T_at) { WriteT_at(output, params.grid.num_a, params.grid.num_t, radiance); }
    if (params.record.T_r) { WriteT_r(output, params.grid.num_r, radiance); }
    if (params.record.T_a) { WriteT_a(output, params.grid.num_a, radiance); }
    if (params.record.T_t) { WriteT_t(output, params.grid.num_t, radiance); }

    // Absorption
    if (params.record.A_rzt) { WriteA_rzt(output, params.grid.num_r, params.grid.num_z, params.grid.num_t, radiance); }
    if (params.record.A_rz) { WriteA_rz(output, params.grid.num_r, params.grid.num_z, radiance); }
    if (params.record.A_zt) { WriteA_zt(output, params.grid.num_z, params.grid.num_t, radiance); }
    if (params.record.A_z) { WriteA_z(output, params.grid.num_z, radiance); }
    if (params.record.A_t) { WriteA_t(output, params.grid.num_t, radiance); }

    // Close if file
    std::fstream& file = dynamic_cast<std::fstream&>(output);

    if (file && file.is_open()) {  // If cast succeeds, it's an fstream
        file.close();
    }

    std::cout << "Results written to file " << params.output_filename << ".\n";
}

void Writer::WriteRadiance(std::ostream& output, Radiance& radiance)
{
    double Rb_rel_error = (radiance.Rb_total) ? radiance.Rb_error / radiance.Rb_total * 100.0 : 0.0;
    double R_rel_error = (radiance.R_total) ? radiance.R_error / radiance.R_total * 100.0 : 0.0;
    double Tb_rel_error = (radiance.Tb_total) ? radiance.Tb_error / radiance.Tb_total * 100.0 : 0.0;
    double T_rel_error = (radiance.T_total) ? radiance.T_error / radiance.T_total * 100.0 : 0.0;
    double A_rel_error = (radiance.A_total) ? radiance.A_error / radiance.A_total * 100.0 : 0.0;

    output << "RAT # Reflectance, Absorption & Transmittance:\n\n";
    output << std::format("# {:<12} {:<18} {:<16}\n", "Average", "Standard Error", "Relative Error");

    output << std::format("{:<14.9f} {:<18} {:<16} {}\n", radiance.R_spec, "", "", "# Rs: Specular reflectance");
    output << std::format("{:<14.9f} {:<18.9f} {:<16.2f} {}\n", radiance.Rb_total, radiance.Rb_error, Rb_rel_error, "# Rb: Ballistic reflectance");
    output << std::format("{:<14.9f} {:<18.9f} {:<16.2f} {}\n", radiance.R_total, radiance.R_error, R_rel_error, "# Rd: Diffuse reflectance");
    output << std::format("{:<14.9f} {:<18.9f} {:<16.2f} {}\n", radiance.Tb_total, radiance.Tb_error, Tb_rel_error, "# Tb: Ballistic transmittance");
    output << std::format("{:<14.9f} {:<18.9f} {:<16.2f} {}\n", radiance.T_total, radiance.T_error, T_rel_error, "# Td: Diffuse transmittance");
    output << std::format("{:<14.9f} {:<18.9f} {:<16.2f} {}\n", radiance.A_total, radiance.A_error, A_rel_error, "# A:  Absorbed fraction");
    output << std::endl;
}

void Writer::WriteAb_zt(std::ostream& output, std::size_t Nz, std::size_t Nt, Radiance& radiance)
{
    output << "# Ab[z][t]. [1/(cm ps)]\n"
        "# Ab[0][0], [0][1],..[0][nt-1]\n"
        "# Ab[1][0], [1][1],..[1][nt-1]\n"
        "# ...\n"
        "# Ab[nz-1][0], [nz-1][1],..[nz-1][nt-1]\n"
        "Ab_zt";

    std::size_t i = 0;
    for (std::size_t iz = 0; iz < Nz; iz++) {
        for (std::size_t it = 0; it < Nt; it++) {
            output << std::format("{:{}.10f} ", radiance.A_zt[iz][it], 15);
            if (++i % 5 == 0) {
                output << std::endl;
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteA_rzt(std::ostream& output, std::size_t Nr, std::size_t Nz, std::size_t Nt, Radiance& radiance)
{
    WriteA_zt(output, Nz, Nt, radiance);

    output << "# A[r][z][t]. [1/(cm³ ps)]\n"
        "# A[0][0][0], [0][0][1],..[0][0][nt-1]\n"
        "# A[0][1][0], [0][1][1],..[0][1][nt-1]\n"
        "# ...\n"
        "# A[nr-1][nz-1][0], [nr-1][nz-1][1],..[nr-1][nz-1][nt-1]\n"
        "A_rzt\n";

    std::size_t i = 0;
    for (std::size_t ir = 0; ir < Nr; ir++) {
        for (std::size_t iz = 0; iz < Nz; iz++) {
            for (std::size_t it = 0; it < Nt; it++) {
                output << std::format("{:{}.10f} ", radiance.A_rzt[ir][iz][it], 15);
                if (++i % 5 == 0) {
                    output << std::endl;
                }
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteAb_z(std::ostream& output, std::size_t Nz, Radiance& radiance)
{
    output << "Ab_z # Ab[0], [1],..Ab[nz-1]. [1/cm]\n";

    for (std::size_t iz = 0; iz < Nz; iz++) {
        output << std::format("{:{}.10f}\n", radiance.Ab_z[iz], 15);
    }

    output << std::endl;
}

void Writer::WriteA_rz(std::ostream& output, std::size_t Nr, std::size_t Nz, Radiance& radiance)
{
    WriteA_z(output, Nz, radiance);

    output << "# A[r][z]. [1/cm³]\n"
        "# A[0][0], [0][1],..[0][nz-1]\n"
        "# ...\n"
        "# A[nr-1][0], [nr-1][1],..[nr-1][nz-1]\n"
        "A_rz\n";

    std::size_t i = 0;
    for (std::size_t ir = 0; ir < Nr; ir++) {
        for (std::size_t iz = 0; iz < Nz; iz++) {
            output << std::format("{:{}.10f} ", radiance.A_rz[ir][iz], 15);
            if (++i % 5 == 0) {
                output << std::endl;
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteA_zt(std::ostream& output, std::size_t Nz, std::size_t Nt, Radiance& radiance)
{
    output << "# A[z][t]. [1/(cm ps)]\n"
        "# A[0][0], [0][1],..[0][nt-1]\n"
        "# A[1][0], [1][1],..[1][nt-1]\n"
        "# ...\n"
        "# A[nz-1][0], [nz-1][1],..[nz-1][nt-1]\n"
        "A_zt\n";

    std::size_t i = 0;
    for (std::size_t iz = 0; iz < Nz; iz++) {
        for (std::size_t it = 0; it < Nt; it++) {
            output << std::format("{:{}.10f} ", radiance.A_zt[iz][it], 15);
            if (++i % 5 == 0) {
                output << std::endl;
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteA_z(std::ostream& output, std::size_t Nz, Radiance& radiance)
{
    output << "A_z # A[0], [1],..A[nz-1]. [1/cm]\n";

    for (std::size_t iz = 0; iz < Nz; iz++) {
        output << std::format("{:{}.10f}\n", radiance.A_z[iz], 15);
    }

    output << std::endl;
}

void Writer::WriteA_t(std::ostream& output, std::size_t Nt, Radiance& radiance)
{
    output << "A_t # A[0], [1],..A[nt-1]. [1/ps]\n";

    for (std::size_t it = 0; it < Nt; it++) {
        output << std::format("{:{}.10f}\n", radiance.A_t[it], 15);
    }

    output << std::endl;
}

void Writer::WriteR_rat(std::ostream& output, std::size_t Nr, std::size_t Na, std::size_t Nt, Radiance& radiance)
{
    output << "# Rd[r][a][t]. [1/(cm² sr ps)]\n"
        "# Rd[0][0][0], [0][0][1],..[0][0][nt-1]\n"
        "# Rd[0][1][0], [0][1][1],..[0][1][nt-1]\n"
        "# ...\n"
        "# Rd[nr-1][na-1][0], [nr-1][na-1][1],..[nr-1][na-1][nt-1]\n"
        "R_rat\n";

    std::size_t i = 0;
    for (std::size_t ir = 0; ir < Nr; ir++) {
        for (std::size_t ia = 0; ia < Na; ia++) {
            for (std::size_t it = 0; it < Nt; it++) {
                output << std::format("{:{}.10f} ", radiance.R_rat[ir][ia][it], 15);
                if (++i % 5 == 0) {
                    output << std::endl;
                }
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteR_ra(std::ostream& output, std::size_t Nr, std::size_t Na, Radiance& radiance)
{
    output << "# Rd[r][angle]. [1/(cm² sr)].\n"
        "# Rd[0][0], [0][1],..[0][na-1]\n"
        "# Rd[1][0], [1][1],..[1][na-1]\n"
        "# ...\n"
        "# Rd[nr-1][0], [nr-1][1],..[nr-1][na-1]\n"
        "R_ra\n";

    for (std::size_t ir = 0; ir < Nr; ir++) {
        for (std::size_t ia = 0; ia < Na; ia++) {
            output << std::format("{:{}.10f} ", radiance.R_ra[ir][ia], 15);
            if ((ir * Na + ia + 1) % 5 == 0) {
                output << std::format("\n");
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteR_rt(std::ostream& output, std::size_t Nr, std::size_t Nt, Radiance& radiance)
{
    output << "# Rd[r][t]. [1/(cm² ps)]\n"
        "# Rd[0][0], [0][1],..[0][nt-1]\n"
        "# Rd[0][0], [0][1],..[0][nt-1]\n"
        "# ...\n"
        "# Rd[nr-1][0], [nr-1][1],..[nr-1][nt-1]\n"
        "R_rt\n";

    std::size_t i = 0;
    for (std::size_t ir = 0; ir < Nr; ir++) {
        for (std::size_t it = 0; it < Nt; it++) {
            output << std::format("{:{}.10f} ", radiance.R_rt[ir][it], 15);
            if (++i % 5 == 0) {
                output << std::format("\n");
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteR_at(std::ostream& output, std::size_t Na, std::size_t Nt, Radiance& radiance)
{
    output << "# Rd[a][t]. [1/(sr ps)]\n"
        "# Rd[0][0], [0][1],..[0][nt-1]\n"
        "# Rd[1][0], [1][1],..[1][nt-1]\n"
        "# ...\n"
        "# Rd[na-1][0], [na-1][1],..[na-1][nt-1]\n"
        "R_at\n";

    std::size_t i = 0;
    for (std::size_t ia = 0; ia < Na; ia++) {
        for (std::size_t it = 0; it < Nt; it++) {
            output << std::format("{:{}.10f} ", radiance.R_at[ia][it], 15);
            if (++i % 5 == 0) {
                output << std::format("\n");
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteR_r(std::ostream& output, std::size_t Nr, Radiance& radiance)
{
    output << "R_r # Rd[0], [1],..Rd[nr-1]. [1/cm²]\n";

    for (std::size_t ir = 0; ir < Nr; ir++) {
        //output << std::format("{:{}.10f}\n", radiance.R_r[ir], 15);
        output << std::format("{:{}.10f}\n", radiance.R_r[ir], 15);
    }

    output << std::endl;
}

void Writer::WriteR_a(std::ostream& output, std::size_t Na, Radiance& radiance)
{
    output << "Rd_a # Rd[0], [1],..Rd[na-1]. [1/sr]\n";

    for (std::size_t ia = 0; ia < Na; ia++) {
        output << std::format("{:{}.10f}\n", radiance.R_a[ia], 15);
    }

    output << std::endl;
}

void Writer::WriteR_t(std::ostream& output, std::size_t Nt, Radiance& radiance)
{
    output << "R_t # Rd[0], [1],..Rd[nt-1]. [1/ps]\n";

    for (std::size_t it = 0; it < Nt; it++) {
        output << std::format("{:{}.10f}\n", radiance.R_t[it], 15);
    }

    output << std::endl;
}

void Writer::WriteT_rat(std::ostream& output, std::size_t Nr, std::size_t Na, std::size_t Nt, Radiance& radiance)
{
    output << "# Td[r][a][t]. [1/(cm² sr ps)]\n"
        "# Td[0][0][0], [0][0][1],..[0][0][nt-1]\n"
        "# Td[0][1][0], [0][1][1],..[0][1][nt-1]\n"
        "# ...\n"
        "# Td[nr-1][na-1][0], [nr-1][na-1][1],..[nr-1][na-1][nt-1]\n"
        "T_rat\n";

    std::size_t i = 0;
    for (std::size_t ir = 0; ir < Nr; ir++) {
        for (std::size_t ia = 0; ia < Na; ia++) {
            for (std::size_t it = 0; it < Nt; it++) {
                output << std::format("{:{}.10f} ", radiance.T_rat[ir][ia][it], 15);
                if (++i % 5 == 0) {
                    output << std::format("\n");
                }
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteT_ra(std::ostream& output, std::size_t Nr, std::size_t Na, Radiance& radiance)
{
    output << "# Td[r][angle]. [1/(cm² sr)].\n",
        "# Td[0][0], [0][1],..[0][na-1]\n",
        "# Td[1][0], [1][1],..[1][na-1]\n",
        "# ...\n",
        "# Td[nr-1][0], [nr-1][1],..[nr-1][na-1]\n",
        "T_ra\n";

    for (std::size_t ir = 0; ir < Nr; ir++) {
        for (std::size_t ia = 0; ia < Na; ia++) {
            output << std::format("{:{}.10f} ", radiance.T_ra[ir][ia], 15);
            if ((ir * Na + ia + 1) % 5 == 0) {
                output << std::format("\n");
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteT_rt(std::ostream& output, std::size_t Nr, std::size_t Nt, Radiance& radiance)
{
    output << "# Td[r][t]. [1/(cm² ps)]\n"
        "# Td[0][0], [0][1],..[0][nt-1]\n"
        "# Td[0][0], [0][1],..[0][nt-1]\n"
        "# ...\n"
        "# Td[nr-1][0], [nr-1][1],..[nr-1][nt-1]\n"
        "T_rt\n";

    std::size_t i = 0;
    for (std::size_t ir = 0; ir < Nr; ir++) {
        for (std::size_t it = 0; it < Nt; it++) {
            output << std::format("{:{}.10f} ", radiance.T_rt[ir][it], 15);
            if (++i % 5 == 0) {
                output << std::format("\n");
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteT_at(std::ostream& output, std::size_t Na, std::size_t Nt, Radiance& radiance)
{
    output << "# Td[a][t]. [1/(sr ps)]\n"
        "# Td[0][0], [0][1],..[0][nt-1]\n"
        "# Td[1][0], [1][1],..[1][nt-1]\n"
        "# ...\n"
        "# Td[na-1][0], [na-1][1],..[na-1][nt-1]\n"
        "T_at\n";

    std::size_t i = 0;
    for (std::size_t ia = 0; ia < Na; ia++) {
        for (std::size_t it = 0; it < Nt; it++) {
            output << std::format("{:{}.10f} ", radiance.T_at[ia][it], 15);
            if (++i % 5 == 0) {
                output << std::format("\n");
            }
        }
    }

    output << std::endl << std::endl;
}

void Writer::WriteT_r(std::ostream& output, std::size_t Nr, Radiance& radiance)
{
    output << "T_r # Td[0], [1],..Td[nr-1]. [1/cm²]\n";

    for (std::size_t ir = 0; ir < Nr; ir++) {
        output << std::format("{:{}.10f}\n", radiance.T_r[ir], 15);
    }

    output << std::endl;
}

void Writer::WriteT_a(std::ostream& output, std::size_t Na, Radiance& radiance)
{
    output << "T_a # Td[0], [1],..Td[na-1]. [1/sr]\n";

    for (std::size_t ia = 0; ia < Na; ia++) {
        output << std::format("{:{}.10f}\n", radiance.T_a[ia], 15);
    }

    output << std::endl;
}

void Writer::WriteT_t(std::ostream& output, std::size_t Nt, Radiance& radiance)
{
    output << "T_t # Rd[0], [1],..Td[nt-1]. [1/ps]\n";

    for (std::size_t it = 0; it < Nt; it++) {
        output << std::format("{:{}.10f}\n", radiance.T_t[it], 15);
    }

    output << std::endl;
}
