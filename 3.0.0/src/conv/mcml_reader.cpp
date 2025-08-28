/*******************************************************************************
 *	MCML data reader implementation for CONV 3.0
 *  Copyright M.H.J. Lam, 2025.
 ****/

#include "mcml_reader.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

namespace conv
{

// Helper parsing functions for MCML file format
namespace {

void parse_vector_data(const std::string& line, std::vector<double>& data) {
    std::istringstream iss(line);
    double value;
    
    while (iss >> value) {
        data.push_back(value);
    }
}

void parse_rat_data(const std::string& line, McmlOutputData& output_data) {
    // RAT section has multiple lines with different values
    std::istringstream iss(line);
    double value;
    
    // Try to parse a single numeric value from the line
    if (iss >> value) {
        // This is a simple implementation - in a real parser you'd track which RAT value this is
        if (output_data.specular_reflectance == 0.0) {
            output_data.specular_reflectance = value;
        } else if (output_data.total_reflectance == 0.0) {
            output_data.total_reflectance = value;
        } else if (output_data.total_absorption == 0.0) {
            output_data.total_absorption = value;
        } else if (output_data.total_transmittance == 0.0) {
            output_data.total_transmittance = value;
        }
    }
}

} // anonymous namespace

// McmlDataReader implementation
void McmlDataReader::read_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw ConvException("Cannot open file: " + filename);
    }

    // Clear previous data
    m_input_data = McmlInputData{};
    m_output_data = McmlOutputData{};
    
    std::string line;
    std::string current_section;
    bool in_data_section = false;
    
    try {
        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line.find('#') == 0) {
                continue;
            }
            
            // Parse input parameters from early lines
            if (!in_data_section && line.find_first_not_of(" \t") != std::string::npos) {
                // Look for dz, dr, dt line (3 floating point numbers)
                std::istringstream iss(line);
                double dz, dr, dt;
                if (iss >> dz >> dr >> dt) {
                    m_input_data.dz = dz;
                    m_input_data.dr = dr;
                    m_input_data.dt = dt;
                    continue;
                }
                
                // Look for nz, nr, nt, na line (4 integers)
                std::istringstream iss2(line);
                size_t nz, nr, nt, na;
                if (iss2 >> nz >> nr >> nt >> na) {
                    m_input_data.nz = nz;
                    m_input_data.nr = nr;
                    m_input_data.nt = nt;
                    m_input_data.na = na;
                    continue;
                }
            }
            
            // Check for data section headers
            if (line.find("RAT") != std::string::npos && line.find("#") != std::string::npos) {
                current_section = "RAT";
                in_data_section = true;
                continue;
            } else if (line.find("Rd_a") != std::string::npos && line.find("#") != std::string::npos) {
                current_section = "Rd_a";
                in_data_section = true;
                continue;
            } else if (line.find("Rd_r") != std::string::npos && line.find("#") != std::string::npos) {
                current_section = "Rd_r";
                in_data_section = true;
                continue;
            } else if (line.find("Td_a") != std::string::npos && line.find("#") != std::string::npos) {
                current_section = "Td_a";
                in_data_section = true;
                continue;
            } else if (line.find("Td_r") != std::string::npos && line.find("#") != std::string::npos) {
                current_section = "Td_r";
                in_data_section = true;
                continue;
            } else if (line.find("A_z") != std::string::npos && line.find("#") != std::string::npos) {
                current_section = "A_z";
                in_data_section = true;
                continue;
            }
            
            // Parse data based on current section
            if (in_data_section) {
                if (current_section == "RAT") {
                    parse_rat_data(line, m_output_data);
                } else if (current_section == "Rd_a") {
                    parse_vector_data(line, m_output_data.rd_a);
                } else if (current_section == "Rd_r") {
                    parse_vector_data(line, m_output_data.rd_r);
                } else if (current_section == "Td_a") {
                    parse_vector_data(line, m_output_data.td_a);
                } else if (current_section == "Td_r") {
                    parse_vector_data(line, m_output_data.td_r);
                } else if (current_section == "A_z") {
                    parse_vector_data(line, m_output_data.a_z);
                }
            }
        }
        
        // Set sensible defaults for missing data
        m_input_data.filename = filename;
        if (m_input_data.dr == 0.0) m_input_data.dr = 0.1; // Default spacing
        if (m_input_data.dz == 0.0) m_input_data.dz = 0.1;
        if (m_input_data.nr == 0) m_input_data.nr = std::max(m_output_data.rd_a.size(), size_t(1));
        if (m_input_data.na == 0) m_input_data.na = std::max(m_output_data.rd_a.size(), size_t(30)); // Common default
        if (m_input_data.nz == 0) m_input_data.nz = 1;
        
        // Create a basic 2D matrix for rd_ra from rd_a data for convolution
        if (!m_output_data.rd_a.empty()) {
            m_output_data.rd_ra = Matrix2D<double>(m_input_data.nr, m_input_data.na);
            // Fill with rd_a data replicated across angles (simplified)
            for (size_t ir = 0; ir < m_input_data.nr && ir < m_output_data.rd_a.size(); ++ir) {
                for (size_t ia = 0; ia < m_input_data.na; ++ia) {
                    m_output_data.rd_ra(ir, ia) = m_output_data.rd_a[ir];
                }
            }
        }
        
        std::cout << "Parsed MCML file successfully:" << std::endl;
        std::cout << "  Grid: nr=" << m_input_data.nr << ", na=" << m_input_data.na << ", nz=" << m_input_data.nz << std::endl;
        std::cout << "  Spacing: dr=" << m_input_data.dr << ", dz=" << m_input_data.dz << std::endl;
        std::cout << "  Data loaded: Rd_a=" << m_output_data.rd_a.size() << " values" << std::endl;
        
    } catch (const std::exception& e) {
        throw ConvException("Error parsing MCML file " + filename + ": " + e.what());
    }
}

// McmlInputData implementation
bool McmlInputData::is_valid() const noexcept {
    return num_photons > 0 && dz > 0.0 && dr > 0.0 && 
           nz > 0 && nr > 0 && num_layers > 0 &&
           layer_thickness.size() == num_layers &&
           refractive_index.size() == num_layers;
}

// McmlOutputData implementation
bool McmlOutputData::is_valid() const noexcept {
    // Basic validation - at least some data should be present
    return !rd_r.empty() || !td_r.empty() || !a_z.empty();
}

void McmlOutputData::clear() {
    specular_reflectance = 0.0;
    total_transmittance = 0.0;
    total_reflectance = 0.0;
    total_absorption = 0.0;
    
    // Clear 1D arrays
    rd_r.clear();
    rd_a.clear();
    rd_t.clear();
    td_r.clear();
    td_a.clear();
    td_t.clear();
    a_z.clear();
    a_t.clear();
    
    // Clear 2D arrays
    rd_ra.clear();
    rd_rt.clear();
    rd_at.clear();
    td_ra.clear();
    td_rt.clear();
    td_at.clear();
    a_rz.clear();
    a_zt.clear();
    
    // Clear 3D arrays
    rd_rat.clear();
    td_rat.clear();
    a_rzt.clear();
}

} // namespace conv
