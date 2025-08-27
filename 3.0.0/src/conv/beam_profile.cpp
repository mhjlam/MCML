/*******************************************************************************
 *	Beam profile implementation for CONV 3.0
 *  Copyright M.H.J. Lam, 2025.
 ****/

#include "beam_profile.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace conv {

BeamProfile::BeamProfile(const BeamParameters& params) {
    set_parameters(params);
}

void BeamProfile::set_parameters(const BeamParameters& params) {
    if (!params.is_valid()) {
        throw ConvException("Invalid beam parameters");
    }
    
    m_params = params;
    m_profile_loaded = false;
    
    // Load arbitrary profile if specified
    if (m_params.type == BeamType::Arbitrary && !m_params.profile_filename.empty()) {
        load_arbitrary_profile(m_params.profile_filename);
    } else if (m_params.type == BeamType::Arbitrary && !m_params.radii.empty()) {
        // Profile data provided directly
        if (!validate_arbitrary_data()) {
            throw ConvException("Invalid arbitrary beam profile data");
        }
        normalize_arbitrary_profile();
        m_profile_loaded = true;
    }
}

void BeamProfile::load_arbitrary_profile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw FileException("Cannot open beam profile file: " + filename);
    }
    
    m_params.radii.clear();
    m_params.power_densities.clear();
    
    std::string line;
    size_t line_number = 0;
    
    while (std::getline(file, line)) {
        ++line_number;
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        double radius, power_density;
        
        if (!(iss >> radius >> power_density)) {
            throw FileException("Invalid data format at line " + std::to_string(line_number) + 
                              " in file: " + filename);
        }
        
        if (radius < 0.0 || power_density < 0.0) {
            throw FileException("Negative values not allowed at line " + std::to_string(line_number) +
                              " in file: " + filename);
        }
        
        m_params.radii.push_back(radius);
        m_params.power_densities.push_back(power_density);
    }
    
    if (m_params.radii.empty()) {
        throw FileException("No valid data found in beam profile file: " + filename);
    }
    
    // Sort by radius if not already sorted
    std::vector<std::pair<double, double>> combined;
    for (size_t i = 0; i < m_params.radii.size(); ++i) {
        combined.emplace_back(m_params.radii[i], m_params.power_densities[i]);
    }
    
    std::sort(combined.begin(), combined.end());
    
    m_params.radii.clear();
    m_params.power_densities.clear();
    
    for (const auto& [radius, power_density] : combined) {
        m_params.radii.push_back(radius);
        m_params.power_densities.push_back(power_density);
    }
    
    if (!validate_arbitrary_data()) {
        throw FileException("Invalid beam profile data in file: " + filename);
    }
    
    normalize_arbitrary_profile();
    m_profile_loaded = true;
}

bool BeamProfile::is_valid() const noexcept {
    if (!m_params.is_valid()) {
        return false;
    }
    
    if (m_params.type == BeamType::Arbitrary) {
        return m_profile_loaded && validate_arbitrary_data();
    }
    
    return true;
}

double BeamProfile::effective_radius(double power_fraction) const {
    if (power_fraction <= 0.0 || power_fraction > 1.0) {
        throw std::invalid_argument("Power fraction must be between 0 and 1");
    }
    
    switch (m_params.type) {
        case BeamType::Original:
            return 0.0;
            
        case BeamType::Flat:
            return m_params.radius * std::sqrt(power_fraction);
            
        case BeamType::Gaussian: {
            // For Gaussian beam, solve: power_fraction = 1 - exp(-r²/(2σ²))
            const double sigma = m_params.radius / std::sqrt(2.0);
            return sigma * std::sqrt(-2.0 * std::log(1.0 - power_fraction));
        }
        
        case BeamType::Arbitrary: {
            if (!m_profile_loaded) return 0.0;
            
            // Numerical integration to find effective radius
            const double max_r = max_radius();
            const size_t n_steps = 1000;
            const double dr = max_r / n_steps;
            
            double total_power = 0.0;
            double cumulative_power = 0.0;
            
            // Calculate total power
            for (size_t i = 0; i < n_steps; ++i) {
                const double r = (i + 0.5) * dr;
                const double power_density = profile_value(r);
                total_power += power_density * 2.0 * std::numbers::pi * r * dr;
            }
            
            const double target_power = power_fraction * total_power;
            
            // Find radius where cumulative power reaches target
            for (size_t i = 0; i < n_steps; ++i) {
                const double r = (i + 0.5) * dr;
                const double power_density = profile_value(r);
                cumulative_power += power_density * 2.0 * std::numbers::pi * r * dr;
                
                if (cumulative_power >= target_power) {
                    return r;
                }
            }
            
            return max_r;
        }
        
        default:
            return 0.0;
    }
}

double BeamProfile::interpolate_arbitrary(double radius) const {
    if (m_params.radii.empty()) {
        return 0.0;
    }
    
    // Find interpolation points
    auto it = std::lower_bound(m_params.radii.begin(), m_params.radii.end(), radius);
    
    if (it == m_params.radii.begin()) {
        // Extrapolate from first point (assume zero at origin)
        if (radius <= 0.0) return 0.0;
        return m_params.power_densities[0] * radius / m_params.radii[0];
    }
    
    if (it == m_params.radii.end()) {
        // Beyond last point - assume zero
        return 0.0;
    }
    
    // Linear interpolation
    const size_t idx = std::distance(m_params.radii.begin(), it);
    const double r1 = m_params.radii[idx - 1];
    const double r2 = m_params.radii[idx];
    const double pd1 = m_params.power_densities[idx - 1];
    const double pd2 = m_params.power_densities[idx];
    
    const double t = (radius - r1) / (r2 - r1);
    return pd1 * (1.0 - t) + pd2 * t;
}

void BeamProfile::normalize_arbitrary_profile() {
    if (m_params.power_densities.empty()) {
        return;
    }
    
    // Numerical integration to find total power
    double total_power = 0.0;
    
    for (size_t i = 0; i < m_params.radii.size() - 1; ++i) {
        const double r1 = m_params.radii[i];
        const double r2 = m_params.radii[i + 1];
        const double pd1 = m_params.power_densities[i];
        const double pd2 = m_params.power_densities[i + 1];
        
        // Trapezoidal rule with cylindrical coordinates
        const double dr = r2 - r1;
        const double avg_power_density = 0.5 * (pd1 + pd2);
        const double avg_radius = 0.5 * (r1 + r2);
        
        total_power += avg_power_density * 2.0 * std::numbers::pi * avg_radius * dr;
    }
    
    if (total_power > 0.0) {
        const double normalization_factor = m_params.total_power / total_power;
        for (double& pd : m_params.power_densities) {
            pd *= normalization_factor;
        }
    }
}

bool BeamProfile::validate_arbitrary_data() const noexcept {
    if (m_params.radii.size() != m_params.power_densities.size()) {
        return false;
    }
    
    if (m_params.radii.size() < 2) {
        return false;
    }
    
    // Check that radii are sorted and non-negative
    for (size_t i = 0; i < m_params.radii.size(); ++i) {
        if (m_params.radii[i] < 0.0 || m_params.power_densities[i] < 0.0) {
            return false;
        }
        
        if (i > 0 && m_params.radii[i] <= m_params.radii[i - 1]) {
            return false; // Not strictly increasing
        }
    }
    
    // Check that first radius is zero or close to zero
    if (m_params.radii[0] > 1e-10) {
        return false;
    }
    
    return true;
}

} // namespace conv
