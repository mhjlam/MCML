/*******************************************************************************
 *	Basic implementation stubs for CONV 3.0 classes
 *  Copyright M.H.J. Lam, 2025.
 ****/

#include "conv.hpp"

namespace conv {

// BeamParameters implementation
bool BeamParameters::is_valid() const noexcept {
    if (total_power <= 0.0) return false;
    
    switch (type) {
        case BeamType::Original:
            return true;
        case BeamType::Flat:
        case BeamType::Gaussian:
            return radius >= 0.0;
        case BeamType::Arbitrary:
            return radii.size() == power_densities.size() && 
                   radii.size() >= 2;
        default:
            return false;
    }
}

// GridParameters implementation
bool GridParameters::is_valid() const noexcept {
    return nr > 0 && dr > 0.0 && nrc > 0 && drc > 0.0 && nxc > 0 && dxc > 0.0;
}

// ConvolutionConfig implementation
bool ConvolutionConfig::is_valid() const noexcept {
    return beam.is_valid() && grid.is_valid() && epsilon > 0.0 && epsilon < 1.0;
}

} // namespace conv
