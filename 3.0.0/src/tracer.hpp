#pragma once

#include "mcml.hpp"

#include <vector>


template <typename T> using vec1 = std::vector<T>;
template <typename T> using vec2 = std::vector<std::vector<T>>;
template <typename T> using vec3 = std::vector<std::vector<std::vector<T>>>;


class Random;
struct RunParams;


class Tracer
{
public:
    Tracer(RunParams& params, std::shared_ptr<Random>& random);
    Tracer(RunParams& params, std::shared_ptr<Random>& random, Radiance radiance);
    ~Tracer() = default;

    /*  Choose a new direction for photon propagation by sampling:
            1. The polar (deflection) angle θ, measuring from downwards z-axis
            2. The azimuthal angle Ψ, measuring rotation around the z-axis in the xy-plane
    
        θ range:    0 - π  (0 to 180 degrees)    [sin(θ) range:  0 to 1]
        Ψ range:    0 - 2π (0 to 360 degrees)    [cos(Ψ) range: -1 to 1]
    
        Since cos²(θ) + sin²(θ) = 1, and if sin(θ) is known; cos(θ) = sqrt(1-sin(θ)²)
    
        In the Henyey-Greenstein phase function, g is the asymmetry parameter and typically 
        takes values in the range [-1, 1]:
            g = 0: isotropic scattering (uniform in all directions).
            g > 0: forward scattering (continue in the same direction).
            g < 0: backward scattering (scatter in the opposite direction). 
    */
    void Spin(Photon& photon, double g);

    /*  Initialize a photon packet.
    
        If an isotropic source is launched inside a glass layer, check whether the 
        photon will be total-internally reflected. If so, kill the photon to avoid
        infinite travelling inside the glass layer.
    */
    Photon Launch();

    //	Move the photon away in the current layer.
    void Hop(Photon& photon, double dist, double n);

    /*  Drop photon weight inside the tissue. The photon is assumed alive.

        The weight drop is dw = w * μa / (μa + μs).
        The dropped weight is assigned to the absorption array elements.
    */
    void Drop(Photon& photon);

    // Determine photon survival via roulette when photon weight becomes too small.
    void Roulette(Photon& photon);

    /*  Compute the Fresnel reflectance.
     
        Make sure that the cosine of the incident angle ai is positive, 
        and the case when the angle is greater than the critical angle is ruled out.
     
        cos_ai: cosine of the incident angle ai.
        cos_at: pointer to the cosine of the transmission angle at.
     
        Avoid trigonometric function operations as much as possible, because they are computationally intensive.
     
        eta_i:  incident refractive index
        eta_t:  transmit refractive index
        cos_ai: cosine of angle ai
    */
    double Fresnel(double eta_i, double eta_t, double cos_ai, double& cos_at);

    /*  Decide whether the photon will be transmitted or reflected on the upper boundary 
        (uz < 0) of the current layer.
     
        If current_layer is the first layer, the photon packet will be partially transmitted and 
        partially reflected if PARTIAL_REFLECTION active, or the photon packet will be either 
        transmitted or reflected determined statistically if PARTIAL_REFLECTION is inactive.
     
        Record the transmitted photon weight as reflection.
     
        If the current_layer is not the first layer and the photon packet is transmitted, 
        move the photon to the previous layer.
     
        Update photon parameters.
    */
    void CrossUp(Photon& photon);

    /*  Decide whether the photon will be transmitted or reflected on the bottom
        boundary (uz > 0) of the current layer.
    
        If the photon is transmitted, move the photon to current_layer + 1.
        If current_layer is the last layer, record the weight as transmittance.
    
        Update the photon parmameters.
    */
    void CrossDown(Photon& photon);

    /*  Set a step size if the previous step has finished.
     
        If the step size fits in the current layer, move the photon, drop weight, 
        and choose a new photon direction for propagation.
     
     	If the step size is long enough for the photon to hit an interface, this step
        is divided into three steps:
     
     	1. Move the photon to the boundary free of absorption or scattering.
        2. Update the step size to the unfinished step size.
        3. Decide whether the photon is reflected or transmitted.
    */
    void HopDropSpin(Photon& photon);

    // Trace a photon, then compute the absorption, transmittance, and reflection
    // constants, including their standard errors.
    void Trace(Photon& photon);


    // Record photon weight exiting the first layer (uz < 0) to the reflectance 
    // array and update its weight.
    void RecordReflectance(Photon& photon, double reflectance);

    // Record the photon weight exiting the last layer (uz > 0), 
    // no matter whether the layer is glass or not, to the transmittance array.
    void RecordTransmittance(Photon& photon, double reflectance);

    // Return radiance
    operator Radiance& () { return m_radiance; }

private:
    RunParams& m_params;

    std::shared_ptr<Random> m_random;

    Radiance m_radiance;
};
