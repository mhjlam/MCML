/*******************************************************************************
 *	Copyright Univ. of Texas M.D. Anderson Cancer Center, 1992-1996.
 *  Copyright M.H.J. Lam, 2025.
 *	Monte Carlo simulation of light transport in multi-layered turbid mediums in C++20.
 ****
 *	Version 1.x:    10/1991.
 *	Version 2.0:    02/1996.
 *  Version 3.0:    03/2025.
 *
 *	@authors
 *	Lihong Wang, Ph.D.
 *	Bioengineering Program
 *	Texas A&M University
 *	College Station, Texas
 *
 *	Steven L. Jacques, Ph.D.
 *	M.D. Anderson Cancer Center
 *	University of Texas
 *	Houston, Texas
 *
 *	Liqiong Zheng, B.S.
 *	Department of Computer Science
 *	University of Houston
 *	Houston, Texas
 *
 *  M.H.J. Lam, MSc.
 *  Graduate School of Natural Sciences
 *  Utrecht University
 *  Utrecht, Netherlands
 *
 *	@brief Monte Carlo Multi-Layer (MCML) simulation
 *
 *	This program was based on:
 *	(1) The Pascal code written by Marleen Keijzer and Steven L. Jacques in this
 *  laboratory in 1989, which deals with multi-layered turbid mediums.
 *
 *	(2) Algorithm for semi-infinite turbid name by S.A. Prahl, M. Keijzer,
 *  S.L. Jacques, A.J. Welch, SPIE Institute Series Vol. IS 5 (1989), and by
 *  A.N. Witt, The Astrophysical Journal Supplement Series 35, 1-6 (1977).
 *
 *	@section changelog Major Modifications
 *	
 *	Major modifications in version 1.x include:
 *		- Conform to ANSI Standard C.
 *		- Removal of limit on number of array elements, because arrays in this
 *        program are dynamically allocated.
 *        This means that the program can accept any number of layers or grid
 *        lines as long as the memory permits.
 *		- Avoiding global variables whenever possible.
 *		- Grouping variables logically using structures.
 *		- Top-down design, keep each subroutine clear & short.
 *		- Reflectance and transmittance are angularly resolved.
 *	
 *	Major modifications in version 2.0 include:
 *		- Allow interactive input of parameters.
 *		- Allow to score various simulation quantities.
 *		- Time-resolved simulation.
 *		- Adjustable source position.
 *		- Support Isotropic source.
 *		- Simulation time control in addition to photon control.
 *		- Compute the standard errors of some physical quantities.
 *		- Allow continuation simulations to reduce standard errors.
 *	
 *	Major modifications in version 3.0 include:
 *      - Conform to C++20 and modern C++ coding standards.
 *      - Using standard library containers and algorithms wherever applicable.
 *      - Using standard library random number generation facilities
 *        (Mersenne Twister instead of Delayed Fibonacci generator).
 *      - Reduce usage of pointers and raw arrays.
 *      - Object-oriented design with SOLID principles.
 *      - Template programming with concepts for type safety.
 *      - Observer pattern for progress reporting.
 *      - Consistent error handling with Result<T> type.
 *      - JSON input/output files support.
 *      - Enhanced documentation with Doxygen comments.
 *      - Improved build system with cross-platform support.
 ****
 *	@section units Physical Units
 *	Dimension of depth: centimeters (cm).
 *  Dimension of angle: steradians (sr).
 *	Dimension of time: picoseconds (ps).
 *
 *  Ballistic quantities relate to photons that travel along a straight path
 *  without scattering.
 *
 ****/

#pragma once

#include "random.hpp"
#include "exception.hpp"
#include "result.hpp"

#include <limits>
#include <random>
#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <concepts>
#include <type_traits>


template <typename T> using vec1 = std::vector<T>;
template <typename T> using vec2 = std::vector<std::vector<T>>;
template <typename T> using vec3 = std::vector<std::vector<std::vector<T>>>;

using unique_str = std::unordered_set<std::string>;

constexpr std::streamsize max_size = std::numeric_limits<std::streamsize>::max();

/**
 * @brief C++20 concepts for type safety and better error messages
 */
namespace mcml::concepts {
    /**
     * @brief Concept for optical coefficients (non-negative values)
     */
    template<typename T>
    concept OpticalCoefficient = std::floating_point<T> && requires(T t) {
        { t >= T{0} } -> std::convertible_to<bool>;
    };
    
    /**
     * @brief Concept for refractive index (positive values)
     */
    template<typename T>
    concept RefractiveIndex = std::floating_point<T> && requires(T t) {
        { t > T{0} } -> std::convertible_to<bool>;
    };
    
    /**
     * @brief Concept for anisotropy factor g (-1 <= g <= 1)
     */
    template<typename T>
    concept AnisotropyFactor = std::floating_point<T> && requires(T t) {
        { t >= T{-1} && t <= T{1} } -> std::convertible_to<bool>;
    };
    
    /**
     * @brief Concept for probability values (0 <= p <= 1)
     */
    template<typename T>
    concept Probability = std::floating_point<T> && requires(T t) {
        { t >= T{0} && t <= T{1} } -> std::convertible_to<bool>;
    };
    
    /**
     * @brief Concept for positive integers
     */
    template<typename T>
    concept PositiveInteger = std::integral<T> && requires(T t) {
        { t > T{0} } -> std::convertible_to<bool>;
    };
}


/**
 * @brief Physical and simulation constants
 */

/// Roulette survival chance when photon weight becomes too small
constexpr double ROULETTE_SURVIVAL = 0.1;

/// Speed of light in vacuum [cm/ps]
constexpr double SPEED_OF_LIGHT = 0.0299792458;

/// Inverse speed of light (1/C) [ps/cm]
constexpr double SPEED_OF_LIGHT_INV = 33.35640952;

/// Split photon if true, otherwise statistical reflection
constexpr bool PARTIAL_REFLECTION = false;

/// Tolerance for cos(theta) near 0 or PI
/// If 1 - cos(theta) <= COS_0_TOLERANCE, abs(theta) <= 1e-6 rad
/// If 1 + cos(theta) <= COS_0_TOLERANCE, abs(PI - theta) <= 1e-6 rad
constexpr double COS_0_TOLERANCE = 1.0E-12;

/// Tolerance for cos(theta) near PI/2
/// If cos(theta) <= COS_90_TOLERANCE, theta >= PI/2 - 1e-6 rad
constexpr double COS_90_TOLERANCE = 1.0E-6;

/// Input file format version identifier
constexpr std::string_view MCI_VERSION = "mcmli_3.0";

/// Output file format version identifier  
constexpr std::string_view MCO_VERSION = "mcmlo_3.0";


/**
 * @brief Enumeration types for simulation configuration
 */

/**
 * @brief Simulation run type
 */
enum class RunType {
    StartNew,                           ///< Start a new simulation
    Continue                            ///< Continue previous simulation
};

/**
 * @brief Light beam type for photon source
 */
enum class BeamType {
    Pencil,                             ///< Pencil beam (collimated)
    Isotropic                           ///< Isotropic source (uniform in all directions)
};

/**
 * @brief Control method for simulation termination
 */
enum class ControlBit {
    NumPhotons,                         ///< Terminate by photon number only
    TimeLimit,                          ///< Terminate by time limit only
    Both                                ///< Terminate by both photon number and time limit
};

/**
 * @brief File format for input/output
 */
enum class FileFormat {
    Ascii                              ///< ASCII text format (only supported format)
};

/**
 * @brief I/O mode specification
 */
enum class IoMode {
    Read,                               ///< Read mode for input
    Write                               ///< Write mode for output
};

/**
 * @brief Scaling mode for results
 */
enum class ScaleMode {
    Scale,                              ///< Scale results by photon count
    Unscale                             ///< Unscale results (raw values)
};

/**
 * @brief Timer operation mode
 */
enum class PunchMode {
    ResetTimer,                         ///< Reset timer to current time
    TimeElapsed,                        ///< Return elapsed time since last reset
    TimeElapsedStr,                     ///< Return time elapsed as formatted string
};


/********************************* Structures *********************************/

struct Vector3
{
    double x{}, y{}, z{};               // Cartesian coordinates
};

/*******************************************************************************
 *  Photon packet.
 ****/
struct Photon
{
    bool        alive{};		        //  Photon alive/terminated
    long        num_scatters{};	        //  Number of scatterings
    std::size_t current_layer{};	    //  Index to layer where photon currently is

    Vector3     position;		        //  Cartesian coordinates [cm]
    Vector3     direction;		        //  Directional cosines of a photon

    double      weight{};		        //  Weight
    double      step_size{};		    //  Current step size [cm]
    double      step_size_left{};	    //  Step size left. dimensionless [-]
    double      flight_time{};	        //  Flight time [picosec]

    double      R_i{};	                // Photon diffuse reflectance
    double      Rb_i{};	                // Photon ballistic reflectance
    double      T_i{};	                // Photon diffuse transmittance
    double      Tb_i{};	                // Photon ballistic transmittance
    double      A_i{};		            // Photon absorption
};

/*******************************************************************************
 *	Structure used to describe the geometry and optical properties of a layer.
 *	Top and bottom are the z coordinates for the upper boundary and lower
 *  boundary respectively.
 *
 *	cos_crit0 and cos_crit1 are the cosines of the critical angle of total
 *  internal reflection for the upper boundary and lower boundary respectively.
 *
 *	They are set to zero if no total internal reflection exists.
 *	They are used for computation speed.
 ****/
struct Layer
{
    std::size_t index{};                // Index of the layer
    std::string name{};                 // Name of the layer

    double      eta{};	                // Refractive index
    double      mu_a{};	                // Absorption coefficient [1/cm]
    double      mu_s{};	                // Scattering coefficient [1/cm]
    double      g{};	                // Henyey-Greenstein asymmetry factor

    double      z0{};                   // Top z coordinate [cm]
    double      z1{};                   // Bottom z coordinate [cm]

    double      cos_theta_c0{};         // Cosine of the top critical angle
    double      cos_theta_c1{};         // Cosine of the bottom critical angle
};

struct LightSource
{
    double      z{};		            // Z coordinate of light source
    BeamType    beam{};	                // Beam type

    std::size_t layer_index{};		    // Put source in this layer
    std::string medium_name{};          // Medium name of source layer
};

struct Grid
{
    double step_z{};		            // Z grid line separation [cm]
    double step_r{};		            // R grid line separation [cm]
    double step_a{};		            // Alpha grid line separation [rad]
    double step_t{};		            // Time grid line separation [ps]

    std::size_t num_z{};		        // Number of Z grid lines
    std::size_t num_r{};		        // Number of R grid lines
    std::size_t num_t{};		        // Number of time grid lines
    std::size_t num_a{};		        // Number of alpha grid lines

    double max_z{};		                // Maximum z [cm]
    double max_r{};		                // Maximum r [cm]
    double max_a{};		            // Maximum alpha [rad]
    double max_t{};		            // Maximum time [ps]
};

struct Target
{
    ControlBit  control_bit{};          // Control of simulation termination.
    
    std::size_t photons_limit{};	    // Number of photons to be traced
    long        time_limit{};	        // Computation time limit [sec]

    std::size_t photons_remaining{};    // Remaining photons to be traced
    long        time_remaining{};       // Computation time remaining
};

struct Record
{
    // Specify which quantity / quantities should be scored.
    // rat: Reflected photon density per unit area, per unit solid angle, per unit time
    // r:   radial distance from the light source
    // z:   depth into the medium (z = 0 is the surface)
    // a:   azimuthal angle
    // t:   time-dependent, reflectance over time

    bool R_rat{ false };                // Diffuse reflectance [1/(cm� sr ps)]
    bool R_ra{ false };                 // Diffuse reflectance [1/(cm� sr)]
    bool R_rt{ false };                 // Diffuse reflectance [1/sr ps]
    bool R_at{ false };                 // Diffuse reflectance [1/(cm� ps)]
    bool R_r{ false };                  // Diffuse reflectance [1/cm�]
    bool R_a{ false };                  // Diffuse reflectance [1/sr]
    bool R_t{ false };                  // Diffuse reflectance [1/ps]

    bool T_rat{ false };                // Diffuse transmittance [1/(cm� sr ps)]
    bool T_ra{ false };                 // Diffuse transmittance [1/(cm� sr)]
    bool T_rt{ false };                 // Diffuse transmittance [1/sr ps]
    bool T_at{ false };                 // Diffuse transmittance [1/(cm� ps)]
    bool T_r{ false };                  // Diffuse transmittance [1/cm�]
    bool T_a{ false };                  // Diffuse transmittance [1/sr]
    bool T_t{ false };                  // Diffuse transmittance [1/ps]

    bool A_rzt{ false };                // Absorption [1/(cm� sr ps)]
    bool A_rz{ false };                 // Absorption [1/cm�]
    bool A_zt{ false };                 // Absorption [1/(cm� ps)]
    bool A_z{ false };                  // Absorption [1/cm�]
    bool A_t{ false };                  // Absorption [1/ps]
};

/*******************************************************************************
 *	Input parameters for each independent run.
 *
 *	z and r are for the cylindrical coordinate system. [cm]
 *	alpha denotes the angle between the photon exiting direction and the surface
 *  normal. [radian]
 *
 *	Member layers will point to an array of structures which store parameters of
 *  each layer. This array has (number_layers + 2) elements; one for each layer.
 *	The first and last layers are the top and bottom ambient layers respectively.
 ****/
struct RunParams
{
    std::string output_filename{};      // Output file name
    unique_str  unique_output_filenames;// Unique output file names

    std::size_t num_runs{};		        // Number of runs
    std::size_t num_layers{};           // Number of intermediate layers

    long        seed{};		            // Random number seed (unused)
    double      weight_threshold{};		// Play roulette if photon weight is less than this threshold

    Grid        grid;                   // Grid parameters
    Target      target;                 // Target parameters
    LightSource source;                 // Light source parameters
    vec1<Layer> layers;                 // Layer parameters
    vec1<Layer> mediums;                // Medium parameters

    Record      record;                 // Quantities to be scored
};

/*******************************************************************************
 *	Structures for scoring physical quantities.
 ****/
struct Radiance
{
    /* Reflectance */

    vec3<double> R_rat; // Diffuse reflectance per unit area, per unit solid angle, per unit time [1/(cm� sr ps)]
    vec2<double> R_ra;  // Diffuse reflectance per unit area, per unit solid angle [1/(cm� sr)]
    vec2<double> R_rt;  // Diffuse reflectance per unit solid angle, per unit time [1/sr ps]
    vec2<double> R_at;  // Diffuse reflectance per unit area, per unit time [1/cm� ps]
    vec1<double> R_r;   // Diffuse reflectance per unit area [1/cm�]
    vec1<double> R_a;   // Diffuse reflectance per unit solid angle [1/sr]
    vec1<double> R_t;   // Diffuse reflectance per unit time [1/ps]

    double R_total{};	// Total diffuse reflectance
    double R_spec{};	// Specular reflectance
    double R_error{};	// Standard error of diffuse reflectance

    double Rb_total{};	// Total ballistic reflectance
    double Rb_error{};	// Standard error for ballistic reflectance

    /* Transmittance */

    vec3<double> T_rat; // Diffuse transmittance per unit area, per unit solid angle, per unit time [1/(cm� sr ps)]
    vec2<double> T_ra;  // Diffuse transmittance per unit area, per unit solid angle [1/(cm� sr)]
    vec2<double> T_rt;  // Diffuse transmittance per unit solid angle, per unit time [1/sr ps]
    vec2<double> T_at;  // Diffuse transmittance per unit area, per unit time [1/cm� ps]
    vec1<double> T_r;	// Diffuse transmittance per unit area [1/cm�]
    vec1<double> T_a;	// Diffuse transmittance per unit solid angle [1/sr]
    vec1<double> T_t;	// Diffuse transmittance per unit time [1/ps]

    double T_total{};	// Total diffuse transmittance
    double T_error{};	// Standard error for diffuse transmittance

    double Tb_total{};	// Ballistic transmittance
    double Tb_error{};	// Standard error for ballistic transmission

    /* Absorption */

    vec3<double> A_rzt; // Rate of absorption per unit volume, per unit time [1/(cm� ps]
    vec2<double> A_rz;	// Rate of absorption per unit volume [1/cm�]
    vec2<double> A_zt;	// Rate of absorption per unit time [1/(cm ps)]
    vec1<double> A_z;	// Absorption per unit depth [1/cm]
    vec1<double> A_t;	// Absorption per unit time [1/ps]

    double A_total{};	// Total absorption
    double A_error{};	// Standard error for absorption

    vec2<double> Ab_zt;	// Ballistic absorption per unit depth, per unit time [1/(cm ps)]
    vec1<double> Ab_z;	// Ballistic absorption per unit depth [1/cm]
};
