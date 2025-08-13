#pragma once

#include <memory>
#include <iostream>


class Random;

struct Radiance;
struct RunParams;


class Writer
{
public:
    Writer(std::string filename = {});
    ~Writer();

    void WriteMediums(std::ostream& output, RunParams& params);
    void WriteFilename(std::ostream& output, RunParams& params);
    void WriteGridParams(std::ostream& output, RunParams& params);
    void WriteGridSize(std::ostream& output, RunParams& params);
    void WriteRecord(std::ostream& output, RunParams& params);
    void WriteWeight(std::ostream& output, RunParams& params);
    void WriteRandomSeed(std::ostream& output, RunParams& params);
    void WriteLayers(std::ostream& output, RunParams& params);
    void WriteEndCriteria(std::ostream& output, RunParams& params);
    void WriteSourceType(std::ostream& output, RunParams& params);
    void WritePhotonSource(std::ostream& output, RunParams& params);
    void WriteParams(std::ostream& output, RunParams& params);
    void WriteVersion(std::ostream& output, const std::string_view& version);
    void WriteRandomizer(std::ostream& output, std::shared_ptr<Random> random);

    // Write result of tracer to an output file.
    void WriteResults(std::ostream& output, RunParams& params, Radiance& radiance, std::shared_ptr<Random> random);

    // Write RAT totals to an output file.
    void WriteRadiance(std::ostream& output, Radiance& radiance);

    // Ballistic absorption per unit depth, per unit time [1/(cm ps)]
    void WriteAb_zt(std::ostream& output, std::size_t Nz, std::size_t Nt, Radiance& radiance);

    // Rate of absorption per unit volume, per unit time [1/(cm³ ps]
    void WriteA_rzt(std::ostream& output, std::size_t Nr, std::size_t Nz, std::size_t Nt, Radiance& radiance);

    // Ballistic absorption per unit depth [1/cm]
    void WriteAb_z(std::ostream& output, std::size_t Nz, Radiance& radiance);

    // Rate of absorption per unit volume [1/cm³]
    void WriteA_rz(std::ostream& output, std::size_t Nr, std::size_t Nz, Radiance& radiance);

    // Rate of absorption per unit time [1/(cm ps)]
    void WriteA_zt(std::ostream& output, std::size_t Nz, std::size_t Nt, Radiance& radiance);

    // Absorption per unit depth [1/cm]
    void WriteA_z(std::ostream& output, std::size_t Nz, Radiance& radiance);

    // Absorption per unit time [1/ps]
    void WriteA_t(std::ostream& output, std::size_t Nt, Radiance& radiance);

    // Diffuse reflectance per unit area, per unit solid angle, per unit time [1/(cm² sr ps)]
    void WriteR_rat(std::ostream& output, std::size_t Nr, std::size_t Na, std::size_t Nt, Radiance& radiance);

    // Diffuse reflectance per unit area, per unit solid angle [1/(cm² sr)]
    void WriteR_ra(std::ostream& output, std::size_t Nr, std::size_t Na, Radiance& radiance);

    // Diffuse reflectance per unit solid angle, per unit time [1/sr ps]
    void WriteR_rt(std::ostream& output, std::size_t Nr, std::size_t Nt, Radiance& radiance);

    // Diffuse reflectance per unit area, per unit time [1/cm² ps]
    void WriteR_at(std::ostream& output, std::size_t Na, std::size_t Nt, Radiance& radiance);

    // Diffuse reflectance distribution per unit area [1/cm²]
    void WriteR_r(std::ostream& output, std::size_t Nr, Radiance& radiance);

    // Diffuse reflectance per unit solid angle [1/sr]
    void WriteR_a(std::ostream& output, std::size_t Na, Radiance& radiance);

    // Diffuse reflectance per unit time [1/ps]
    void WriteR_t(std::ostream& output, std::size_t Nt, Radiance& radiance);

    // Diffuse transmittance per unit area, per unit solid angle, per unit time [1/(cm² sr ps)]
    void WriteT_rat(std::ostream& output, std::size_t Nr, std::size_t Na, std::size_t Nt, Radiance& radiance);

    // Diffuse transmittance per unit area, per unit solid angle [1/(cm² sr)]
    void WriteT_ra(std::ostream& output, std::size_t Nr, std::size_t Na, Radiance& radiance);

    // Diffuse transmittance per unit solid angle, per unit time [1/sr ps]
    void WriteT_rt(std::ostream& output, std::size_t Nr, std::size_t Nt, Radiance& radiance);

    // Diffuse transmittance per unit area, per unit time [1/cm² ps]
    void WriteT_at(std::ostream& output, std::size_t Na, std::size_t Nt, Radiance& radiance);

    // Diffuse reflectance per unit area [1/cm²]
    void WriteT_r(std::ostream& output, std::size_t Nr, Radiance& radiance);

    // Diffuse reflectance per unit solid angle [1/sr]
    void WriteT_a(std::ostream& output, std::size_t Na, Radiance& radiance);

    // Diffuse reflectance per unit time [1/ps]
    void WriteT_t(std::ostream& output, std::size_t Nt, Radiance& radiance);

public:
    operator std::ostream& () { return *m_output; }

protected:
    std::unique_ptr<std::ostream> m_output;
};
