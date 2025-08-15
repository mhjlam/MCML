#pragma once

#include <string>
#include <istream>

#include "mcml.hpp"
#include "reader.hpp"

class CinReader : public Reader
{

public:
    CinReader() : Reader{ {} } {}
    ~CinReader() = default;

    // Read the input parameters for all runs and count number of runs.
    bool ReadParams(std::istream& in, RunParams& params) override;

    // Read the mediums list.
    bool ReadMediums(std::istream& in, vec1<Layer>& out) override;

    // Read the input name and the input format.
    bool ReadOutput(std::istream& in, std::string& out) override;

    // Read the parameters of all layers.
    bool ReadLayers(std::istream& in, RunParams& params, vec1<Layer>& out) override;

    // Read the beam source type (Pencil or Isotropic) and starting position.
    bool ReadSource(std::istream& in, RunParams& params, LightSource& out) override;

    // Read the grid separation parameters (z, r, t) and number of grid lines (z, r, t, and alpha).
    bool ReadGrid(std::istream& in, Grid& out) override;

    // Read which quantity is to be scored.
    bool ReadRecord(std::istream& in, RunParams& params, Record& out) override;

    // Read the number of photons and computation time limit.
    bool ReadTarget(std::istream& in, RunParams& params, Target& out, bool add = false) override;

    // Read the weight threshold.
    bool ReadWeight(std::istream& in, double& out) override;

};
