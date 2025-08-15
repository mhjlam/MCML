#pragma once

#include "mcml.hpp"

#include <memory>


class Reader;
class CinReader;

class Writer;
class CoutWriter;

class Tracer;

class Timer;
class Random;


class Simulator
{
public:
    Simulator(std::string in_file = {});
    ~Simulator() = default;

    // Read input file and do number of runs
    void Simulate();

    // Input results of previous simulation, add photons, and do one run
    void Resume();

    // Read params interactively, then do one run
    void Interactive();

    // Read params interactively, show edit menu, then do one run
    void InteractiveEdit();

private:
    // Do one run non-interactively
    void run(std::size_t run_index = 0, bool start_new = true);

    // Do one run interactively
    // Return true if simulation should continue
    bool interactiveRun();

    bool editMenu(char command);

    bool validateParams();


    // Report start of run and target photons / time
    void reportTarget(std::size_t runs_remaining);

    // Report estimated time
    void reportProgress(std::size_t photons_done);

    // Report time, photon number traced, write results
    void reportResult();

    bool promptFileName(std::string& result, std::string file_type = ".mci");

    // Continue to change input parameters or quit.
    bool promptEdit();

    void editMediums();
    void editOutput();
    void editGrid();
    void editRecord();
    void editWeight();
    void editLayers();
    void editTarget();
    void editSource();

    void showEditMenuHelp();


    void scaleReflectance(Radiance& radiance, ScaleMode mode = ScaleMode::Scale);
    void scaleTransmittance(Radiance& radiance, ScaleMode mode = ScaleMode::Scale);
    void scaleAbsorption(Radiance& radiance, ScaleMode mode = ScaleMode::Scale);


private:
    std::string m_mci;

    RunParams m_params;

    std::shared_ptr<Reader> m_reader;
    std::shared_ptr<CinReader> m_cin_reader;

    std::shared_ptr<Writer> m_writer;
    std::shared_ptr<CoutWriter> m_cout_writer;

    std::shared_ptr<Timer> m_timer;
    std::shared_ptr<Random> m_random;

    std::shared_ptr<Tracer> m_tracer;
};
