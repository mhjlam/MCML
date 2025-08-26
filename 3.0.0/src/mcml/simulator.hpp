#pragma once

#include "mcml.hpp"
#include "observer.hpp"

#include <memory>

class Reader;
class ReaderCin;
class Writer;
class Tracer;
class Timer;
class Random;

class Simulator
{
public:
	Simulator(std::string in_file = {});
	~Simulator() = default;

	// Read input file and do number of runs
	void simulate();

	// Input results of previous simulation, add photons, and do one run
	void resume();

	// Read params interactively, then do one run
	void interactive();

	// Observer pattern methods
	void add_observer(std::shared_ptr<mcml::SimulationObserver> observer);
	void remove_observer(const std::shared_ptr<mcml::SimulationObserver>& observer);
	const mcml::ProgressInfo& progress() const;

	// Read params interactively, show edit menu, then do one run
	void interactive_edit();

private:
	// Do one run non-interactively
	void run(std::size_t run_index = 0, bool start_new = true);

	// Do one run interactively
	// Return true if simulation should continue
	bool interactive_run();

	bool edit_menu(char command);

	bool validate_params();

	// Report start of run and target photons / time
	void report_target(std::size_t runs_remaining);

	// Report estimated time
	void report_progress(std::size_t photons_done);

	// Report time, photon number traced, write results
	void report_result();

	bool prompt_filename(std::string& result, std::string file_type = ".mci");

	// Continue to change input parameters or quit.
	bool prompt_edit();

	void edit_mediums();
	void edit_output();
	void edit_grid();
	void edit_grid_spacing(); // Change dz, dr, dt only (for 'd' command)
	void edit_grid_size();    // Change nz, nr, nt, na only (for 'n' command)
	void edit_record();
	void edit_weight();
	void edit_layers();
	void edit_target();
	void edit_source();
	void edit_random_seed();
	void edit_source_position();

	void show_edit_menu_help();

	void scale_reflectance(Radiance& radiance, ScaleMode mode = ScaleMode::Scale);
	void scale_transmittance(Radiance& radiance, ScaleMode mode = ScaleMode::Scale);
	void scale_absorption(Radiance& radiance, ScaleMode mode = ScaleMode::Scale);

private:
	std::string m_mci;
	RunParams m_params;

	std::shared_ptr<Reader> m_reader;
	std::shared_ptr<ReaderCin> m_cin_reader;
	std::shared_ptr<Writer> m_writer;
	std::shared_ptr<Timer> m_timer;
	std::shared_ptr<Random> m_random;
	std::shared_ptr<Tracer> m_tracer;

	// Observer pattern for progress reporting
	mcml::SimulationSubject m_observer_subject;
};
