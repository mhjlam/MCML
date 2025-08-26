/**
 * @file observer.hpp
 * @brief Observer pattern implementation for simulation progress reporting
 * @author M.H.J. Lam
 * @date 2025
 */

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

namespace mcml {

/**
 * @brief Progress information for simulation updates
 */
struct ProgressInfo {
    std::size_t current_photon{0};      ///< Current photon being processed
    std::size_t total_photons{0};       ///< Total photons to process
    double elapsed_time{0.0};           ///< Elapsed time in seconds
    double estimated_remaining{0.0};    ///< Estimated remaining time in seconds
    double progress_percent{0.0};       ///< Progress as percentage (0-100)
    std::string current_stage;          ///< Current processing stage
    
    /**
     * @brief Calculate progress percentage
     */
    void update_progress() {
        if (total_photons > 0) {
            progress_percent = (static_cast<double>(current_photon) / static_cast<double>(total_photons)) * 100.0;
        }
    }
};

/**
 * @brief Simulation event types
 */
enum class SimulationEvent {
    Started,            ///< Simulation has started
    PhotonProcessed,    ///< Single photon processed
    Progress,           ///< General progress update
    StageChanged,       ///< Processing stage changed
    Completed,          ///< Simulation completed
    Error,              ///< Error occurred
    Paused,             ///< Simulation paused
    Resumed             ///< Simulation resumed
};

/**
 * @brief Abstract observer interface for simulation events
 */
class SimulationObserver {
public:
    virtual ~SimulationObserver() = default;
    
    /**
     * @brief Called when simulation event occurs
     * @param event The type of event
     * @param info Progress information
     * @param message Optional message for the event
     */
    virtual void on_event(SimulationEvent event, 
                         const ProgressInfo& info, 
                         const std::string& message = {}) = 0;
    
    /**
     * @brief Called for generic progress updates
     * @param info Current progress information
     */
    virtual void on_progress(const ProgressInfo& info) {
        on_event(SimulationEvent::Progress, info);
    }
    
    /**
     * @brief Called when error occurs
     * @param message Error message
     * @param info Current progress information
     */
    virtual void on_error(const std::string& message, const ProgressInfo& info) {
        on_event(SimulationEvent::Error, info, message);
    }
};

/**
 * @brief Console observer that prints progress to stdout
 */
class ConsoleObserver : public SimulationObserver {
private:
    std::chrono::steady_clock::time_point last_update_;
    std::chrono::milliseconds update_interval_{1000}; // 1 second
    bool verbose_{false};
    
public:
    explicit ConsoleObserver(bool verbose = false, std::chrono::milliseconds interval = std::chrono::milliseconds{1000})
        : last_update_(std::chrono::steady_clock::now()), update_interval_(interval), verbose_(verbose) {}
    
    void on_event(SimulationEvent event, const ProgressInfo& info, const std::string& message = {}) override {
        auto now = std::chrono::steady_clock::now();
        
        switch (event) {
            case SimulationEvent::Started:
                std::cout << "Starting simulation with " << info.total_photons << " photons\n";
                last_update_ = now;
                break;
                
            case SimulationEvent::Progress:
                // Throttle progress updates
                if (now - last_update_ >= update_interval_) {
                    print_progress(info);
                    last_update_ = now;
                }
                break;
                
            case SimulationEvent::PhotonProcessed:
                if (verbose_ && (info.current_photon % 10000 == 0)) {
                    print_progress(info);
                }
                break;
                
            case SimulationEvent::StageChanged:
                std::cout << "Stage: " << info.current_stage << std::endl;
                break;
                
            case SimulationEvent::Completed:
                std::cout << "Simulation completed in " << info.elapsed_time << " seconds\n";
                break;
                
            case SimulationEvent::Error:
                std::cout << "Error: " << message << std::endl;
                break;
                
            case SimulationEvent::Paused:
                std::cout << "Simulation paused\n";
                break;
                
            case SimulationEvent::Resumed:
                std::cout << "Simulation resumed\n";
                break;
        }
    }
    
private:
    void print_progress(const ProgressInfo& info) {
        std::cout << "Progress: " << info.current_photon << "/" << info.total_photons 
                  << " (" << std::fixed << std::setprecision(1) << info.progress_percent << "%) ";
        
        if (info.estimated_remaining > 0) {
            std::cout << "ETA: " << format_time(info.estimated_remaining);
        }
        
        if (!info.current_stage.empty()) {
            std::cout << " [" << info.current_stage << "]";
        }
        
        std::cout << std::endl;
    }
    
    std::string format_time(double seconds) const {
        int hours = static_cast<int>(seconds / 3600);
        int minutes = static_cast<int>((seconds - hours * 3600) / 60);
        int secs = static_cast<int>(seconds - hours * 3600 - minutes * 60);
        
        if (hours > 0) {
            return std::to_string(hours) + "h " + std::to_string(minutes) + "m " + std::to_string(secs) + "s";
        } else if (minutes > 0) {
            return std::to_string(minutes) + "m " + std::to_string(secs) + "s";
        } else {
            return std::to_string(secs) + "s";
        }
    }
};

/**
 * @brief File observer that logs progress to a file
 */
class FileObserver : public SimulationObserver {
private:
    std::string filename_;
    std::ofstream log_file_;
    
public:
    explicit FileObserver(const std::string& filename) : filename_(filename) {
        log_file_.open(filename_, std::ios::app);
        if (!log_file_.is_open()) {
            throw std::runtime_error("Could not open log file: " + filename_);
        }
    }
    
    ~FileObserver() override {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }
    
    void on_event(SimulationEvent event, const ProgressInfo& info, const std::string& message = {}) override {
        if (!log_file_.is_open()) return;
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        log_file_ << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] ";
        
        switch (event) {
            case SimulationEvent::Started:
                log_file_ << "STARTED: " << info.total_photons << " photons";
                break;
            case SimulationEvent::Progress:
                log_file_ << "PROGRESS: " << info.current_photon << "/" << info.total_photons 
                         << " (" << info.progress_percent << "%)";
                break;
            case SimulationEvent::Completed:
                log_file_ << "COMPLETED: " << info.elapsed_time << "s";
                break;
            case SimulationEvent::Error:
                log_file_ << "ERROR: " << message;
                break;
            default:
                log_file_ << "EVENT: " << static_cast<int>(event);
                if (!message.empty()) {
                    log_file_ << " - " << message;
                }
                break;
        }
        
        log_file_ << std::endl;
        log_file_.flush();
    }
};

/**
 * @brief Observable subject that manages observers
 */
class SimulationSubject {
private:
    std::vector<std::shared_ptr<SimulationObserver>> observers_;
    ProgressInfo current_progress_;
    std::chrono::steady_clock::time_point start_time_;
    
public:
    /**
     * @brief Add an observer
     */
    void add_observer(std::shared_ptr<SimulationObserver> observer) {
        observers_.push_back(std::move(observer));
    }
    
    /**
     * @brief Remove an observer
     */
    void remove_observer(const std::shared_ptr<SimulationObserver>& observer) {
        observers_.erase(
            std::remove(observers_.begin(), observers_.end(), observer),
            observers_.end()
        );
    }
    
    /**
     * @brief Notify all observers of an event
     */
    void notify_event(SimulationEvent event, const std::string& message = {}) {
        update_progress_timing();
        
        for (auto& observer : observers_) {
            if (observer) {
                observer->on_event(event, current_progress_, message);
            }
        }
    }
    
    /**
     * @brief Set total number of photons
     */
    void set_total_photons(std::size_t total) {
        current_progress_.total_photons = total;
        start_time_ = std::chrono::steady_clock::now();
    }
    
    /**
     * @brief Update current photon count
     */
    void set_current_photon(std::size_t current) {
        current_progress_.current_photon = current;
        current_progress_.update_progress();
        
        // Estimate remaining time
        update_progress_timing();
    }
    
    /**
     * @brief Set current processing stage
     */
    void set_stage(const std::string& stage) {
        current_progress_.current_stage = stage;
    }
    
    /**
     * @brief Get current progress info
     */
    const ProgressInfo& progress() const {
        return current_progress_;
    }
    
private:
    void update_progress_timing() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time_).count();
        current_progress_.elapsed_time = elapsed;
        
        // Estimate remaining time based on current progress
        if (current_progress_.current_photon > 0 && current_progress_.total_photons > 0) {
            double rate = static_cast<double>(current_progress_.current_photon) / elapsed;
            double remaining_photons = static_cast<double>(current_progress_.total_photons - current_progress_.current_photon);
            current_progress_.estimated_remaining = remaining_photons / rate;
        }
    }
};

} // namespace mcml
