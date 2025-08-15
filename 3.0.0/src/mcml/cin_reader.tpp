#pragma once


#include <tuple>
#include <ranges>
#include <string>
#include <istream>
#include <iostream>
#include <algorithm>
#include <functional>

#include "cin_reader.hpp"
#include "reader_util.hpp"


static std::string trim(const std::string& str) {
    auto start = str.find_first_not_of(" \t\n\r");
    auto end = str.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

static std::string next_line_in(std::istream& in, bool* aborted)
{
    *aborted = false;

    std::string line;
    while (std::getline(in, line)) {
        // Find first non-whitespace character
        auto it = std::ranges::find_if(line, [](char c) {
            return !std::isspace(c);
        });

        // Skip whitespace-only lines
        if (it == line.end()) {
            continue;
        }

        // Skip comment lines
        if (*it == '#') {
            continue;
        }

        // Abort input
        if (*it == 'Q') {
            *aborted = true;
            return {};
        }

        return line;
    }

    // Return empty string if no valid lines found
    std::cerr << "Error: No valid data line found." << std::endl;
    return {};
}

static std::string default_prompt(auto type) {
    using T = decltype(type);
    if      constexpr (std::is_same_v<T, std::string>) { return "Input string"; }
    else if constexpr (std::is_same_v<T, bool>) { return "Input boolean (0 or 1)"; }
    else if constexpr (std::is_same_v<T, char>) { return "Input character"; }
    else if constexpr (std::is_arithmetic_v<T>) { return "Input number"; }
    else { return "Input"; }
}

static std::string default_error(auto type) {
    using T = decltype(type);
    if constexpr (std::is_arithmetic_v<T>) { return "Invalid value"; }
    else { return "Invalid input"; }
}


template <typename... Ts> std::tuple<bool, Ts...>
read_in(std::istream& in, std::string prompt, std::string error, bool allow_opt, std::function<bool(const std::tuple<Ts...>&)> cond)
{
    // Set prompt if not provided
    if (prompt.empty()) {
        if constexpr (sizeof...(Ts) == 1) {
            prompt = default_prompt(Ts{}...);
        }
        else {
            prompt = "Input values separated by spaces";
        }
    }
    if (!allow_opt) {
        prompt += " (or Q to abort)";
    }
    prompt += ": ";

    // Set error message if not provided
    if (error.empty()) {
        if constexpr (sizeof...(Ts) == 1) {
            error = default_error(Ts{}...);
        }
        else {
            error = "Invalid input";
        }
    }
    error += ". Please try again.";

    // Initialize variables
    int retries = 10;
    bool success = false;
    std::tuple<Ts...> values;

    // Repeat until valid input is received and the condition (if provided) is met
    do {
        retries--;
        std::cout << prompt;

        bool aborted = false;
        std::string line = next_line_in(in, &aborted);

        if (line.empty()) {
            std::cerr << error << std::endl;
            continue;
        }

        if (allow_opt && aborted) {
            std::cout << "Aborted." << std::endl;
            return { false, Ts{}... };
        }

        std::vector<alpha_num> vec;
        std::vector<alpha_num> expected = { alpha_num{Ts{}}... };
        success = extract(line, vec, expected, allow_opt);

        if (success && vec.size() == sizeof...(Ts)) {
            // Extract values into the tuple
            std::apply([&vec](Ts&... args) {
                size_t index = 0;
                ((args = std::get<Ts>(vec[index++])), ...);
            }, values);

            // Check the condition if provided
            if (cond == nullptr || cond(values)) {
                // Success: return flattened tuple with bool and extracted values
                return std::tuple_cat(std::make_tuple(true), values);
            }
            else {
                success = false; // Force retry
            }
        }
        else {
            std::cout << error << std::endl;
        }
    } while (!success && retries >= 0);

    // Failure: return false with default-constructed values
    return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...>{});
}


template <typename T> std::tuple<bool, T> 
read_in(std::istream& input, std::string prompt = {}, std::string error = {}, bool allow_opt = false, std::function<bool(const T&)> cond = nullptr)
{
    auto [success, value] = read_in<T>(input, prompt, error, allow_opt, [&cond](const std::tuple<T>& values) {
        if (!cond) { return true; }
        return cond(std::get<0>(values));
    });
    return { success, value };
}
