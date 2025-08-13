#include "reader_util.hpp"

#include <sstream>
#include <iostream>
#include <algorithm>


std::string next_line(std::istream& in)
{
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

        return line;
    }

    // Return empty string if no valid lines found
    std::cerr << "Error: No valid data line found." << std::endl;
    return {};
}


bool extract(const std::string& input, std::vector<alpha_num>& output, const std::vector<alpha_num>& expected, bool allow_opt)
{
    auto parse = [&](const std::string& str, const alpha_num& type) -> opt_alpha_num {
        std::istringstream iss(str);
        if (std::holds_alternative<char>(type)) {
            char value;
            if (iss >> value) { return value; }
        }
        else if (std::holds_alternative<int>(type)) {
            int value;
            if (iss >> value) { return value; }
        }
        else if (std::holds_alternative<double>(type)) {
            double value;
            if (iss >> value) { return value; }
        }
        else if (std::holds_alternative<std::string>(type)) {
            return str;
        }
        return std::nullopt;
    };

    std::istringstream iss(input);
    std::string token;

    for (const auto& type : expected) {
        if (!(iss >> token)) { break; }

        opt_alpha_num value = parse(token, type);

        if (!value.has_value()) {
            if (allow_opt) {
                continue;
            }
            return false;
        }
        else {
            output.push_back(value.value());
        }
    }

    if (!allow_opt && output.size() != expected.size()) {
        return false;
    }

    return true;
}

bool extract(const std::string& input, alpha_num& output, const alpha_num& expected)
{
    auto parse = [&](const std::string& str, const alpha_num& type) -> opt_alpha_num {
        std::istringstream iss(str);
        if (std::holds_alternative<char>(type)) {
            char value;
            if (iss >> value) { return value; }
        }
        else if (std::holds_alternative<int>(type)) {
            int value;
            if (iss >> value) { return value; }
        }
        else if (std::holds_alternative<double>(type)) {
            double value;
            if (iss >> value) { return value; }
        }
        else if (std::holds_alternative<std::string>(type)) {
            return str;
        }
        return std::nullopt;
    };

    // Attempt to parse the input
    opt_alpha_num value = parse(input, expected);

    // Check if parsing succeeded
    if (value.has_value()) {
        output = value.value();
        return true;
    }

    return false;
}

std::string& uppercase(std::string& string)
{
    std::ranges::transform(string, string.begin(), [](unsigned char ch) {
        return std::toupper(ch);
    });
    return string;
}
