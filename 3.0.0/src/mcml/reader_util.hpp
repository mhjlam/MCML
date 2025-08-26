#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

/** @brief Variant type for alphanumeric values */
using alpha_num = std::variant<char, int, double, std::string>;

/** @brief Optional alphanumeric value */
using opt_alpha_num = std::optional<alpha_num>;

/** @brief Tuple of 2 doubles */
using double2 = std::tuple<double, double>;

/** @brief Tuple of 3 doubles */
using double3 = std::tuple<double, double, double>;

/** @brief Tuple of 4 doubles */
using double4 = std::tuple<double, double, double, double>;

/** @brief Tuple of 5 doubles */
using double5 = std::tuple<double, double, double, double, double>;

/** @brief Read next non-empty line from input stream */
extern std::string next_line(std::istream& in);

/**
 * @brief Extract values from string and validate against expected types
 * @param in Input string to parse
 * @param out Output vector of extracted values
 * @param expected Vector of expected value types
 * @param allow_opt Allow optional values (default: false)
 * @return True if extraction successful, false otherwise
 */
extern bool extract(const std::string& in, std::vector<alpha_num>& out, const std::vector<alpha_num>& expected,
					bool allow_opt = false);

/**
 * @brief Extract single value from string and validate against expected type
 * @param in Input string to parse
 * @param out Output extracted value
 * @param expected Expected value type
 * @return True if extraction successful, false otherwise
 */
extern bool extract(const std::string& in, alpha_num& out, const alpha_num& expected);

/** @brief Convert string to uppercase in-place */
extern std::string& uppercase(std::string& str);
