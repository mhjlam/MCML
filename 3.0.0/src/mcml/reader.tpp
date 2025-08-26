#pragma once

#include "reader.hpp"
#include "reader_util.hpp"

#include <concepts>
#include <functional>
#include <iostream>
#include <istream>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>

/**
 * @brief C++20 concepts for template constraints
 */
namespace mcml::concepts
{
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<typename T>
concept Parseable = Numeric<T> || std::same_as<T, std::string>;

template<typename F, typename... Args>
concept Predicate = std::invocable<F, Args...> && std::convertible_to<std::invoke_result_t<F, Args...>, bool>;
} // namespace mcml::concepts

/**
 * @brief Get human-readable type name for error messages
 * @tparam T Type to get name for
 * @return Readable type name
 */
template<typename T>
std::string typename_name() {
	std::string readable_name = typeid(T).name();

	// Simplify common types
	if (readable_name.find("basic_string") != std::string::npos) {
		readable_name = "string";
	}
	else if (readable_name.find("double") != std::string::npos) {
		readable_name = "double";
	}
	else if (readable_name.find("int") != std::string::npos) {
		readable_name = "int";
	}
	return readable_name;
}

/**
 * @brief Get comma-separated list of type names
 * @tparam Ts Parameter pack of types
 * @return Comma-separated type names
 */
template<typename... Ts>
std::string typename_types() {
	if constexpr (sizeof...(Ts) == 0) {
		return "";
	}
	else {
		std::ostringstream oss;
		((oss << typename_name<Ts>() << ", "), ...);
		std::string result = oss.str();
		if (result.size() >= 2) {
			result.erase(result.size() - 2); // Remove trailing ", "
		}
		return result;
	}
}

/**
 * @brief Read multiple values from input stream with validation
 * @tparam Ts Parameter pack of parseable types
 * @tparam Predicate Callable that validates the parsed tuple
 * @param in Input stream to read from
 * @param error Error message to display on failure
 * @param check Validation predicate (optional)
 * @return Tuple containing success flag and parsed values
 */
template<mcml::concepts::Parseable... Ts, typename Predicate = std::nullptr_t>
	requires(mcml::concepts::Predicate<Predicate, const std::tuple<Ts...>&> || std::same_as<Predicate, std::nullptr_t>)
std::tuple<bool, Ts...> read(std::istream& in, std::string error = {}, Predicate check = nullptr) {
	if (error.empty()) {
		error = "Invalid values for types (" + typename_types<Ts...>() + ").";
	}

	// Get next data line
	std::string line = next_line(in);
	if (line.empty()) {
		std::cerr << error << std::endl;
		return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...> {});
	}

	std::vector<alpha_num> vec;
	std::vector<alpha_num> expected = {alpha_num {Ts {}}...};
	bool success = extract(line, vec, expected);

	// Check if the number of extracted values matches the expected number
	if (!success || vec.size() != sizeof...(Ts)) {
		std::cerr << error << std::endl;
		return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...> {});
	}

	// Extract values into a tuple
	std::tuple<Ts...> values;
	try {
		std::apply(
			[&vec](Ts&... args) {
				size_t index = 0;
				((args = std::visit(
					  [](auto&& value) -> Ts {
						  if constexpr (std::is_convertible_v<decltype(value), Ts>) {
							  return static_cast<Ts>(value);
						  }
						  else {
							  throw std::bad_cast();
						  }
					  },
					  vec[index++])),
				 ...);
			},
			values);
	}
	catch (const std::bad_cast&) {
		std::cerr << "Error converting extracted values: incompatible types." << std::endl;
		return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...> {});
	}
	catch (...) {
		std::cerr << "Error converting extracted values." << std::endl;
		return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...> {});
	}

	// Check if the condition is met
	if constexpr (!std::same_as<Predicate, std::nullptr_t>) {
		if (!check(values)) {
			std::cerr << error << " (validation failed)" << std::endl;
			return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...> {});
		}
	}

	// Return true (success) and values
	return std::tuple_cat(std::make_tuple(true), values);
}

/**
 * @brief Read single value from input stream with validation
 * @tparam T Type to parse
 * @tparam Predicate Callable that validates the parsed value
 * @param in Input stream to read from
 * @param error Error message to display on failure
 * @param check Validation predicate (optional)
 * @return Tuple containing success flag and parsed value
 */
template<mcml::concepts::Parseable T, typename Predicate>
	requires(mcml::concepts::Predicate<Predicate, const T&>
			 && !mcml::concepts::Predicate<Predicate, const std::tuple<T>&>)
std::tuple<bool, T> read_single(std::istream& in, std::string error = {}, Predicate check = nullptr) {
	auto [success, value] = read<T>(in, error, [check](const std::tuple<T>& values) -> bool {
		if constexpr (std::same_as<Predicate, std::nullptr_t>) {
			return true;
		}
		else {
			return check(std::get<0>(values));
		}
	});
	return {success, value};
}

/**
 * @brief Overload for nullptr predicate
 */
template<mcml::concepts::Parseable T>
std::tuple<bool, T> read_single(std::istream& in, std::string error = {}, std::nullptr_t check = nullptr) {
	auto [success, value] = read<T>(in, error);
	return {success, value};
}

/**
 * @brief Parse multiple values from a string line with validation
 * @tparam Ts Parameter pack of parseable types
 * @tparam Predicate Callable that validates the parsed tuple
 * @param line String line to parse
 * @param error Error message to display on failure
 * @param check Validation predicate (optional)
 * @return Tuple containing success flag and parsed values
 */
template<mcml::concepts::Parseable... Ts, typename Predicate = std::nullptr_t>
	requires(mcml::concepts::Predicate<Predicate, const std::tuple<Ts...>&> || std::same_as<Predicate, std::nullptr_t>)
std::tuple<bool, Ts...> read_line(std::string& line, std::string error = {}, Predicate check = nullptr) {
	if (error.empty()) {
		error = "Invalid values for types (" + typename_types<Ts...>() + ").";
	}

	// Extract data from string
	std::vector<alpha_num> vec;
	std::vector<alpha_num> expected = {alpha_num {Ts {}}...};
	bool success = extract(line, vec, expected);

	// Check if the number of extracted values matches the expected number
	if (!success || vec.size() != sizeof...(Ts)) {
		std::cerr << error << std::endl;
		return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...> {});
	}

	// Extract values into a tuple
	std::tuple<Ts...> values;
	try {
		std::apply(
			[&vec](Ts&... args) {
				size_t index = 0;
				((args = std::visit(
					  [](auto&& value) -> Ts {
						  if constexpr (std::is_convertible_v<decltype(value), Ts>) {
							  return static_cast<Ts>(value);
						  }
						  else {
							  throw std::bad_cast();
						  }
					  },
					  vec[index++])),
				 ...);
			},
			values);
	}
	catch (const std::bad_cast&) {
		std::cerr << "Error converting extracted values: incompatible types." << std::endl;
		return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...> {});
	}
	catch (...) {
		std::cerr << "Error converting extracted values." << std::endl;
		return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...> {});
	}

	// Check if the condition is met
	if constexpr (!std::same_as<Predicate, std::nullptr_t>) {
		if (!check(values)) {
			std::cerr << error << " (validation failed)" << std::endl;
			return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...> {});
		}
	}

	// Return true (success) and values
	return std::tuple_cat(std::make_tuple(true), values);
}

/**
 * @brief Parse single value from a string line with validation
 * @tparam T Type to parse
 * @tparam Predicate Callable that validates the parsed value
 * @param line String line to parse
 * @param error Error message to display on failure
 * @param check Validation predicate (optional)
 * @return Tuple containing success flag and parsed value
 */
template<mcml::concepts::Parseable T, typename Predicate>
	requires(mcml::concepts::Predicate<Predicate, const T&>
			 && !mcml::concepts::Predicate<Predicate, const std::tuple<T>&>)
std::tuple<bool, T> read_line_single(std::string& line, std::string error = {}, Predicate check = nullptr) {
	auto [success, value] = read_line<T>(line, error, [check](const std::tuple<T>& values) -> bool {
		if constexpr (std::same_as<Predicate, std::nullptr_t>) {
			return true;
		}
		else {
			return check(std::get<0>(values));
		}
	});
	return {success, value};
}

/**
 * @brief Overload for nullptr predicate
 */
template<mcml::concepts::Parseable T>
std::tuple<bool, T> read_line_single(std::string& line, std::string error = {}, std::nullptr_t check = nullptr) {
	auto [success, value] = read_line<T>(line, error);
	return {success, value};
}
