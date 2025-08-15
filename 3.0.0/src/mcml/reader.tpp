#pragma once

#include <tuple>
#include <string>
#include <istream>
#include <variant>
#include <iostream>
#include <functional>

#include "reader.hpp"
#include "reader_util.hpp"


template <typename T> std::string typename_name()
{
    std::string readable_name = typeid(T).name();

    // Simplify common types like std::string
    if (readable_name == "class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> >") {
        readable_name = "string";
    }
    return readable_name;
}


template <typename... Ts> std::string typename_types()
{
    std::ostringstream oss;
    ((oss << typename_name<Ts>() << ", "), ...);
    std::string result = oss.str();
    if (!result.empty()) {
        result.pop_back(); // Remove trailing space
        result.pop_back(); // Remove trailing comma
    }
    return result;
}

template <typename... Ts> std::tuple<bool, Ts...>
read(std::istream& in, std::string error = {}, std::function<bool(const std::tuple<Ts...>&)> check = nullptr)
{
    if (error.empty()) {
        error = "Invalid values for types (" + typename_types<Ts...>() + ").";
    }

    // Get next data line
    std::string line = next_line(in);
    if (line.empty()) {
        std::cerr << error << std::endl;
        return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...>{});
    }

    std::vector<alpha_num> vec;
    std::vector<alpha_num> expected = { alpha_num{Ts{}}... };
    bool success = extract(line, vec, expected);

    // Check if the number of extracted values matches the expected number
    if (!success || vec.size() != sizeof...(Ts)) {
        std::cerr << error << std::endl;
        return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...>{});
    }

    // Extract values into a tuple
    std::tuple<Ts...> values;
    try {
        std::apply([&vec](Ts&... args) {
            size_t index = 0;
            ((args = std::visit([](auto&& value) -> Ts {
                if constexpr (std::is_convertible_v<decltype(value), Ts>) {
                    return static_cast<Ts>(value);
                }
                else {
                    throw std::bad_cast();
                }
            }, vec[index++])), ...);
        }, values);
    }
    catch (const std::bad_cast&) {
        std::cerr << "Error converting extracted values: incompatible types." << std::endl;
        return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...>{});
    }
    catch (...) {
        std::cerr << "Error converting extracted values." << std::endl;
        return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...>{});
    }

    // Check if the condition is met
    if (check != nullptr && !check(values)) {
        return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...>{});
    }

    // Return true (success) and values
    return std::tuple_cat(std::make_tuple(true), values);
}


template <typename T> std::tuple<bool, T>
read(std::istream& in, std::string error = {}, std::function<bool(const T&)> check = nullptr)
{
    auto [success, value] = read<T>(in, error, [&check](const std::tuple<T>& values) {
        if (!check) { return true; }
        return check(std::get<0>(values));
    });
    return { success, value };
}

template <typename... Ts> std::tuple<bool, Ts...>
read_line(std::string& line, std::string error = {}, std::function<bool(const std::tuple<Ts...>&)> check = nullptr)
{
    if (error.empty()) {
        error = "Invalid values for types (" + typename_types<Ts...>() + ").";
    }

    // Extract data from string
    std::vector<alpha_num> vec;
    std::vector<alpha_num> expected = { alpha_num{Ts{}}... };
    bool success = extract(line, vec, expected);

    // Check if the number of extracted values matches the expected number
    if (!success || vec.size() != sizeof...(Ts))
    {
        std::cerr << error << std::endl;
        return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...>{});
    }

    // Extract values into a tuple
    std::tuple<Ts...> values;
    try {
        std::apply([&vec](Ts&... args) {
            size_t index = 0;
            ((args = std::visit([](auto&& value) -> Ts {
                if constexpr (std::is_convertible_v<decltype(value), Ts>) {
                    return static_cast<Ts>(value);
                }
                else {
                    throw std::bad_cast();
                }
            }, vec[index++])), ...);
        }, values);
    }
    catch (const std::bad_cast&) {
        std::cerr << "Error converting extracted values: incompatible types." << std::endl;
        return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...>{});
    }
    catch (...) {
        std::cerr << "Error converting extracted values." << std::endl;
        return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...>{});
    }

    // Check if the condition is met
    if (check != nullptr && !check(values)) {
        return std::tuple_cat(std::make_tuple(false), std::tuple<Ts...>{});
    }

    // Return true (success) and values
    return std::tuple_cat(std::make_tuple(true), values);
}

template <typename T> std::tuple<bool, T>
read_line(std::string& line, std::string error = {}, std::function<bool(const T&)> check = nullptr)
{
    auto [success, value] = read_line<T>(line, error, [&check](const std::tuple<T>& values) {
        if (!check) { return true; }
        return check(std::get<0>(values));
    });
    return { success, value };
}

