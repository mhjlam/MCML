#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <variant>
#include <optional>


using alpha_num = std::variant<char, int, double, std::string>;
using opt_alpha_num = std::optional<alpha_num>;

using double2 = std::tuple<double, double>;
using double3 = std::tuple<double, double, double>;
using double4 = std::tuple<double, double, double, double>;
using double5 = std::tuple<double, double, double, double, double>;


extern std::string next_line(std::istream& in);

extern bool extract(const std::string& in, std::vector<alpha_num>& out, const std::vector<alpha_num>& expected, bool allow_opt = false);
extern bool extract(const std::string& in, alpha_num& out, const alpha_num& expected);

extern std::string& uppercase(std::string& str);
