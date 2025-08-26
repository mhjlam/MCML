#pragma once

#include <chrono>
#include <format>
#include <optional>
#include <string>

/**
 * @brief High-resolution timer for performance measurement and time formatting
 */
class Timer
{
public:
	Timer() : rt0(std::chrono::system_clock::now()) {}

	/** @brief Reset the timer to current time */
	void reset() { rt0 = std::chrono::system_clock::now(); }

	/** @brief Get elapsed seconds since timer start or last reset */
	long long punch() const {
		auto now = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed = now - rt0;
		return std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
	}

	/** @brief Get elapsed time as formatted HH:MM:SS string */
	std::string hms_str(long long add_seconds = 0) const {
		auto seconds = punch() + add_seconds;
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::seconds(seconds));
		return std::format("{:%H:%M:%S}", duration);
	}

	/** @brief Get local time with optional seconds offset */
	static auto get_local_time(long long add_seconds = 0) {
		std::chrono::zoned_time local_time {std::chrono::current_zone(), std::chrono::system_clock::now()};
		return local_time.get_local_time() + std::chrono::seconds {add_seconds};
	}

	/**
	 * @brief Convert time point to hours, minutes, seconds tuple
	 * @param add_seconds Additional seconds to add to current time
	 * @param opt_time_point Optional specific time point to use instead of current time
	 * @return Tuple of (hours, minutes, seconds)
	 */
	static std::tuple<int, int, int>
	time_point_hms(long long add_seconds = 0,
				   std::optional<std::chrono::system_clock::time_point> opt_time_point = std::nullopt) {
		using namespace std::chrono;

		// Get the current time
		system_clock::time_point now = system_clock::now();
		system_clock::time_point tp = now + seconds {add_seconds};

		// Override with optional time point if provided
		if (opt_time_point.has_value()) {
			tp = opt_time_point.value();
		}

		// Convert current time and target time to local time zone
		zoned_time now_local {current_zone(), now};
		zoned_time tp_local {current_zone(), tp};

		// Calculate the difference between the two local time points
		auto diff = tp_local.get_local_time() - now_local.get_local_time();

		// Extract hours, minutes, and seconds from the difference
		auto h = duration_cast<hours>(diff);
		diff -= h;
		auto m = duration_cast<minutes>(diff);
		diff -= m;
		auto s = duration_cast<seconds>(diff);

		return std::make_tuple(h.count(), m.count(), static_cast<int>(s.count()));
	}

	/**
	 * @brief Format hours, minutes, seconds as human-readable string
	 * @param h Hours
	 * @param m Minutes  
	 * @param s Seconds
	 * @return Formatted string like "2 hours, 30 minutes and 15 seconds"
	 */
	static std::string format_hms(long h, long m, long s) {
		std::stringstream ss;
		if (h > 0) {
			ss << h << " hour" << (h > 1 ? "s" : "");
			if (m > 0 || s > 0) {
				ss << ", ";
			}
		}
		if (m > 0) {
			ss << m << " minute" << (m > 1 ? "s" : "");
			if (s > 0) {
				ss << " and ";
			}
		}
		if (s > 0) {
			ss << s << " second" << (s > 1 ? "s" : "");
		}
		if (h == 0 && m == 0 && s == 0) {
			ss << "less than a second.";
		}
		return ss.str();
	}

	/**
	 * @brief Format time as datetime string with format "at HH:MM on YYYY/MM/DD"
	 * @param time_in_seconds Time in seconds from epoch
	 * @return Formatted datetime string
	 */
	static std::string format_datetime(long long time_in_seconds) {
		return std::format(" (at {:%H:%M on %Y/%m/%d})", Timer::get_local_time(time_in_seconds));
	}

private:
	std::chrono::system_clock::time_point rt0;
};
