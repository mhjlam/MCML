#pragma once

#include <chrono>
#include <format>
#include <string>
#include <optional>


class Timer
{
public:
    Timer() : rt0(std::chrono::system_clock::now()) {}

    void reset()
    {
        rt0 = std::chrono::system_clock::now();
    }

    long long punch() const
    {
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = now - rt0;
        return std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    }

    std::string hms_str(long long add_seconds = 0) const
    {
        auto seconds = punch() + add_seconds;
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::seconds(seconds));
        return std::format("{:%H:%M:%S}", duration);
    }

    static auto get_local_time(long long add_seconds = 0)
    {
        std::chrono::zoned_time local_time{ std::chrono::current_zone(), std::chrono::system_clock::now() };
        return local_time.get_local_time() + std::chrono::seconds{ add_seconds };
    }

    static std::tuple<int, int, int> time_point_hms(long long add_seconds = 0, std::optional<std::chrono::system_clock::time_point> opt_time_point = std::nullopt)
    {
        using namespace std::chrono;

        // Get the current time
        system_clock::time_point now = system_clock::now();
        system_clock::time_point tp = now + seconds{ add_seconds };

        // Override with optional time point if provided
        if (opt_time_point.has_value()) {
            tp = opt_time_point.value();
        }

        // Convert current time and target time to local time zone
        zoned_time now_local{ current_zone(), now };
        zoned_time tp_local{ current_zone(), tp };

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

    static std::string format_hms(long h, long m, long s)
    {
        std::stringstream ss;
        if (h > 0) {
            ss << h << " hour" << (h > 1 ? "s" : "");
            if (m > 0 || s > 0) { ss << ", "; }
        }
        if (m > 0) {
            ss << m << " minute" << (m > 1 ? "s" : "");
            if (s > 0) { ss << " and "; }
        }
        if (s > 0) { 
            ss << s << " second" << (s > 1 ? "s" : "");
        }
        if (h == 0 && m == 0 && s == 0) { 
            ss << "less than a second.";
        }
        return ss.str();
    }

    static std::string format_datetime(long long time_in_seconds)
    {
        return std::format(" (at {:%H:%M on %Y/%m/%d})", Timer::get_local_time(time_in_seconds));
    }

private:
    std::chrono::system_clock::time_point rt0;
};
