/**
 * @file result.hpp
 * @brief Result type for consistent error handling
 * @author M.H.J. Lam
 * @date 2025
 */

#pragma once

#include <variant>
#include <string>
#include <stdexcept>
#include <concepts>

namespace mcml {

/**
 * @brief Error information with context
 */
struct Error {
    std::string message;    ///< Error description
    std::string context;    ///< Additional context (file, function, etc.)
    int code{0};           ///< Error code (optional)
    
    Error() = default;
    explicit Error(std::string msg, std::string ctx = {}, int c = 0) 
        : message(std::move(msg)), context(std::move(ctx)), code(c) {}
    
    /**
     * @brief Get formatted error message
     */
    std::string what() const {
        if (context.empty()) {
            return message;
        }
        return context + ": " + message;
    }
};

/**
 * @brief Forward declaration and specialization for void
 */
template<typename T>
class Result;

/**
 * @brief Specialization for void type
 */
template<>
class Result<void> {
private:
    std::variant<std::monostate, Error> data_;

public:
    Result() : data_(std::monostate{}) {}
    explicit Result(Error&& error) : data_(std::move(error)) {}
    explicit Result(const Error& error) : data_(error) {}
    
    [[nodiscard]] bool has_value() const noexcept {
        return std::holds_alternative<std::monostate>(data_);
    }
    
    [[nodiscard]] bool has_error() const noexcept {
        return std::holds_alternative<Error>(data_);
    }
    
    void value() const {
        if (has_error()) {
            throw std::runtime_error(error().what());
        }
    }
    
    [[nodiscard]] const Error& error() const& {
        if (has_value()) {
            throw std::runtime_error("Called error() on successful result");
        }
        return std::get<Error>(data_);
    }
    
    explicit operator bool() const noexcept {
        return has_value();
    }
};

/**
 * @brief Result type similar to std::expected (C++23)
 * @tparam T Success value type
 */
template<typename T>
class Result {
private:
    std::variant<T, Error> data_;

public:
    /**
     * @brief Construct successful result
     */
    template<typename U = T>
    requires (!std::same_as<U, void>)
    explicit Result(U&& value) : data_(std::forward<U>(value)) {}
    
    template<typename U = T>
    requires (!std::same_as<U, void>)
    explicit Result(const U& value) : data_(value) {}
    
    /**
     * @brief Construct error result
     */
    explicit Result(Error&& error) : data_(std::move(error)) {}
    explicit Result(const Error& error) : data_(error) {}
    
    /**
     * @brief Check if result is successful
     */
    [[nodiscard]] bool has_value() const noexcept {
        return std::holds_alternative<T>(data_);
    }
    
    /**
     * @brief Check if result is an error
     */
    [[nodiscard]] bool has_error() const noexcept {
        return std::holds_alternative<Error>(data_);
    }
    
    /**
     * @brief Get success value (throws if error)
     */
    [[nodiscard]] T& value() & {
        if (has_error()) {
            throw std::runtime_error(error().what());
        }
        return std::get<T>(data_);
    }
    
    /**
     * @brief Get success value (throws if error)
     */
    [[nodiscard]] const T& value() const& {
        if (has_error()) {
            throw std::runtime_error(error().what());
        }
        return std::get<T>(data_);
    }
    
    /**
     * @brief Get success value (throws if error)
     */
    [[nodiscard]] T&& value() && {
        if (has_error()) {
            throw std::runtime_error(error().what());
        }
        return std::get<T>(std::move(data_));
    }
    
    /**
     * @brief Get error (throws if success)
     */
    [[nodiscard]] const Error& error() const& {
        if (has_value()) {
            throw std::runtime_error("Called error() on successful result");
        }
        return std::get<Error>(data_);
    }
    
    /**
     * @brief Get value or default
     */
    template<typename U>
    [[nodiscard]] T value_or(U&& default_value) const& {
        return has_value() ? value() : static_cast<T>(std::forward<U>(default_value));
    }
    
    /**
     * @brief Get value or default
     */
    template<typename U>
    [[nodiscard]] T value_or(U&& default_value) && {
        return has_value() ? std::move(value()) : static_cast<T>(std::forward<U>(default_value));
    }
    
    /**
     * @brief Conversion operators for convenience
     */
    explicit operator bool() const noexcept {
        return has_value();
    }
    
    /**
     * @brief Transform success value if present
     */
    template<typename F>
    auto transform(F&& func) -> Result<std::invoke_result_t<F, T>> {
        using U = std::invoke_result_t<F, T>;
        
        if (has_error()) {
            if constexpr (std::same_as<U, void>) {
                return Result<void>(error());
            } else {
                return Result<U>(error());
            }
        }
        
        if constexpr (std::same_as<U, void>) {
            func(value());
            return Result<void>{};
        } else {
            return Result<U>(func(value()));
        }
    }
    
    /**
     * @brief Chain operations that return Results
     */
    template<typename F>
    auto and_then(F&& func) -> std::invoke_result_t<F, T> {
        if (has_error()) {
            using RetType = std::invoke_result_t<F, T>;
            return RetType(error());
        }
        return func(value());
    }
};

/**
 * @brief Helper functions for creating Results
 */
template<typename T>
[[nodiscard]] Result<std::decay_t<T>> Ok(T&& value) {
    return Result<std::decay_t<T>>(std::forward<T>(value));
}

[[nodiscard]] inline Result<void> Ok() {
    return Result<void>{};
}

[[nodiscard]] inline Error Err(std::string message, std::string context = {}, int code = 0) {
    return Error(std::move(message), std::move(context), code);
}

} // namespace mcml
