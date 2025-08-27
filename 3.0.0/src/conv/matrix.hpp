/*******************************************************************************
 *	Matrix container templates for CONV 3.0
 *  Copyright M.H.J. Lam, 2025.
 ****/

#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace conv
{

/**
 * @brief 2D matrix template with RAII and bounds checking
 */
template<typename T>
class Matrix2D
{
public:
	Matrix2D() = default;

	Matrix2D(size_t rows, size_t cols, const T& initial_value = T {}) :
		m_rows(rows), m_cols(cols), m_data(rows * cols, initial_value) {
		if (rows == 0 || cols == 0) {
			throw std::invalid_argument("Matrix dimensions must be positive");
		}
	}

	// Copy constructor
	Matrix2D(const Matrix2D& other) = default;

	// Move constructor
	Matrix2D(Matrix2D&& other) noexcept = default;

	// Assignment operators
	Matrix2D& operator=(const Matrix2D& other) = default;
	Matrix2D& operator=(Matrix2D&& other) noexcept = default;

	// Destructor
	~Matrix2D() = default;

	// Element access with bounds checking
	T& operator()(size_t row, size_t col) {
		check_bounds(row, col);
		return m_data[row * m_cols + col];
	}

	const T& operator()(size_t row, size_t col) const {
		check_bounds(row, col);
		return m_data[row * m_cols + col];
	}

	// Unchecked access for performance-critical code
	T& at_unchecked(size_t row, size_t col) noexcept { return m_data[row * m_cols + col]; }

	const T& at_unchecked(size_t row, size_t col) const noexcept { return m_data[row * m_cols + col]; }

	// Properties
	[[nodiscard]] size_t rows() const noexcept { return m_rows; }
	[[nodiscard]] size_t cols() const noexcept { return m_cols; }
	[[nodiscard]] size_t size() const noexcept { return m_data.size(); }
	[[nodiscard]] bool empty() const noexcept { return m_data.empty(); }

	// Data access
	T* data() noexcept { return m_data.data(); }
	const T* data() const noexcept { return m_data.data(); }

	// Row access
	T* row_data(size_t row) {
		if (row >= m_rows) {
			throw std::out_of_range("Row index out of bounds");
		}
		return m_data.data() + row * m_cols;
	}

	const T* row_data(size_t row) const {
		if (row >= m_rows) {
			throw std::out_of_range("Row index out of bounds");
		}
		return m_data.data() + row * m_cols;
	}

	// Resize
	void resize(size_t rows, size_t cols, const T& value = T {}) {
		if (rows == 0 || cols == 0) {
			throw std::invalid_argument("Matrix dimensions must be positive");
		}
		m_rows = rows;
		m_cols = cols;
		m_data.assign(rows * cols, value);
	}

	// Clear
	void clear() {
		m_rows = 0;
		m_cols = 0;
		m_data.clear();
	}

	// Fill with value
	void fill(const T& value) { std::fill(m_data.begin(), m_data.end(), value); }

private:
	void check_bounds(size_t row, size_t col) const {
		if (row >= m_rows || col >= m_cols) {
			throw std::out_of_range("Matrix index out of bounds");
		}
	}

	size_t m_rows = 0;
	size_t m_cols = 0;
	std::vector<T> m_data;
};

/**
 * @brief 3D matrix template with RAII and bounds checking
 */
template<typename T>
class Matrix3D
{
public:
	Matrix3D() = default;

	Matrix3D(size_t dim1, size_t dim2, size_t dim3, const T& initial_value = T {}) :
		m_dim1(dim1), m_dim2(dim2), m_dim3(dim3), m_data(dim1 * dim2 * dim3, initial_value) {
		if (dim1 == 0 || dim2 == 0 || dim3 == 0) {
			throw std::invalid_argument("Matrix dimensions must be positive");
		}
	}

	// Copy constructor
	Matrix3D(const Matrix3D& other) = default;

	// Move constructor
	Matrix3D(Matrix3D&& other) noexcept = default;

	// Assignment operators
	Matrix3D& operator=(const Matrix3D& other) = default;
	Matrix3D& operator=(Matrix3D&& other) noexcept = default;

	// Destructor
	~Matrix3D() = default;

	// Element access with bounds checking
	T& operator()(size_t i, size_t j, size_t k) {
		check_bounds(i, j, k);
		return m_data[(i * m_dim2 + j) * m_dim3 + k];
	}

	const T& operator()(size_t i, size_t j, size_t k) const {
		check_bounds(i, j, k);
		return m_data[(i * m_dim2 + j) * m_dim3 + k];
	}

	// Unchecked access for performance-critical code
	T& at_unchecked(size_t i, size_t j, size_t k) noexcept { return m_data[(i * m_dim2 + j) * m_dim3 + k]; }

	const T& at_unchecked(size_t i, size_t j, size_t k) const noexcept { return m_data[(i * m_dim2 + j) * m_dim3 + k]; }

	// Properties
	[[nodiscard]] size_t dim1() const noexcept { return m_dim1; }
	[[nodiscard]] size_t dim2() const noexcept { return m_dim2; }
	[[nodiscard]] size_t dim3() const noexcept { return m_dim3; }
	[[nodiscard]] size_t size() const noexcept { return m_data.size(); }
	[[nodiscard]] bool empty() const noexcept { return m_data.empty(); }

	// Data access
	T* data() noexcept { return m_data.data(); }
	const T* data() const noexcept { return m_data.data(); }

	// Resize
	void resize(size_t dim1, size_t dim2, size_t dim3, const T& value = T {}) {
		if (dim1 == 0 || dim2 == 0 || dim3 == 0) {
			throw std::invalid_argument("Matrix dimensions must be positive");
		}
		m_dim1 = dim1;
		m_dim2 = dim2;
		m_dim3 = dim3;
		m_data.assign(dim1 * dim2 * dim3, value);
	}

	// Clear
	void clear() {
		m_dim1 = 0;
		m_dim2 = 0;
		m_dim3 = 0;
		m_data.clear();
	}

	// Fill with value
	void fill(const T& value) { std::fill(m_data.begin(), m_data.end(), value); }

private:
	void check_bounds(size_t i, size_t j, size_t k) const {
		if (i >= m_dim1 || j >= m_dim2 || k >= m_dim3) {
			throw std::out_of_range("Matrix index out of bounds");
		}
	}

	size_t m_dim1 = 0;
	size_t m_dim2 = 0;
	size_t m_dim3 = 0;
	std::vector<T> m_data;
};

} // namespace conv
