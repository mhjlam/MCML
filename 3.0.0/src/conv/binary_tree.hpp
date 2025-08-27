/*******************************************************************************
 *	Binary tree for caching convolution computations in CONV 3.0
 *  Copyright M.H.J. Lam, 2025.
 ****/

#pragma once

#include <memory>
#include <optional>
#include <utility>

namespace conv
{

/**
 * @brief Node in the binary search tree for caching intermediate computations
 * Used to store evaluations of parts of the integrand during convolution
 */
template<typename Key, typename Value>
struct TreeNode {
	Key key;
	Value value;
	std::unique_ptr<TreeNode> left;
	std::unique_ptr<TreeNode> right;

	TreeNode(Key k, Value v) : key(std::move(k)), value(std::move(v)) {}
};

/**
 * @brief Binary search tree for efficient caching of computation results
 * Template class that can cache different types of intermediate results
 */
template<typename Key, typename Value>
class BinaryTree
{
public:
	using NodeType = TreeNode<Key, Value>;
	using NodePtr = std::unique_ptr<NodeType>;

	BinaryTree() = default;
	~BinaryTree() = default;

	// Non-copyable but movable
	BinaryTree(const BinaryTree&) = delete;
	BinaryTree& operator=(const BinaryTree&) = delete;
	BinaryTree(BinaryTree&&) = default;
	BinaryTree& operator=(BinaryTree&&) = default;

	/**
	 * @brief Insert or update a key-value pair
	 * @param key The key to store
	 * @param value The value associated with the key
	 */
	void insert(const Key& key, const Value& value) {
		m_root = insert_recursive(std::move(m_root), key, value);
		++m_size;
	}

	/**
	 * @brief Find a value by key
	 * @param key The key to search for
	 * @return Pointer to value if found, nullptr otherwise
	 */
	[[nodiscard]] const Value* find(const Key& key) const {
		const NodeType* node = find_recursive(m_root.get(), key);
		return node ? &node->value : nullptr;
	}

	/**
	 * @brief Check if tree contains a key
	 * @param key The key to search for
	 * @return true if key exists
	 */
	[[nodiscard]] bool contains(const Key& key) const { return find(key) != nullptr; }

	/**
	 * @brief Clear all nodes from the tree
	 */
	void clear() {
		m_root.reset();
		m_size = 0;
	}

	/**
	 * @brief Get the number of nodes in the tree
	 * @return Number of nodes
	 */
	[[nodiscard]] size_t size() const noexcept { return m_size; }

	/**
	 * @brief Check if tree is empty
	 * @return true if empty
	 */
	[[nodiscard]] bool empty() const noexcept { return m_root == nullptr; }

	/**
	 * @brief Get memory usage estimate in bytes
	 * @return Approximate memory usage
	 */
	[[nodiscard]] size_t memory_usage() const noexcept {
		return m_size * (sizeof(NodeType) + sizeof(Key) + sizeof(Value));
	}

private:
	/**
	 * @brief Recursive insertion helper
	 * @param node Current node
	 * @param key Key to insert
	 * @param value Value to insert
	 * @return Updated node
	 */
	NodePtr insert_recursive(NodePtr node, const Key& key, const Value& value) {
		if (!node) {
			return std::make_unique<NodeType>(key, value);
		}

		if (key < node->key) {
			node->left = insert_recursive(std::move(node->left), key, value);
		}
		else if (key > node->key) {
			node->right = insert_recursive(std::move(node->right), key, value);
		}
		else {
			// Key already exists, update value
			node->value = value;
			--m_size; // Compensate for increment in insert()
		}

		return node;
	}

	/**
	 * @brief Recursive find helper
	 * @param node Current node
	 * @param key Key to find
	 * @return Pointer to node if found, nullptr otherwise
	 */
	[[nodiscard]] const NodeType* find_recursive(const NodeType* node, const Key& key) const {
		if (!node) {
			return nullptr;
		}

		if (key < node->key) {
			return find_recursive(node->left.get(), key);
		}
		else if (key > node->key) {
			return find_recursive(node->right.get(), key);
		}
		else {
			return node;
		}
	}

	NodePtr m_root;
	size_t m_size = 0;
};

/**
 * @brief Specialization for caching floating-point computations with tolerance
 * Useful for caching results of expensive mathematical functions
 */
class FloatCache
{
public:
	FloatCache(double tolerance = 1e-10) : m_tolerance(tolerance) {}

	/**
	 * @brief Insert a computation result
	 * @param key Input parameter
	 * @param value Computed result
	 */
	void insert(double key, double value) { m_tree.insert(discretize_key(key), value); }

	/**
	 * @brief Find a cached result within tolerance
	 * @param key Input parameter
	 * @return Cached result if found within tolerance
	 */
	[[nodiscard]] std::optional<double> find(double key) const {
		const double* result = m_tree.find(discretize_key(key));
		return result ? std::optional<double>(*result) : std::nullopt;
	}

	/**
	 * @brief Clear all cached results
	 */
	void clear() { m_tree.clear(); }

	/**
	 * @brief Get cache statistics
	 */
	[[nodiscard]] size_t size() const noexcept { return m_tree.size(); }
	[[nodiscard]] bool empty() const noexcept { return m_tree.empty(); }
	[[nodiscard]] size_t memory_usage() const noexcept { return m_tree.memory_usage(); }

private:
	/**
	 * @brief Discretize floating-point key for consistent caching
	 * @param key Original key
	 * @return Discretized key
	 */
	[[nodiscard]] long long discretize_key(double key) const { return static_cast<long long>(key / m_tolerance); }

	BinaryTree<long long, double> m_tree;
	double m_tolerance;
};

} // namespace conv
