# AVL Tree Implementation

## Overview

This repository contains a Python implementation of an AVL (Adelson-Velsky and Landis) tree, a self-balancing binary search tree. The code provides various AVL tree operations, including rotations, size and height updates, node deletion, and tree joining.

## Code Structure

The code is organized into a class `AVLTree` with several methods for AVL tree operations. Each method is accompanied by comments, providing insights into the functionality and complexity analysis.

### Key Methods

- **Insertion:** The `insert` method adds a new node to the AVL tree while maintaining balance.

- **Deletion:** The `delete` method removes a node from the AVL tree, and it handles cases where a node has zero, one, or two children.

- **Predecessor and Successor:** The `predecessor` and `successor` methods find the predecessor and successor of a given node, respectively.

- **Rotations:** The `rotateRight` and `rotateLeft` methods perform right and left rotations, respectively. The `rotateAndFix` method checks for imbalances and performs the necessary rotations to restore balance.

- **Joining Trees:** The `join` method merges two AVL trees with a connecting node.

## Comments and Documentation

The code includes comments explaining each method's purpose. This aids in understanding the implementation.

## Complexity Analysis

The code provides complexity analysis for some methods, helping to assess the efficiency of the implementation.

## Usage

To use the AVL tree implementation, create an instance of the `AVLTree` class and utilize its methods for insertion, deletion, and other operations.

```python
# Example Usage
avl_tree = AVLTree()
avl_tree.insert(10)
avl_tree.insert(5)
avl_tree.delete(5)
# Continue with other operations...
```
