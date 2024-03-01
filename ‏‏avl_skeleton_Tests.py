"""
	Tests file

"""
import random

"""A class represnting a node in an AVL tree"""


class AVLNode(object):
	"""Constructor, you are allowed to add more fields.

    @type value: str
    @param value: data of your node
    """

	def __init__(self, value):
		self.value = value
		self.left = None
		self.right = None
		self.parent = None
		self.height = -1
		self.size = 0  # New Field- For Ranked Tree

	"""returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child
    """

	def getLeft(self):
		return self.left

	"""returns the right child

    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    """

	def getRight(self):
		return self.right

	"""returns the parent 

    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

	def getParent(self):
		return self.parent

	"""return the value

    @rtype: str
    @returns: the value of self, None if the node is virtual
    """

	def getValue(self):
		return self.value

	"""returns the height

    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

	def getHeight(self):
		return self.height

	"""sets left child

    @type node: AVLNode
    @param node: a node
    """

	def setLeft(self, node):
		self.left = node

	"""sets right child

    @type node: AVLNode
    @param node: a node
    """

	def setRight(self, node):
		self.right = node

	"""sets parent

    @type node: AVLNode
    @param node: a node
    """

	def setParent(self, node):
		self.parent = node

	"""sets value

    @type value: str
    @param value: data
    """

	def setValue(self, value):
		self.value = value

	"""sets the balance factor of the node

    @type h: int
    @param h: the height
    """

	def setHeight(self, h):
		self.height = h

	"""returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherise.
    """

	def isRealNode(self):
		return self.height != -1

	"""New Set Function - sets the size of the sub-tree of the node

    @type s: int
    @param s: the size
    """

	def setSize(self, s):
		self.size = s

	"""New Get Function - returns the size

    @rtype: int
    @returns: the size of self, 0 if the node is virtual
    """

	def getSize(self):
		return self.size

	"""
    A class implementing the ADT list, using an AVL tree.
    """


class AVLTreeList(object):
	"""
    Constructor, you are allowed to add more fields.

    """

	def __init__(self):
		virtualNode = AVLNode("None")  # Empty list contains only one virtual node
		self.root = virtualNode
		self.min = virtualNode  # New Field - first node in list
		self.max = virtualNode  # New Field - last node in list

	"""returns whether the list is empty

    @rtype: bool
    @returns: True if the list is empty, False otherwise
    """

	def empty(self):  # Complexity - O(1) - Access to a pointer and comparison
		return self.root.height == -1

	def appendd(self, val):
		return self.insert(self.length() , val)
	"""retrieves the value of the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the the value of the i'th item in the list
    """

	def retrieve(self, i):  # Complexity - O(logn)
		if self.root.size <= i or i < 0:  # Complexity - O(1) - access to a pointer
			return None

		return (self.treeSelect(self.root, i + 1)).value  # Complexity - O(logn)

	"""inserts val at position i in the list

    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

	def insert(self, i, val):  # not finished
		rebalanceOp = 0
		# rightCase = False

		new_node = AVLNode(val)
		new_node.setHeight(0)  # need to build func for update height
		new_node.setSize(1)  # need to build func for update size
		left_son = AVLNode("None")
		right_son = AVLNode("None")
		new_node.setLeft(left_son)
		new_node.setRight(right_son)
		left_son.setParent(new_node)
		right_son.setParent(new_node)

		if self.empty():
			self.root.setLeft(new_node)
			new_node.setParent(self.root)
			self.root = new_node
			self.min = new_node
			self.max = new_node
			return rebalanceOp

		if i == self.length():
			self.max.right = new_node
			new_node.setParent(self.max)
			self.max = new_node
		elif i < self.length():
			current_node_in_i = self.treeSelect(self.root, i + 1)
			if not current_node_in_i.left.isRealNode():
				current_node_in_i.setLeft(new_node)
				new_node.setParent(current_node_in_i)
			else:
				pred_current_node_in_i = self.predeccesor(current_node_in_i)
				pred_current_node_in_i.setRight(new_node)
				new_node.setParent(pred_current_node_in_i)
			if i == 0:
				self.min = new_node
		temp_node = new_node.parent

		while temp_node.isRealNode():
			self.updateSize(temp_node)
			bf_temp_node = self.balanceFactor(temp_node)
			old_height = temp_node.getHeight()
			curr_new_height = max(temp_node.getLeft().getHeight(), temp_node.getRight().getHeight()) + 1
			if abs(bf_temp_node) < 2 and curr_new_height == old_height:
				break
			elif abs(bf_temp_node) < 2:
				temp_node.setHeight(curr_new_height)
				rebalanceOp += 1
				# rightCase = self.isRightChild(temp_node)
				temp_node = temp_node.parent
			elif abs(bf_temp_node) == 2:
				rebalanceOp += self.rotateAndFix(temp_node)
				break

		while temp_node.isRealNode():
			self.updateSize(temp_node)
			temp_node = temp_node.parent

		return rebalanceOp

	"""deletes the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

	def delete(self, i):
		if (i > self.length() - 1) or (i < 0):
			return -1

		rebalanceOp = 0
		is_two_children = False
		node_to_delete = self.treeSelect(self.root, i + 1)
		node_to_delete_height = node_to_delete.getHeight()
		left_child = node_to_delete.getLeft()
		right_child = node_to_delete.getRight()

		# checks if we deleted min
		if i == 0 and self.length() > 1:
			self.min = self.successor(node_to_delete)

		# checks if we deleted max
		if i == self.length() - 1 and self.length() > 1:
			self.max = self.predeccesor(node_to_delete)

		# if node_to_delete is a leaf
		if node_to_delete.getSize() == 1:
			# if the tree has only one node (we deleted the root)
			if node_to_delete == self.root:
				node_to_delete.getParent().setRight(None)
				node_to_delete.getParent().setLeft(None)
				self.root = node_to_delete.getParent()
				self.min = node_to_delete.getParent()
				self.max = node_to_delete.getParent()
				parent_node = node_to_delete.getParent()
			else:
				parent_node = self.deleteLeaf(node_to_delete)

		# if node_to_delete has one child
		elif (left_child.isRealNode() and (not right_child.isRealNode())) or (
				right_child.isRealNode() and (not left_child.isRealNode())):
			parent_node = self.deleteNodeWithOneChild(node_to_delete)
			# if we deleted the root
			if not parent_node.isRealNode():
				if parent_node.getLeft().isRealNode():
					self.root = parent_node.getLeft()
					self.min = parent_node.getLeft()
					self.max = parent_node.getLeft()
				else:
					self.root = parent_node.getRight()
					self.min = parent_node.getRight()
					self.max = parent_node.getRight()

		# if node_to_delete has two children
		else:
			parent_node = self.deleteNodeWithTwoChildren(node_to_delete)
			is_two_children = True
			# if we deleted the root
			if not parent_node.isRealNode():
				if parent_node.getLeft().isRealNode():
					self.root = parent_node.getLeft()
				else:
					self.root = parent_node.getRight()

		while parent_node.isRealNode():
			self.updateSize(parent_node)
			bf_parent_node = self.balanceFactor(parent_node)
			if is_two_children:
				old_height = node_to_delete_height
				is_two_children = False
			else:
				old_height = parent_node.getHeight()
			curr_new_height = max(parent_node.getLeft().getHeight(), parent_node.getRight().getHeight()) + 1
			if abs(bf_parent_node) < 2 and curr_new_height == old_height:
				break
			elif abs(bf_parent_node) < 2:
				parent_node.setHeight(curr_new_height)
				rebalanceOp += 1
				parent_node = parent_node.parent
			elif abs(bf_parent_node) == 2:
				rebalanceOp += self.rotateAndFix(parent_node)
				parent_node = parent_node.parent.parent

		while parent_node.isRealNode():
			self.updateHeight(parent_node)
			self.updateSize(parent_node)
			parent_node = parent_node.parent

		return rebalanceOp

	"""returns the value of the first item in the list

    @rtype: str
    @returns: the value of the first item, None if the list is empty
    """

	def first(self):  # Complexity - O(1)
		if self.empty():  # Complexity - O(1)
			return None
		else:
			return self.min.value  # Complexity - O(1) - Access to a pointer

	"""returns the value of the last item in the list

    @rtype: str
    @returns: the value of the last item, None if the list is empty
    """

	def last(self):  # Complexity - O(1)
		if self.empty():  # Complexity - O(1)
			return None
		else:
			return self.max.value  # Complexity - O(1) - Access to a pointer

	"""returns an array representing list 

    @rtype: list
    @returns: a list of strings representing the data structure
    """

	def listToArray(self):
		res_array = []
		if self.empty():
			return res_array
		curr_node = self.min
		while curr_node != self.max:
			res_array.append(curr_node.value)
			curr_node = self.successor(curr_node)
		res_array.append(curr_node.value)
		return res_array

	"""returns the size of the list 

    @rtype: int
    @returns: the size of the list
    """

	def length(self):  # Complexity - O(1) - Access to a pointer
		return self.root.size  # if list is empty - root size is 0

	"""splits the list at the i'th index

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list according to whom we split
    @rtype: list
    @returns: a list [left, val, right], where left is an AVLTreeList representing the list until index i-1,
    right is an AVLTreeList representing the list from index i+1, and val is the value at the i'th index.
    """

	def split(self, i):
		tree1 = AVLTreeList()
		tree2 = AVLTreeList()

		if i == 0:
			split_node_val = self.min.value
			self.delete(i)
			return [tree1, split_node_val, self]

		elif i == self.length() - 1:
			split_node_val = self.max.value
			self.delete(i)
			return [self, split_node_val, tree2]

		else:
			split_node = self.treeSelect(self.getRoot(), i + 1)
			split_node_val = split_node.value
			min_tree1 = self.min
			max_tree1 = self.predeccesor(split_node)
			min_tree2 = self.successor(split_node)
			max_tree2 = self.max

			# if split_node has left child
			if split_node.getLeft().isRealNode():
				tree1.getRoot().setLeft(split_node.getLeft())
				split_node.getLeft().setParent(tree1.getRoot())
				tree1.root = split_node.getLeft()

			# if split_node has right child
			if split_node.getRight().isRealNode():
				tree2.getRoot().setLeft(split_node.getRight())
				split_node.getRight().setParent(tree2.getRoot())
				tree2.root = split_node.getRight()

			# disconnect split_node from his children and his parent after saving his parent
			split_node.setRight(None)
			split_node.setLeft(None)
			is_right_child = self.isRightChild(split_node)
			curr_node = split_node.getParent()
			split_node.setParent(None)

			while curr_node.isRealNode():
				curr_parent = curr_node.getParent()
				next_is_right_child = self.isRightChild(curr_node)
				if is_right_child:
					help_tree1 = AVLTreeList()
					if curr_node.getLeft().isRealNode():
						help_tree1.getRoot().setLeft(curr_node.getLeft())
						curr_node.getLeft().setParent(help_tree1.getRoot())
						help_tree1.root = curr_node.getLeft()
					else:
						curr_node.getLeft().setParent(None)
					help_tree1.join(curr_node, tree1)
					tree1.root = help_tree1.root

				else:
					help_tree2 = AVLTreeList()
					if curr_node.getRight().isRealNode():
						help_tree2.getRoot().setLeft(curr_node.getRight())
						curr_node.getRight().setParent(help_tree2.getRoot())
						help_tree2.root = curr_node.getRight()
					else:
						curr_node.getRight().setParent(None)
					tree2.join(curr_node, help_tree2)

				is_right_child = next_is_right_child
				curr_node = curr_parent

			curr_node.setRight(None)
			curr_node.setLeft(None)
			tree1.min = min_tree1
			tree1.max = max_tree1
			tree2.min = min_tree2
			tree2.max = max_tree2

		return [tree1, split_node_val, tree2]

	"""concatenates lst to self

    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

	def concat(self, lst):
		new_min = self.min
		height_diff = abs(self.getRoot().getHeight() - lst.getRoot().getHeight())

		if self.empty():
			self.root = lst.root
			self.min = lst.min
			self.max = lst.max
			return height_diff

		if lst.empty():
			return height_diff

		new_max = lst.max
		max_node = self.max
		self.delete(self.length() - 1)
		self.join(max_node, lst)
		self.max = new_max
		self.min = new_min
		return height_diff

	"""searches for a *value* in the list

    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    """

	def search(self,
			   val):  # Complexity - O(n) - we go through each edge twice at most (based on recitation - proof at documentary file)
		curr_node = self.min
		index = 0
		while curr_node != self.max:
			if curr_node.value == val:
				return index
			curr_node = self.successor(curr_node)
			index += 1

		if curr_node.value == val:
			return index
		else:
			return -1

	"""returns the root of the tree representing the list

    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """

	def getRoot(self):
		return self.root

	"""Auxiliary Function - returns the node with rank k (the node at index k-1)

    @rtype: AVLNode
    @returns: the node with rank k
    @precondition:  1 <= k <= self.length
    """

	def treeSelect(self, curr_node, k):  # Complexity - O(logn) - linear in the height of the tree
		root_and_left_size = curr_node.left.size + 1
		if k == root_and_left_size:
			return curr_node
		elif k < root_and_left_size:
			return self.treeSelect(curr_node.left, k)
		else:
			return self.treeSelect(curr_node.right, k - root_and_left_size)

	"""Auxiliary Function - returns the node with rank(node) + 1 (the node at the next index)

        @rtype: AVLNode
        @returns: the node at the next index 

        """

	def successor(self, node):  # Complexity - O(logn) - linear in the height of the tree
		curr_node = node.right
		#  if the node has a right child
		if curr_node.isRealNode():
			if not curr_node.left.isRealNode():
				return curr_node
			else:
				curr_node = curr_node.left
				while curr_node.left.isRealNode():
					curr_node = curr_node.left

				return curr_node

		# if the node does not have a right child
		else:
			while self.isRightChild(curr_node):
				curr_node = curr_node.parent

			return curr_node.parent

	"""Auxiliary Function - returns the node with rank(node) - 1 (the node at the previous index)

            @rtype: AVLNode
            @returns: the node at the previous index 

            """

	def predeccesor(self, node):  # Complexity - O(logn) - linear in the height of the tree
		curr_node = node.left
		#  if the node has a left child
		if curr_node.isRealNode():
			if not curr_node.right.isRealNode():
				return curr_node
			else:
				curr_node = curr_node.right
				while curr_node.right.isRealNode():
					curr_node = curr_node.right

				return curr_node

		# if the node does not have a left child
		else:
			while not self.isRightChild(curr_node):
				curr_node = curr_node.parent

			return curr_node.parent

	"""Auxiliary Function - general rotate - checks which rotation needs to be made and calls it

                        @rtype: integer
                        @returns: number of rotations 

                        """

	def rotateAndFix(self, node):  # Complexity
		node_bf = self.balanceFactor(node)

		if node_bf == 2:
			left_child_bf = self.balanceFactor(node.getLeft())

			# Right Rotation
			if left_child_bf == 1 or left_child_bf == 0:
				self.rotateRight(node)
				return 1

			# Left Then Right Rotation
			else:
				self.rotateLeft(node.getLeft())
				self.rotateRight(node)
				return 2

		else:
			right_child_bf = self.balanceFactor(node.getRight())

			# Left Rotation
			if right_child_bf == -1 or right_child_bf == 0:
				self.rotateLeft(node)
				return 1

			# Right Then Left Rotation
			else:
				self.rotateRight(node.getRight())
				self.rotateLeft(node)
				return 2

	"""Auxiliary Function - rotates right

      """

	def rotateRight(self, parent):  # Complexity
		isParentRightChild = self.isRightChild(parent)
		child = parent.getLeft()
		parent.setLeft(child.getRight())
		parent.getLeft().setParent(parent)
		child.setRight(parent)
		child.setParent(parent.getParent())
		if isParentRightChild:
			child.getParent().setRight(child)
		else:
			child.getParent().setLeft(child)
		parent.setParent(child)

		if self.root == parent:
			self.root = child

		self.updateSize(parent)
		self.updateHeight(parent)
		self.updateSize(child)
		self.updateHeight(child)

	"""Auxiliary Function - rotates left

    """

	def rotateLeft(self, parent):  # Complexity
		isParentRightChild = self.isRightChild(parent)
		child = parent.getRight()
		parent.setRight(child.getLeft())
		parent.getRight().setParent(parent)
		child.setLeft(parent)
		child.setParent(parent.getParent())
		if isParentRightChild:
			child.getParent().setRight(child)
		else:
			child.getParent().setLeft(child)
		parent.setParent(child)

		if self.root == parent:
			self.root = child

		self.updateSize(parent)
		self.updateHeight(parent)
		self.updateSize(child)
		self.updateHeight(child)

	"""Auxiliary Function - returns true if the node is a right child of its parent

            @rtype: boolean value
            @returns: true if the node is a right child of its parent

            """

	def isRightChild(self, node):  # Complexity - O(1) - access to a pointer
		parent = node.parent
		return parent.right == node

	"""Auxiliary Function - calculates BF of a node

                    @rtype: integer
                    @returns: node's BF

                    """

	def balanceFactor(self, node):
		return node.getLeft().getHeight() - node.getRight().getHeight()

	"""Auxiliary Function - updates the size of a node

        """

	def updateSize(self, node):
		node.setSize(node.getLeft().getSize() + node.getRight().getSize() + 1)

	"""Auxiliary Function - updates the height of a node

        """

	def updateHeight(self, node):
		node.setHeight(max(node.getLeft().getHeight(), node.getRight().getHeight()) + 1)

	"""Auxiliary Function - deletes the leaf it gets

                @rtype: AVLNode
                @returns: the AVLNode from which the node has physically changed

            """

	def deleteLeaf(self, node_to_delete):  # Complexity
		parent_res = node_to_delete.getParent()
		if self.isRightChild(node_to_delete):
			node_to_delete.getParent().setRight(node_to_delete.getRight())
			node_to_delete.getRight().setParent(node_to_delete.getParent())
		else:
			node_to_delete.getParent().setLeft(node_to_delete.getLeft())
			node_to_delete.getLeft().setParent(node_to_delete.getParent())

		node_to_delete.setParent(None)
		node_to_delete.setRight(None)
		node_to_delete.setLeft(None)
		return parent_res

	"""Auxiliary Function - deletes the node it gets

                @rtype: AVLNode
                @returns: the AVLNode from which the node has physically changed

            """

	def deleteNodeWithOneChild(self, node_to_delete):  # Complexity
		# if node_to_delete has only right child
		if node_to_delete.getRight().isRealNode() and (not node_to_delete.getLeft().isRealNode()):
			if self.isRightChild(node_to_delete):
				node_to_delete.getParent().setRight(node_to_delete.getRight())
			else:
				node_to_delete.getParent().setLeft(node_to_delete.getRight())
			node_to_delete.getRight().setParent(node_to_delete.getParent())

		# if node_to_delete has only left child
		elif (not node_to_delete.getRight().isRealNode()) and node_to_delete.getLeft().isRealNode():
			if self.isRightChild(node_to_delete):
				node_to_delete.getParent().setRight(node_to_delete.getLeft())
			else:
				node_to_delete.getParent().setLeft(node_to_delete.getLeft())
			node_to_delete.getLeft().setParent(node_to_delete.getParent())

		return node_to_delete.getParent()

	"""Auxiliary Function - deletes the node it gets

                    @rtype: AVLNode
                    @returns: the AVLNode from which the node has physically changed

                """

	def deleteNodeWithTwoChildren(self, node_to_delete):  # Complexity
		successor_to_delete = self.successor(node_to_delete)
		if successor_to_delete.getSize() == 1:
			parent_of_successor = self.deleteLeaf(successor_to_delete)

		else:
			parent_of_successor = self.deleteNodeWithOneChild(successor_to_delete)

		if self.isRightChild(node_to_delete):
			node_to_delete.getParent().setRight(successor_to_delete)
		else:
			node_to_delete.getParent().setLeft(successor_to_delete)
		successor_to_delete.setParent(node_to_delete.getParent())

		successor_to_delete.setRight(node_to_delete.getRight())
		node_to_delete.getRight().setParent(successor_to_delete)
		successor_to_delete.setLeft(node_to_delete.getLeft())
		node_to_delete.getLeft().setParent(successor_to_delete)

		if parent_of_successor == node_to_delete:
			parent_of_successor = successor_to_delete

		if node_to_delete == self.getRoot():
			self.root = successor_to_delete

		node_to_delete.setParent(None)
		node_to_delete.setRight(None)
		node_to_delete.setLeft(None)
		return parent_of_successor

	"""Auxiliary Function - joins two trees with a connecting node 
                        @rtype: AVLTree
                        @returns: the AVLTree after joining two trees

                    """

	def join(self, connecting_node, tree):  # Complexity
		tree_is_bigger = False
		# if two trees have the same height
		if self.getRoot().getHeight() == tree.getRoot().getHeight():
			connecting_node.setLeft(self.getRoot())
			connecting_node.setRight(tree.getRoot())
			if not (self.empty() and tree.empty()):
				connecting_node.setParent(self.root.getParent())
				if self.isRightChild(self.root):
					self.root.getParent().setRight(connecting_node)
				else:
					self.root.getParent().setLeft(connecting_node)
			else:
				virtual = AVLNode("None")
				connecting_node.setParent(virtual)
				virtual.setLeft(connecting_node)
			self.getRoot().setParent(connecting_node)
			tree.getRoot().setParent(connecting_node)

			self.root = connecting_node
			self.updateSize(self.getRoot())
			self.updateHeight(self.getRoot())

		else:
			if self.getRoot().getHeight() > tree.getRoot().getHeight():
				lower_tree_height = tree.getRoot().getHeight()

				# linking connecting_node to shorter tree
				connecting_node.setRight(tree.getRoot())
				if not tree.empty():
					"""
                    if tree.isRightChild(tree.getRoot()):
                        tree.getRoot().getParent().setRight(None)
                    else:
                        tree.getRoot().getParent().setLeft(None)
                    """
				tree.getRoot().setParent(connecting_node)

				# finding the left child of connecting_node
				curr_node = self.getRoot()
				while curr_node.getHeight() > lower_tree_height:
					curr_node = curr_node.getRight()

				# linking connecting_node to higher tree
				connecting_node.setParent(curr_node.getParent())
				curr_node.getParent().setRight(connecting_node)
				connecting_node.setLeft(curr_node)
				curr_node.setParent(connecting_node)

			else:
				lower_tree_height = self.getRoot().getHeight()

				tree_is_bigger = True
				# linking connecting_node to shorter tree
				connecting_node.setLeft(self.getRoot())
				if not self.empty():
					"""
                    if self.isRightChild(self.getRoot()):
                        self.getRoot().getParent().setRight(None)
                    else:
                        self.getRoot().getParent().setLeft(None)
                    """
				self.getRoot().setParent(connecting_node)

				# finding the right child of connecting_node
				curr_node = tree.getRoot()
				while curr_node.getHeight() > lower_tree_height:
					curr_node = curr_node.getLeft()

				# linking connecting_node to higher tree
				connecting_node.setParent(curr_node.getParent())
				curr_node.getParent().setLeft(connecting_node)
				connecting_node.setRight(curr_node)
				curr_node.setParent(connecting_node)

			self.updateSize(connecting_node)
			self.updateHeight(connecting_node)

			parent_node = connecting_node.getParent()
			while parent_node.isRealNode():
				self.updateSize(parent_node)
				bf_parent_node = self.balanceFactor(parent_node)
				old_height = parent_node.getHeight()
				curr_new_height = max(parent_node.getLeft().getHeight(), parent_node.getRight().getHeight()) + 1
				if abs(bf_parent_node) < 2 and curr_new_height == old_height:
					break
				elif abs(bf_parent_node) < 2:
					parent_node.setHeight(curr_new_height)
					parent_node = parent_node.parent
				elif abs(bf_parent_node) == 2:
					if tree_is_bigger:
						tree.rotateAndFix(parent_node)
					else:
						self.rotateAndFix(parent_node)
					parent_node = parent_node.parent.parent

			while parent_node.isRealNode():
				self.updateSize(parent_node)
				parent_node = parent_node.parent

			if tree_is_bigger:
				self.root = tree.root

	def getTreeHeight(self):
		return self.root.height

	##################################### inside AVLTreeList class ######################################
	"""Checks if the AVL tree properties are consistent

	@rtype: boolean 
	@returns: True if the AVL tree properties are consistent
	"""

	def check(self):
		if not self.isAVL():
			print("The tree is not an AVL tree!")
		if not self.isSizeConsistent():
			print("The sizes of the tree nodes are inconsistent!")
		if not self.isHeightConsistent():
			print("The heights of the tree nodes are inconsistent!")
		# if not self.isRankConsistent():
			# print("The ranks of the tree nodes are inconsistent!")

	"""Checks if the tree is an AVL

	@rtype: boolean 
	@returns: True if the tree is an AVL tree
	"""

	def isAVL(self):
		return self.isAVLRec(self.getRoot())

	"""Checks if the subtree is an AVL
	@type x: AVLNode
	@param x: The root of the subtree
	@rtype: boolean 
	@returns: True if the subtree is an AVL tree
	"""

	def isAVLRec(self, x):
		# If x is a virtual node return True
		if not x.isRealNode():
			return True
		# Check abs(balance factor) <= 1
		bf = self.balanceFactor(x)
		if bf > 1 or bf < -1:
			return False
		# Recursive calls
		return self.isAVLRec(x.getLeft()) and self.isAVLRec(x.getRight())

	"""Checks if sizes of the nodes in the tree are consistent

	@rtype: boolean 
	@returns: True if sizes of the nodes in the tree are consistent
	"""

	def isSizeConsistent(self):
		return self.isSizeConsistentRec(self.getRoot())

	"""Checks if sizes of the nodes in the subtree are consistent

	@type x: AVLNode
	@param x: The root of the subtree
	@rtype: boolean 
	@returns: True if sizes of the nodes in the subtree are consistent
	"""

	def isSizeConsistentRec(self, x):
		# If x is a virtual node return True
		if not x.isRealNode():
			return True
		# Size of x should be x.left.size + x.right.size + 1
		if x.getSize() != (x.getLeft().getSize() + x.getRight().getSize() + 1):
			return False
		# Recursive calls
		return self.isSizeConsistentRec(x.getLeft()) and self.isSizeConsistentRec(x.getRight())

	"""Checks if heights of the nodes in the tree are consistent

	@rtype: boolean 
	@returns: True if heights of the nodes in the tree are consistent
	"""

	def isHeightConsistent(self):
		return self.isHeightConsistentRec(self.getRoot())

	"""Checks if heights of the nodes in the subtree are consistent

	@type x: AVLNode
	@param x: The root of the subtree
	@rtype: boolean 
	@returns: True if heights of the nodes in the subtree are consistent
	"""

	def isHeightConsistentRec(self, x):
		# If x is a virtual node return True
		if not x.isRealNode():
			return True
		# Height of x should be maximum of children heights + 1
		if x.getHeight() != max(x.getLeft().getHeight(), x.getRight().getHeight()) + 1:
			return False
		# Recursive calls
		return self.isSizeConsistentRec(x.getLeft()) and self.isSizeConsistentRec(x.getRight())

	"""Checks if the ranks of the nodes in the tree are consistent

	@returns: True if the ranks of the nodes in the tree are consistent
	"""

	def isRankConsistent(self):
		root = self.getRoot()
		for i in range(1, root.getSize()):
			if i != self.rank(self.select(i)):
				return False
		nodesList = self.nodes()
		for node in nodesList:
			if node != self.select(self.rank(node)):
				return False
		return True

	"""Returns a list of the nodes in the tree sorted by index in O(n)

	@rtype: list
	@returns: A list of the nodes in the tree sorted by index
	"""

	def nodes(self):
		lst = []
		self.nodesInOrder(self.getRoot(), lst)
		return lst

	"""Adds the nodes in the subtree to the list
	 following an in-order traversal in O(n)

	@type x: AVLNode
	@type lst: list
	@param x: The root of the subtree
	@param lst: The list
	"""

	def nodesInOrder(self, x, lst):
		if not x.isRealNode():
			return
		self.nodesInOrder(x.getLeft(), lst)
		lst.append(x)
		self.nodesInOrder(x.getRight(), lst)


	def printree(self, t, bykey=False):
		return t.trepr(t, t.getRoot(), bykey)

	def trepr(self, t, node, bykey=False):
		"""Return a list of textual representations of the levels in t
        bykey=True: show keys instead of values"""
		if not node.isRealNode():
			# You might want to change this, depending on your implementation
			return ["#"]  # Hashtag marks a virtual node

		thistr = str(node.getValue())
		thistr	= thistr + ", s :" + str(node.getSize())
		thistr = thistr + ", h :" + str(node.getHeight())
		thistr = thistr + ", bf :" + str(t.balanceFactor(node))
		if abs(t.balanceFactor(node)) > 1:
			thistr = thistr + "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"


		return self.conc(self.trepr(t, node.getLeft(), bykey), thistr, self.trepr(t, node.getRight(), bykey))

	def conc(self, left, root, right):
		"""Return a concatenation of textual representations of
        a root node, its left node, and its right node
        root is a string, and left and right are lists of strings"""

		lwid = len(left[-1])
		rwid = len(right[-1])
		rootwid = len(root)

		result = [(lwid + 1) * " " + root + (rwid + 1) * " "]

		ls = self.leftspace(left[0])
		rs = self.rightspace(right[0])
		result.append(ls * " " + (lwid - ls) * "_" + "/" + rootwid * " " + "\\" + rs * "_" + (rwid - rs) * " ")

		for i in range(max(len(left), len(right))):
			row = ""
			if i < len(left):
				row += left[i]
			else:
				row += lwid * " "

			row += (rootwid + 2) * " "

			if i < len(right):
				row += right[i]
			else:
				row += rwid * " "

			result.append(row)

		return result

	def leftspace(self, row):
		"""helper for conc"""
		# row is the first row of a left node
		# returns the index of where the second whitespace starts
		i = len(row) - 1
		while row[i] == " ":
			i -= 1
		return i + 1

	def rightspace(self, row):
		"""helper for conc"""
		# row is the first row of a right node
		# returns the index of where the first whitespace ends
		i = 0
		while row[i] == " ":
			i += 1
		return i




class tests2:
	@staticmethod
	def insertest(tree, size):
		for i in range(size):
			index = random.randint(0, tree.length())
			tree.insert(index, str(index))

	@staticmethod
	def printush(tree):
		arr = tree.printree(tree)
		leni = len(arr)
		for i in range(leni):
			print(arr[i])

	@staticmethod
	def deltest(tree):
		for i in range(25):
			index = random.randint(0, tree.length()-1)
			tree.delete(index)
			tests2.printush(tree)

#tree = AVLTreeList()
#tests2.insertest(tree)
#tests2.printush(tree)

#print(tree.listToArray())
# tests2.deltest(tree)

# tree.delete(15)
# tests2.printush(tree)


class tests:
	def checkBF(t):
		lst = []
		lst2 = []
		if t.empty():
			return []
		tests.checkBFrec(t.root, lst)
		return lst

	def checkBFrec(node, lst):
		if AVLNode.isRealNode(node):
			tests.checkBFrec(node.left, lst)
			bf = node.getLeft().getHeight() - node.getRight().getHeight()
			if abs(bf) > 1:
				lst.append((node.value, bf))
			tests.checkBFrec(node.right, lst)
		return lst

	def checkInsert():
		checkLst = []
		t = AVLTreeList()
		for i in range(1000000):
			index = random.randint(0, len(checkLst))
			t.insert(index, str(i))
			checkLst.insert(index, str(i))
			curTree = t.listToArray()
			if t.first() != curTree[0] or t.last() != curTree[len(curTree) - 1]:
				print("Error in first or last")
				return None
			if curTree != checkLst:
				print("error inorder after insert to index: ", index)
				print("tree after bad insert: ", curTree)
				return None
			if tests.checkBF(t) != []:
				print("error in BF after insert to index: ", index)
				print("tree after bad insert: ", curTree)
				return None
		# print ("final tree afer", i, "insertions:", curTree)
		print("insert- well done")
		return t

	def checkDelete():
		t = tests.checkInsert()
		checkLst = t.listToArray()
		for i in range(1000000):
			index = random.randint(0, len(checkLst) - 1)
			value = t.retrieve(index)
			# tests2.printush(t)
			t.delete(index)
			checkLst.remove(str(value))
			curTree = t.listToArray()
			if not t.empty() and (t.first() != curTree[0] or t.last() != curTree[len(curTree) - 1]):
				print("Error in first or last")
			if curTree != checkLst:
				print("error inorder after delete index: ", index)
				print("tree after bad delete: ", curTree)
				return None
			if tests.checkBF(t) != []:
				print("error in BF after delete index: ", index)
				print("tree after bad delete: ", curTree)
				return None
		# print ("final tree afer", i, "deletions:", curTree)
		print("delete-well done")

	def checkSplitConcat():
		t = tests.checkInsert()
		checkLst = t.listToArray()
		# print("in order: ", checkLst, "pre order: ", t.listToArrayPreOrd() )
		while t.length() > 0:
			index = random.randint(0, t.length() - 1)
			# print("spliting:")
			# print("splitting by: ", index)
			# print(t.listToArray())
			ret = t.split(index)
			# print("tree after spliring:")
			# tests2.printush(ret[0])
			# tests2.printush(ret[2])
			checkLst1 = checkLst[:index]
			checkLst2 = checkLst[index + 1:]

			# print(AVLTreeList.listToArray(ret[0]), AVLTreeList.listToArray(ret[2]))
			# print("t1 min: " , (ret[0].min).value)
			# print("t1 max: ", (ret[0].max).value)
			# print("t2 min: ", (ret[2].min).value)
			# print("t2 max: ", (ret[2].max).value)
			if AVLTreeList.listToArray(ret[0]) != checkLst1 or AVLTreeList.listToArray(ret[2]) != checkLst2:
				print("error inorder after split by index: ", index)
				print("tree after bad split: ", ret[0], ret[2])
				return None
			if tests.checkBF(ret[0]) != [] or tests.checkBF(ret[2]) != []:
				print("error in BF after split by index: ", index)
				print("tree after bad split: ", ret[0], ret[2])
				return None
			# print("concatting:")
			checkLst = checkLst1 + checkLst2
			AVLTreeList.concat(ret[0], ret[2])
			t = ret[0]
			# print("tree after concating:")
			# tests2.printush(t)
			currTree = t.listToArray()
			# print (currTree)
			if currTree != checkLst:
				print("error inorder after concat")
				print("tree after bad concat: ", currTree)
				return None
			if tests.checkBF(t) != []:
				print("error in BF after concating")
				print("tree after bad concat: ", currTree)
				return None
			if not t.empty() and (t.first() != currTree[0] or t.last() != currTree[len(currTree) - 1]):
				print("Error in first or last")
		# print("final tree afer all splits and concats:", currTree)
		print("split and concat- well done")

	def checkListToArray():
		t = tests.checkInsert()
		checkLst = t.listToArray()
		for i in range(len(checkLst)):
			val = t.retrieve(i)
			if val != checkLst[i]:
				print ("oof")
			if checkLst[i] != (t.treeSelect( t.getRoot(),i+1)).value:
				print ("oof2, ", i)
			if t.search(checkLst[i]) != checkLst.index(checkLst[i]):
				print("oof3")
		print("nadir")

# tree = AVLTreeList()
# tests2.insertest(tree, 1000000)

# tests.checkInsert()
# tests.checkDelete()
# tests.checkSplitConcat()
# tests.checkListToArray()

"""
tree = AVLTreeList()
tests2.insertest(tree, 35)
tests2.printush(tree)
print(tree.listToArray())

listaftersplit = tree.split(34)

print("******************************************************************************")
print("****************************tree 1 after split: ***************************")

print(listaftersplit[0].listToArray())


print("******************************************************************************")
print("****************************tree 2 after split: ***************************")

print(listaftersplit[2].listToArray())


print("******************************************************************************")
print("****************************and split node is: ***************************")

print(listaftersplit[1])




print("******************************************************************************")
print("****************************tree1***************************")
tree = AVLTreeList()
tests2.insertest(tree, 20)
print(tree.listToArray())
# tests2.printush(tree)

print("******************************************************************************")
print("****************************tree2***************************")

tree2 = AVLTreeList()
tests2.insertest(tree2, 15)
print(tree2.listToArray())

# tests2.printush(tree2)

tree.concat(tree2)

print("******************************************************************************")
print("****************************tree after concat: ***************************")

print(tree.listToArray())

listaftersplit = tree.split(5)

print("******************************************************************************")
print("****************************tree 1 after split: ***************************")

# print(listaftersplit[0].listToArray())


print("******************************************************************************")
print("****************************tree 2 after split: ***************************")

# print(listaftersplit[2].listToArray())


a1 = tree.listToArray()
print(a1)
print("a1 len:", len(a1))
tests2.printush(tree)
print("a1 first: ", tree.first())
print("a1 last: ",tree.last())
tree2 = AVLTreeList()
tests2.insertest(tree2)
a2 = tree2.listToArray()
print(a2)
print("a2 len:", len(a2))
tests2.printush(tree2)
print("a2 first: ",tree2.first())
print("a2 last: ",tree2.last())




tree.concat(tree2)

# tests2.printush(tree)
a3 = tree.listToArray()
print("a3 len:", len(a3))
tests2.printush(tree)
print("a3 first: ",tree.first())
print("a3 last: ",tree.last())

print ("is equeal: " , a1+a2 == a3)
print(tests.checkBF(tree))

tree.delete(tree.length()-1)
print("a3 first: ",tree.first())
print("a3 last: ",tree.last())

tree.delete(tree.length()-1)
print("a3 first: ",tree.first())
print("a3 last: ",tree.last())

tree.delete(tree.length()-1)
print("a3 first: ",tree.first())
print("a3 last: ",tree.last())

tree.delete(tree.length()-1)
print("a3 first: ",tree.first())
print("a3 last: ",tree.last())

tree.delete(tree.length()-1)
print("a3 first: ",tree.first())
print("a3 last: ",tree.last())


print(tree.treeSelect(tree.root))




"""







# tests.checkInsert()

# tests.checkDelete()

# tests.checkSplitConcat()

# tree = AVLTreeList()
# tests2.insertest(tree)
# tests2.printush(tree)
# tests2.deltest(tree)

#tests2.printush(tree)

#print((tree.successor(tree.treeSelect(tree.getRoot(),25))).value)

#tree.delete(21)
#tests2.printush(tree)

#tree.delete(25)
#tests2.printush(tree)

#tree.delete(0)
#tests2.printush(tree)
"""
tree = AVLTreeList()
tree.insert(0, '0')
tree.insert(1,'1')
tree.insert(2,'2')
tree.insert(3,'3')
tree.insert(4,'4')

printush(tree)


tree.check()
arr = tree.printree(tree)
len = len(arr)
for i in range(len):
	print(arr[i])
def checksBF(node, tree):
	if not node.isRealNode():
		return
	else:
		checksBF(node.getLeft(), tree)
		print("node: " + node.value + " , bf: " + str(tree.balanceFactor(node)))
		checksBF(node.getRight(), tree)


def tests():
	tree = AVLTreeList()
	print(tree.empty())
	tree.insert(0, "1")
	print(tree.empty())
	tree.insert(1, "18")
	tree.insert(2, "3")
	print("search: " + str(tree.search("44")))
	print(tree.root.value)
	print(tree.last())
	print(tree.first())
	a = tree.listToArray()
	print(''.join(a))
	print(tree.retrieve(3))
	print("**********")
	print(tree.insert(3, "14"))
	tree.insert(3, "15")
	tree.insert(2, "44")
	tree.insert(4, "22")
	b = tree.listToArray()
	print(','.join(b))
	checksBF(tree.root , tree)
	a = tree.listToArray()
	print(','.join(a))
	tree.delete(3)
	a = tree.listToArray()
	print(','.join(a))

tests()

"""



import unittest

"""
IN ORDER TO USE THE TEST TOU NEED TO DO THE FOLLOWING:
1. IMPLEMENT APPEND METHOD IN AVLTreeList THIS WAY:
def append(self, val):
        self.insert(self.length(), val)
2. YOU NEED TO HAVE FIELDS POINTING TO FIRST AND LAST NODE IN THE TREE REPRESENTING THE LIST. 
NAME THEM firstItem and lastItem OR ALTERNATIVELY CHANGE THE TEST ITSELF TO FIT WITH THE NAMES
OF THESE FIELDS AT YOUR IMPLEMENTATION. 
3. YOU NEED TO HAVE A METHOD THAT RETURNS THE TREE HEIGHT AND NAME IT 'getTreeHeight'
"""


class testAVLList(unittest.TestCase):

    emptyList = AVLTreeList()
    twentyTree = AVLTreeList()
    twentylist = []

    for i in range(20):
        twentylist.append(i)
        twentyTree.appendd(i)

    def in_order(self, tree, node, func):
        if node is not None:
            if node.isRealNode():
                self.in_order(tree, node.getLeft(), func)
                func(node, tree)
                self.in_order(tree, node.getRight(), func)

    def compare_with_list_by_in_order(self, tree, lst):
        def rec(node, cnt, lst):
            if node.isRealNode():
                rec(node.getLeft(), cnt, lst)
                self.assertEqual(node.getValue(), lst[cnt[0]])
                cnt[0] += 1
                rec(node.getRight(), cnt, lst)

        cnt = [0]
        if not tree.empty():
            rec(tree.getRoot(), cnt, lst)
        else:
            self.assertEqual(len(lst), 0)

    def compare_with_list_by_retrieve(self, tree, lst):
        for i in range(max(len(lst), tree.length())):
            self.assertEqual(tree.retrieve(i), lst[i])

    def test_empty(self):
        self.assertTrue(self.emptyList.empty())
        self.assertFalse(self.twentyTree.empty())

    def test_retrieve_basic(self):
        self.assertIsNone(self.emptyList.retrieve(0))
        self.assertIsNone(self.emptyList.retrieve(59))
        self.assertIsNone(self.twentyTree.retrieve(30))
        self.assertIsNone(self.twentyTree.retrieve(-1))
        for i in range(20):
            self.assertEqual(self.twentylist[i], self.twentyTree.retrieve(i))
        T = AVLTreeList()
        T.appendd('a')
        self.assertEqual(T.retrieve(0), "a")

    def check_first(self, tree, lst):
        if not tree.empty():
            self.assertEqual(tree.first(), lst[0])
        else:
            self.assertEqual(len(lst), 0)
            self.assertIsNone(tree.first())

    def check_last(self, tree, lst):
        if not tree.empty():
            self.assertEqual(tree.last(), lst[-1])
        else:
            self.assertEqual(len(lst), 0)
            self.assertIsNone(tree.last())

    def test_assert_height_difference_non_empty_lists_big(self):
        T1 = AVLTreeList()
        T2 = AVLTreeList()
        for i in range(10):
            T1.appendd(i)
        for i in range(5):
            T2.appendd(i)
        self.assertEqual(abs(T1.getRoot().getHeight() -
                             T2.getRoot().getHeight()), T1.concat(T2))
        #self.assertEqual(
          #  abs(T1.getTreeHeight()-T2.getTreeHeight()), T1.concat(T2))
        T3 = AVLTreeList()
        T4 = AVLTreeList()
        for i in range(10):
            T4.appendd(i)
        for i in range(5):
            T3.appendd(i)
        self.assertEqual(
            abs(T3.getTreeHeight()-T4.getTreeHeight()), T3.concat(T4))
        #self.assertEqual(abs(T3.getRoot().getHeight() -
                  #           T4.getRoot().getHeight()), T3.concat(T4))


    ###TESTING INSERTION###

    def test_insertBasic(self):
        T1 = AVLTreeList()
        T1.insert(0, 1)
        self.assertEqual(T1.getRoot().getValue(), 1)

    def test_insert_at_start(self):
        # inserts at start
        T2 = AVLTreeList()
        L2 = []

        for i in range(50):
            T2.insert(0, i)
            L2.insert(0, i)
            self.compare_with_list_by_in_order(T2, L2)
            self.compare_with_list_by_retrieve(T2, L2)

        self.check_first(T2, L2)
        self.check_last(T2, L2)

    def test_insert_at_end_small(self):
        T1 = AVLTreeList()
        L1 = []

        for i in range(3):
            T1.appendd(i)
            L1.append(i)
            self.compare_with_list_by_in_order(T1, L1)
            self.compare_with_list_by_retrieve(T1, L1)

        for j in range(10, 20):
            T3 = AVLTreeList()
            L3 = []
            for i in range(j):
                T3.insert(T3.length(), i)
                L3.insert(len(L3), i)
                self.compare_with_list_by_retrieve(T3, L3)
                self.compare_with_list_by_in_order(T3, L3)

        self.check_first(T3, L3)
        self.check_last(T3, L3)
        self.check_first(T1, L1)
        self.check_last(T1, L1)

    def test_insert_at_end_big(self):
        T3 = AVLTreeList()
        L3 = []
        for i in range(100):
            T3.insert(T3.length(), i)
            L3.insert(len(L3), i)
        self.compare_with_list_by_retrieve(T3, L3)
        self.compare_with_list_by_in_order(T3, L3)
        self.check_first(T3, L3)
        self.check_last(T3, L3)

    def test_insert_at_the_middle_small(self):
        for j in range(10, 20):
            T = AVLTreeList()
            L = []

            for i in range(j):
                T.insert(i//2, i)
                L.insert(i//2, i)
                self.compare_with_list_by_retrieve(T, L)
                self.compare_with_list_by_in_order(T, L)
        self.check_first(T, L)
        self.check_last(T, L)

    def test_insert_at_the_middle_big(self):
        T4 = AVLTreeList()
        L4 = []

        for i in range(100):
            T4.insert(i//2, i)
            L4.insert(i//2, i)
            self.compare_with_list_by_retrieve(T4, L4)
            self.compare_with_list_by_in_order(T4, L4)
            self.check_first(T4, L4)
            self.check_last(T4, L4)

    def test_insert_alternately(self):
        T5 = AVLTreeList()
        L5 = []

        for i in range(100):
            if i % 5 == 0:
                T5.insert(0, i)
                L5.insert(0, i)
            elif i % 5 == 1:
                T5.insert(len(L5), i)
                L5.insert(len(L5), i)
            elif i % 5 == 2:
                T5.insert(i//2, i)
                L5.insert(i//2, i)
            elif i % 5 == 3:
                T5.insert(i//3, i)
                L5.insert(i//3, i)
            else:
                T5.insert(i//7, i)
                L5.insert(i//7, i)
            if i % 10 == 0:
                self.compare_with_list_by_retrieve(T5, L5)
                self.compare_with_list_by_in_order(T5, L5)
                self.check_first(T5, L5)
                self.check_last(T5, L5)

    ### TESTING DELETION ### (assuming insertion works perfectly)#
    def test_deleting_not_existing(self):
        self.assertEqual(self.emptyList.delete(0), -1)
        self.assertEqual(self.twentyTree.delete(-1), -1)
        self.assertEqual(self.twentyTree.delete(30), -1)

    def test_delete_list_with_only_one_element(self):
        T = AVLTreeList()
        T.insert(0, 1)
        T.delete(0)
        self.assertEqual(T.getRoot().value, "None")
        self.assertEqual(T.min.value, "None")
        self.assertEqual(T.max.value, "None")
        self.assertEqual(T.first(),  None)
        self.assertEqual(T.last(), None)

    def test_delete_from_list_with_two_elements(self):
        T1 = AVLTreeList()
        for i in range(2):
            T1.appendd(i)
        T1.delete(0)

        self.assertEqual(T1.getRoot().getValue(), 1)
        T1.delete(0)
        self.assertEqual(T1.getRoot().value, "None")

        T1 = AVLTreeList()
        for i in range(2):
            T1.appendd(i)
        T1.delete(1)
        self.assertEqual(T1.getRoot().getValue(), 0)
        T1.delete(0)
        self.assertEqual(T1.getRoot().value, "None")

        T1 = AVLTreeList()
        for i in range(2):
            T1.appendd(0)

        T1.delete(0)
        self.assertEqual(T1.getRoot().getValue(), 0)
        T1.delete(0)
        self.assertEqual(T1.getRoot().value, "None")

    def test_delete_from_start_small(self):
        for j in range(10, 20):
            T = AVLTreeList()
            L = []
            for i in range(j):
                T.appendd(i)
                L.append(i)

            while not T.empty():
                self.compare_with_list_by_in_order(T, L)
                self.compare_with_list_by_retrieve(T, L)
                self.check_first(T, L)
                self.check_last(T, L)
                T.delete(0)
                L.pop(0)

        self.assertEqual(len(L), 0)

    def test_delete_from_start_big(self):
        T = AVLTreeList()
        L = []
        for i in range(100):
            T.appendd(i)
            L.append(i)

        while not T.empty():
            self.compare_with_list_by_in_order(T, L)
            self.compare_with_list_by_retrieve(T, L)
            self.check_first(T, L)
            self.check_last(T, L)
            T.delete(0)
            L.pop(0)

        self.assertEqual(len(L), 0)

    def test_delete_from_end_small(self):
        for j in range(10, 20):
            T = AVLTreeList()
            L = []
            for i in range(j):
                T.appendd(i)
                L.append(i)

            while not T.empty():
                self.compare_with_list_by_in_order(T, L)
                self.compare_with_list_by_retrieve(T, L)
                self.check_first(T, L)
                self.check_last(T, L)
                T.delete(T.length()-1)
                L.pop(len(L)-1)

            self.assertEqual(len(L), 0)

    def test_delete_from_end_big(self):
        T = AVLTreeList()
        L = []
        for i in range(100):
            T.appendd(i)
            L.append(i)

        while not T.empty():
            self.compare_with_list_by_in_order(T, L)
            self.compare_with_list_by_retrieve(T, L)
            self.check_first(T, L)
            self.check_last(T, L)
            T.delete(T.length()-1)
            L.pop(len(L)-1)

        self.assertEqual(len(L), 0)

    def test_delete_from_middle_small(self):
        for j in range(10, 20):
            T = AVLTreeList()
            L = []
            for i in range(j):
                T.appendd(i)
                L.append(i)

            while not T.empty():
                self.compare_with_list_by_in_order(T, L)
                self.compare_with_list_by_retrieve(T, L)
                self.check_first(T, L)
                self.check_last(T, L)
                T.delete(T.length()//2)
                L.pop(len(L)//2)

            self.assertEqual(len(L), 0)

    def test_delete_from_middle_big(self):
        T = AVLTreeList()
        L = []

        for i in range(100):
            T.appendd(i)
            L.append(i)

        while not T.empty():
            self.compare_with_list_by_in_order(T, L)
            self.compare_with_list_by_retrieve(T, L)
            self.check_first(T, L)
            self.check_last(T, L)
            T.delete(T.length()//2)
            L.pop(len(L)//2)

        self.assertEqual(len(L), 0)

    def test_delete_altenatly(self):
        T = AVLTreeList()
        L = []
        for i in range(100):
            T.appendd(i)
            L.append(i)
        cnt = 0
        while not T.empty():
            self.compare_with_list_by_in_order(T, L)
            self.compare_with_list_by_retrieve(T, L)
            self.check_first(T, L)
            self.check_last(T, L)
            if cnt % 4 == 0:
                T.delete(T.length()//2)
                L.pop(len(L)//2)
            elif cnt % 4 == 1:
                T.delete(0)
                L.pop(0)
            elif cnt % 4 == 2:
                T.delete(T.length()-1)
                L.pop(len(L)-1)
            else:
                T.delete(T.length()//4)
                L.pop(len(L)//4)
            cnt += 1

        self.assertEqual(len(L), 0)

    ### TESTING INSERTION AND DELETION TOGETHER###

    def test_delete_and_insert_equal_number_of_times_small(self):
        cnt = 1
        T = AVLTreeList()
        for i in range(10):
            T.appendd(i)

        for i in range(100):
            if cnt % 2 == 0:
                T.insert(cnt % T.length(), i+10)
            else:
                T.delete(cnt % T.length())
            cnt += 17

    def test_delete_and_insert_equal_number_of_times_big(self):
        cnt = 1
        T = AVLTreeList()
        for i in range(100):
            T.appendd(i)

        for i in range(1000):
            if cnt % 2 == 0:
                T.insert(cnt % T.length(), i+10)
            else:
                T.delete(cnt % T.length())
            cnt += 17

    def test_delete_and_insert_with_more_insertions_small(self):
        T = AVLTreeList()
        T.appendd(40)
        for i in range(30):
            if (i % 3 != 2):
                T.insert((i*17) % T.length(), i)
            else:
                T.delete((i*17) % T.length())

    def test_delete_and_insert_with_more_insertions_big(self):
        T = AVLTreeList()
        T.appendd(40)
        for i in range(150):
            if (i % 3 != 2):
                T.insert((i*17) % T.length(), i)
            else:
                T.delete((i*17) % T.length())

    def test_delete_and_insert_altenatly_small(self):
        T = AVLTreeList()
        L = []

        for i in range(20):
            if i % 3 == 0:
                T.insert(T.length()//2, i)
                L.insert(len(L)//2, i)
            elif i % 3 == 1:
                T.insert(0, i)
                L.insert(0, i)
            else:
                T.delete(T.length()//2)
                L.pop(len(L)//2)
            self.compare_with_list_by_in_order(T, L)
            self.compare_with_list_by_retrieve(T, L)
            self.check_first(T, L)
            self.check_last(T, L)

    ### TESTING FAMILTY ### (testing that node == node.getchild.gerparent)#

    def check_family(self, node, tree):
        self.assertEqual(node, node.getLeft().getParent())
        self.assertEqual(node, node.getRight().getParent())

    def test_family_basic(self):
        self.in_order(self.twentyTree, self.twentyTree.getRoot(),
                      self.check_family)

        self.assertEqual(self.twentyTree.getRoot().getParent().value, "None")

    def test_family_after_insertion_at_start(self):
        T2 = AVLTreeList()

        for i in range(50):
            T2.insert(0, i)
            self.in_order(T2, T2.getRoot(), self.check_family)

    def test_family_after_deletion_from_start(self):
        T = AVLTreeList()
        for i in range(50):
            T.insert(0, i)

        for i in range(49):
            T.delete(0)
            self.in_order(T, T.getRoot(), self.check_family)

    def test_family_after_insertion_at_end(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(T3.length(), i)
            self.in_order(T3, T3.getRoot(), self.check_family)

    def test_family_after_deletion_from_end(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(T3.length(), i)

        for i in range(49):
            T3.delete(T3.length()-1)
            self.in_order(T3, T3.getRoot(), self.check_family)

    def test_family_after_insertion_at_middle(self):
        T4 = AVLTreeList()

        for i in range(50):
            T4.insert(i//2, i)
            self.in_order(T4, T4.getRoot(), self.check_family)

    def test_family_after_deletion_from_middle(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(0, i)

        for i in range(49):
            if i//2 < T3.length():
                T3.delete(i//2)
                self.in_order(T3, T3.getRoot(), self.check_family)

    def test_family_after_insertion_alternatly(self):
        T5 = AVLTreeList()

        for i in range(200):
            if i % 5 == 0:
                T5.insert(0, i)
            elif i % 5 == 1:
                T5.insert(T5.length(), i)
            elif i % 5 == 2:
                T5.insert(i//2, i)
            elif i % 5 == 3:
                T5.insert(i//3, i)
            else:
                T5.insert(i//7, i)
            self.in_order(T5, T5.getRoot(), self.check_family)

    def test_family_after_deletion_alternatly(self):
        T = AVLTreeList()

        for i in range(100):
            T.insert(0, i)

        for i in range(99):
            if i % 5 == 0:
                T.delete(0)
            elif i % 5 == 1:
                T.delete(T.length()-1)
            elif i % 5 == 2:
                T.delete((T.length()-1)//2)
            elif i % 5 == 3:
                T.delete((T.length()-1)//3)
            else:
                T.delete((T.length()-1)//7)
            self.in_order(T, T.getRoot(), self.check_family)

    def test_family_after_deleting_and_inserting_small(self):
        T = AVLTreeList()

        for i in range(20):
            if i % 3 == 0:
                T.insert(T.length()//2, i)
            elif i % 3 == 1:
                T.insert(0, i)
            else:
                T.delete(T.length()//2)
            self.in_order(T, T.getRoot(), self.check_family)

    def test_family_after_deleting_and_inserting_big(self):
        T = AVLTreeList()

        for i in range(500):
            if i % 3 == 0:
                T.insert(T.length()//2, i)
            elif i % 3 == 1:
                T.insert(0, i)
            else:
                T.delete(T.length()//2)
            self.in_order(T, T.getRoot(), self.check_family)

    ###TESTING SIZE###

    def check_size(self, node, tree):
        self.assertEqual(node.getSize(), node.getLeft(
        ).getSize() + node.getRight().getSize() + 1)

    def test_size_after_insertion_at_start(self):
        T2 = AVLTreeList()

        for i in range(50):
            T2.insert(0, i)
            self.in_order(T2, T2.getRoot(), self.check_size)

    def test_size_after_deletion_from_start(self):
        T = AVLTreeList()
        for i in range(50):
            T.insert(0, i)

        for i in range(49):
            T.delete(0)
            self.in_order(T, T.getRoot(), self.check_size)

    def test_size_after_insertion_at_end(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(T3.length(), i)
            self.in_order(T3, T3.getRoot(), self.check_size)

    def test_size_after_deletion_from_end(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(T3.length(), i)

        for i in range(49):
            T3.delete(T3.length()-1)
            self.in_order(T3, T3.getRoot(), self.check_size)

    def test_size_after_insertion_at_middle(self):
        T4 = AVLTreeList()

        for i in range(50):
            T4.insert(i//2, i)
            self.in_order(T4, T4.getRoot(), self.check_size)

    def test_size_after_deletion_from_middle(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(0, i)

        for i in range(49):
            if i//2 < T3.length():
                T3.delete(i//2)
                self.in_order(T3, T3.getRoot(), self.check_size)

    def test_size_after_insertion_alternatly(self):
        T5 = AVLTreeList()

        for i in range(200):
            if i % 5 == 0:
                T5.insert(0, i)
            elif i % 5 == 1:
                T5.insert(T5.length(), i)
            elif i % 5 == 2:
                T5.insert(i//2, i)
            elif i % 5 == 3:
                T5.insert(i//3, i)
            else:
                T5.insert(i//7, i)
            self.in_order(T5, T5.getRoot(), self.check_size)

    def test_size_after_deletion_alternatly(self):
        T = AVLTreeList()

        for i in range(100):
            T.insert(0, i)

        for i in range(99):
            if i % 5 == 0:
                T.delete(0)
            elif i % 5 == 1:
                T.delete(T.length()-1)
            elif i % 5 == 2:
                T.delete((T.length()-1)//2)
            elif i % 5 == 3:
                T.delete((T.length()-1)//3)
            else:
                T.delete((T.length()-1)//7)
            self.in_order(T, T.getRoot(), self.check_size)

    def test_size_after_deleting_and_inserting_small(self):
        T = AVLTreeList()

        for i in range(20):
            if i % 3 == 0:
                T.insert(T.length()//2, i)
            elif i % 3 == 1:
                T.insert(0, i)
            else:
                T.delete(T.length()//2)
            self.in_order(T, T.getRoot(), self.check_size)

    def test_size_after_deleting_and_inserting_big(self):
        T = AVLTreeList()

        for i in range(500):
            if i % 3 == 0:
                T.insert(T.length()//2, i)
            elif i % 3 == 1:
                T.insert(0, i)
            else:
                T.delete(T.length()//2)
            self.in_order(T, T.getRoot(), self.check_size)

        ###TESTING HEIGHT###

    def check_height(self, node, tree):
        self.assertEqual(node.getHeight(), max(node.getLeft(
        ).getHeight(), node.getRight().getHeight()) + 1)

    def test_height_after_insertion_at_start(self):
        T2 = AVLTreeList()

        for i in range(50):
            T2.insert(0, i)
            self.in_order(T2, T2.getRoot(), self.check_height)

    def test_height_after_deletion_from_start(self):
        T = AVLTreeList()
        for i in range(50):
            T.insert(0, i)

        for i in range(49):
            T.delete(0)
            self.in_order(T, T.getRoot(), self.check_height)

    def test_height_after_insertion_at_end(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(T3.length(), i)
            self.in_order(T3, T3.getRoot(), self.check_height)

    def test_height_after_deletion_from_end(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(T3.length(), i)

        for i in range(49):
            T3.delete(T3.length()-1)
            self.in_order(T3, T3.getRoot(), self.check_height)

    def test_height_after_insertion_at_middle(self):
        T4 = AVLTreeList()

        for i in range(50):
            T4.insert(i//2, i)
            self.in_order(T4, T4.getRoot(), self.check_height)

    def test_height_after_deletion_from_middle(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(0, i)

        for i in range(49):
            if i//2 < T3.length():
                T3.delete(i//2)
                self.in_order(T3, T3.getRoot(), self.check_height)

    def test_height_after_insertion_alternatly(self):
        T5 = AVLTreeList()

        for i in range(200):
            if i % 5 == 0:
                T5.insert(0, i)
            elif i % 5 == 1:
                T5.insert(T5.length(), i)
            elif i % 5 == 2:
                T5.insert(i//2, i)
            elif i % 5 == 3:
                T5.insert(i//3, i)
            else:
                T5.insert(i//7, i)
            self.in_order(T5, T5.getRoot(), self.check_height)

    def test_height_after_deletion_alternatly(self):
        T = AVLTreeList()

        for i in range(100):
            T.insert(0, i)

        for i in range(99):
            if i % 5 == 0:
                T.delete(0)
            elif i % 5 == 1:
                T.delete(T.length()-1)
            elif i % 5 == 2:
                T.delete((T.length()-1)//2)
            elif i % 5 == 3:
                T.delete((T.length()-1)//3)
            else:
                T.delete((T.length()-1)//7)
            self.in_order(T, T.getRoot(), self.check_height)

    def test_height_after_deleting_and_inserting_small(self):
        T = AVLTreeList()

        for i in range(20):
            if i % 3 == 0:
                T.insert(T.length()//2, i)
            elif i % 3 == 1:
                T.insert(0, i)
            else:
                T.delete(T.length()//2)
            self.in_order(T, T.getRoot(), self.check_height)

    def test_height_after_deleting_and_inserting_big(self):
        T = AVLTreeList()

        for i in range(500):
            if i % 3 == 0:
                T.insert(T.length()//2, i)
            elif i % 3 == 1:
                T.insert(0, i)
            else:
                T.delete(T.length()//2)
            self.in_order(T, T.getRoot(), self.check_height)

    ### TESTING BALACNE FACTOR ###

    def check_BF(self, node, tree):
        self.assertTrue(abs(node.getLeft().getHeight() -
                            node.getRight().getHeight()) < 2)

    def test_BF_after_insertion_at_start(self):
        T2 = AVLTreeList()

        for i in range(50):
            T2.insert(0, i)
            self.in_order(T2, T2.getRoot(), self.check_BF)

    def test_BF_after_deletion_from_start(self):
        T = AVLTreeList()
        for i in range(50):
            T.insert(0, i)

        for i in range(49):
            T.delete(0)
            self.in_order(T, T.getRoot(), self.check_BF)

    def test_BF_after_insertion_at_end(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(T3.length(), i)
            self.in_order(T3, T3.getRoot(), self.check_BF)

    def test_BF_after_deletion_from_end(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(T3.length(), i)

        for i in range(49):
            T3.delete(T3.length()-1)
            self.in_order(T3, T3.getRoot(), self.check_BF)

    def test_BF_after_insertion_at_middle(self):
        T4 = AVLTreeList()

        for i in range(50):
            T4.insert(i//2, i)
            self.in_order(T4, T4.getRoot(), self.check_BF)

    def test_BF_after_deletion_from_middle(self):
        T3 = AVLTreeList()

        for i in range(50):
            T3.insert(0, i)

        for i in range(49):
            if i//2 < T3.length():
                T3.delete(i//2)
                self.in_order(T3, T3.getRoot(), self.check_BF)

    def test_BF_after_insertion_alternatly(self):
        T5 = AVLTreeList()

        for i in range(200):
            if i % 5 == 0:
                T5.insert(0, i)
            elif i % 5 == 1:
                T5.insert(T5.length(), i)
            elif i % 5 == 2:
                T5.insert(i//2, i)
            elif i % 5 == 3:
                T5.insert(i//3, i)
            else:
                T5.insert(i//7, i)
            self.in_order(T5, T5.getRoot(), self.check_BF)

    def test_BF_after_deletion_alternatly(self):
        T = AVLTreeList()

        for i in range(100):
            T.insert(0, i)

        for i in range(99):
            if i % 5 == 0:
                T.delete(0)
            elif i % 5 == 1:
                T.delete(T.length()-1)
            elif i % 5 == 2:
                T.delete((T.length()-1)//2)
            elif i % 5 == 3:
                T.delete((T.length()-1)//3)
            else:
                T.delete((T.length()-1)//7)
            self.in_order(T, T.getRoot(), self.check_BF)

    def test_BF_after_deleting_and_inserting_small(self):
        T = AVLTreeList()

        for i in range(20):
            if i % 3 == 0:
                T.insert(T.length()//2, i)
            elif i % 3 == 1:
                T.insert(0, i)
            else:
                T.delete(T.length()//2)
            self.in_order(T, T.getRoot(), self.check_BF)

    def test_BF_after_deleting_and_inserting_big(self):
        T = AVLTreeList()

        for i in range(500):
            if i % 3 == 0:
                T.insert(T.length()//2, i)
            elif i % 3 == 1:
                T.insert(0, i)
            else:
                T.delete(T.length()//2)
            self.in_order(T, T.getRoot(), self.check_BF)

    ###TESTING SEARCH###

    def test_search_basic(self):
        self.assertEqual(-1, self.emptyList.search(None))
        self.assertEqual(-1, self.emptyList.search(20))
        for i in range(20):
            self.assertEqual(i, self.twentyTree.search(i))
        self.assertEqual(-1, self.twentyTree.search(21))

    def test_search_after_insertion_at_start(self):
        T2 = AVLTreeList()
        L2 = []

        for i in range(50):
            T2.insert(0, i)
            L2.insert(0, i)
            for j in range(len(L2)):
                self.assertEqual(T2.search(L2[j]), j)
            self.assertEqual(-1, T2.search(-20))

    def test_search_after_deletion_from_start(self):
        T = AVLTreeList()
        L = []
        for i in range(50):
            T.appendd(i)
            L.append(i)

        for i in range(49):
            T.delete(0)
            L.pop(0)
            for j in range(len(L)):
                self.assertEqual(T.search(L[j]), j)
            self.assertEqual(-1, T.search(-20))

    def test_search_after_insertion_at_end(self):
        T = AVLTreeList()
        L = []

        for i in range(50):
            T.appendd(i)
            L.append(i)
            for j in range(len(L)):
                self.assertEqual(T.search(L[j]), j)
            self.assertEqual(-1, T.search(-20))

    def test_search_after_deletion_from_end(self):
        T = AVLTreeList()
        L = []

        for i in range(50):
            T.appendd(i)
            L.append(i)

        for i in range(49):
            T.delete(T.length()-1)
            L.pop(len(L)-1)
            for j in range(len(L)):
                self.assertEqual(T.search(L[j]), j)
            self.assertEqual(-1, T.search(-20))

    def test_search_after_insertion_at_middle(self):
        T = AVLTreeList()
        L = []

        for i in range(50):
            T.insert(i//2, i)
            L.insert(i//2, i)
            for j in range(len(L)):
                self.assertEqual(T.search(L[j]), j)
            self.assertEqual(-1, T.search(-20))

    def test_search_after_deletion_from_middle(self):
        T = AVLTreeList()
        L = []

        for i in range(50):
            T.insert(0, i)
            L.insert(0, i)

        for i in range(49):
            T.delete(T.length()//2)
            L.pop(len(L)//2)
            for j in range(len(L)):
                self.assertEqual(T.search(L[j]), j)
            self.assertEqual(-1, T.search(-20))

    def test_search_after_insertion_alternatly(self):
        T = AVLTreeList()
        L = []

        for i in range(200):
            if i % 5 == 0:
                T.insert(0, i)
                L.insert(0, i)
            elif i % 5 == 1:
                T.appendd(i)
                L.append(i)
            elif i % 5 == 2:
                T.insert(i//2, i)
                L.insert(i//2, i)
            elif i % 5 == 3:
                T.insert(i//3, i)
                L.insert(i//3, i)
            else:
                T.insert(i//7, i)
                L.insert(i//7, i)
            for j in range(len(L)):
                self.assertEqual(T.search(L[j]), j)
            self.assertEqual(-1, T.search(-20))

    def test_height_search_deletion_alternatly(self):
        T = AVLTreeList()
        L = []

        for i in range(100):
            T.insert(0, i)
            L.insert(0, i)

        for i in range(99):
            if i % 5 == 0:
                T.delete(0)
                L.pop(0)
            elif i % 5 == 1:
                T.delete(T.length()-1)
                L.pop(len(L)-1)
            elif i % 5 == 2:
                T.delete((T.length()-1)//2)
                L.pop((len(L)-1)//2)
            elif i % 5 == 3:
                T.delete((T.length()-1)//3)
                L.pop((len(L)-1)//3)
            else:
                T.delete((T.length()-1)//7)
                L.pop((len(L)-1)//7)

            for j in range(len(L)):
                self.assertEqual(T.search(L[j]), j)
            self.assertEqual(-1, T.search(-20))

    def test_search_after_deleting_and_inserting_small(self):
        T = AVLTreeList()
        L = []

        for i in range(20):
            if i % 3 == 0:
                T.insert(T.length()//2, i)
                L.insert(len(L)//2, i)
            elif i % 3 == 1:
                T.insert(0, i)
                L.insert(0, i)
            else:
                T.delete(T.length()//2)
                L.pop(len(L)//2)
            for j in range(len(L)):
                self.assertEqual(T.search(L[j]), j)
            self.assertEqual(-1, T.search(-20))

    def test_search_after_deleting_and_inserting_big(self):
        T = AVLTreeList()
        L = []

        for i in range(500):
            if i % 3 == 0:
                T.insert(T.length()//2, i)
                L.insert(len(L)//2, i)
            elif i % 3 == 1:
                T.insert(0, i)
                L.insert(0, i)
            else:
                T.delete(T.length()//2)
                L.pop(len(L)//2)
            for j in range(len(L)):
                self.assertEqual(T.search(L[j]), j)
            self.assertEqual(-1, T.search(-20))

    ### TESTING CONCAT AND LISTASARRAY ###
    TR1 = AVLTreeList()
    LR1 = list()
    TR2 = AVLTreeList()
    LR2 = list()

    for i in range(20):
        TR1.appendd(i)
        TR2.appendd(i+10)
        LR1.append(i)
        LR2.append(i+10)

    # def test_compare_treelist_and_list(self):
        # self.assertEqual (self.TR1.listToArray(),self.LR1)

    TR1.concat(TR2)
    LR3 = LR1 + LR2

    def test_compare_concatinated_treelists_and_list(self):
        self.compare_with_list_by_in_order(self.TR1, self.LR3)
        self.compare_with_list_by_retrieve(self.TR1, self.LR3)
        self.check_first(self.TR1, self.LR3)
        self.check_last(self.TR1, self.LR3)
        self.assertEqual(self.TR1.listToArray(), self.LR3)
        self.in_order(self.TR1, self.TR1.getRoot(), self.check_BF)
        self.in_order(self.TR1, self.TR1.getRoot(), self.check_height)
        self.in_order(self.TR1, self.TR1.getRoot(), self.check_size)
        T1 = AVLTreeList()
        T2 = AVLTreeList()
        L1 = list()
        L2 = list()
        for i in range(10):
            T1.appendd(i)
            L1.append(i)
        for i in range(5):
            T2.appendd(i)
            L2.append(i)
        T1.concat(T2)
        L3 = L1+L2
        self.compare_with_list_by_in_order(T1, L3)
        self.compare_with_list_by_retrieve(T1, L3)
        self.check_first(T1, L3)
        self.check_last(T1, L3)
        self.assertEqual(T1.listToArray(), L3)
        self.in_order(T1, T1.getRoot(), self.check_BF)
        self.in_order(T1, T1.getRoot(), self.check_height)
        self.in_order(T1, T1.getRoot(), self.check_size)
        T3 = AVLTreeList()
        T4 = AVLTreeList()
        L3 = list()
        L4 = list()
        for i in range(10):
            T4.appendd(i)
            L4.append(i)
        for i in range(5):
            T3.appendd(i)
            L3.append(i)
        T3.concat(T4)
        L5 = L3+L4
        self.compare_with_list_by_in_order(T3, L5)
        self.compare_with_list_by_retrieve(T3, L5)
        self.check_first(T3, L5)
        self.check_last(T3, L5)
        self.assertEqual(T3.listToArray(), L5)
        self.in_order(T3, T3.getRoot(), self.check_BF)
        self.in_order(T3, T3.getRoot(), self.check_height)
        self.in_order(T3, T3.getRoot(), self.check_size)

    def test_compare_concatinated_treelists_and_list_small(self):
        T1 = AVLTreeList()
        T2 = AVLTreeList()
        L1 = list()
        L2 = list()
        for i in range(3):
            T1.appendd(i)
            L1.append(i)
        for i in range(1):
            T2.appendd(i)
            L2.append(i)
        T1.concat(T2)
        L3 = L1+L2
        self.compare_with_list_by_in_order(T1, L3)
        self.compare_with_list_by_retrieve(T1, L3)
        self.check_first(T1, L3)
        self.check_last(T1, L3)
        self.assertEqual(T1.listToArray(), L3)
        self.in_order(T1, T1.getRoot(), self.check_BF)
        self.in_order(T1, T1.getRoot(), self.check_height)
        self.in_order(T1, T1.getRoot(), self.check_size)
        T3 = AVLTreeList()
        T4 = AVLTreeList()
        L3 = list()
        L4 = list()
        for i in range(3):
            T4.appendd(i)
            L4.append(i)
        for i in range(1):
            T3.appendd(i)
            L3.append(i)
        T3.concat(T4)
        L5 = L3+L4
        self.compare_with_list_by_in_order(T3, L5)
        self.compare_with_list_by_retrieve(T3, L5)
        self.check_first(T3, L5)
        self.check_last(T3, L5)
        self.assertEqual(T3.listToArray(), L5)
        self.in_order(T3, T3.getRoot(), self.check_BF)
        self.in_order(T3, T3.getRoot(), self.check_height)
        self.in_order(T3, T3.getRoot(), self.check_size)

    def test_concat_with_empty_list_as_self(self):
        empty = AVLTreeList()
        LR4 = [i for i in range(10)]
        TR4 = AVLTreeList()
        for i in range(10):
            TR4.appendd(i)
        empty.concat(TR4)
        self.assertEqual(empty.listToArray(), LR4)
        self.check_first(empty, LR4)
        self.check_last(empty, LR4)

    def test_concat_with_empty_list_as_lst(self):
        self.TR1.concat(self.emptyList)
        self.assertEqual(self.TR1.listToArray(), self.LR3)
        self.check_first(self.TR1, self.LR3)
        self.check_last(self.TR1, self.LR3)

    def test_concat_first(self):
        self.assertEqual(self.TR1.first(), self.LR3[0])

    def test_concat_last(self):
        self.assertEqual(self.TR1.last(), self.LR3[-1])

    def test_concat_height_difference_empty_lists(self):
        T1 = AVLTreeList()
        T2 = AVLTreeList()
        self.assertEqual(T1.concat(T2), 0)

    def test_assert_height_difference_one_empty_list(self):
        T1 = AVLTreeList()
        T2 = AVLTreeList()
        T3 = AVLTreeList()
        T4 = AVLTreeList()
        for i in range(3):
            T1.appendd(i)
            T2.appendd(i)

        self.assertEqual(T1.concat(T3), 2)
        self.assertEqual(T4.concat(T2), 2)

    def test_assert_height_difference_non_empty_lists(self):
        T1 = AVLTreeList()
        T2 = AVLTreeList()
        for i in range(3):
            T1.appendd(i)
        for i in range(1):
            T2.appendd(i)
        self.assertEqual(T1.concat(T2), 1)
        T3 = AVLTreeList()
        T4 = AVLTreeList()
        for i in range(3):
            T4.appendd(i)
        for i in range(1):
            T3.appendd(i)
        self.assertEqual(T3.concat(T4), 1)


    ### TESTING SPLIT ###
    def check_root(self, tree):
        if not tree.empty():
            self.assertEqual(tree.getRoot().getParent().value, "None")

    def check_split(self, lst, res, i):
        self.assertEqual(lst[i], res[1])
        L1 = lst[:i]
        L2 = lst[i+1:]

        ##checks values##
        self.assertEqual(res[0].listToArray(), L1)
        self.compare_with_list_by_retrieve(res[0], L1)
        self.compare_with_list_by_in_order(res[0], L1)
        self.assertEqual(res[2].listToArray(), L2)
        self.compare_with_list_by_retrieve(res[2], L2)
        self.compare_with_list_by_in_order(res[2], L2)

        ##checks fields##
        self.check_first(res[0], L1)
        self.check_last(res[0], L1)
        self.in_order(res[0], res[0].getRoot(), self.check_family)
        self.in_order(res[0], res[0].getRoot(), self.check_height)
        self.in_order(res[0], res[0].getRoot(), self.check_size)
        self.in_order(res[0], res[0].getRoot(), self.check_BF)
        self.check_root(res[0])

        self.check_first(res[2], L2)
        self.check_last(res[2], L2)
        self.in_order(res[2], res[2].getRoot(), self.check_family)
        self.in_order(res[2], res[2].getRoot(), self.check_height)
        self.in_order(res[2], res[2].getRoot(), self.check_size)
        self.in_order(res[2], res[2].getRoot(), self.check_BF)
        self.check_root(res[2])

    def test_split_basic(self):
        L = []
        T = AVLTreeList()

        for i in range(10):
            L.append(i)
            T.appendd(i)

        res = T.split(5)
        self.check_split(L, res, 5)

    def test_split_basic_in_range(self):
        for j in range(10):
            L = []
            T = AVLTreeList()

            for i in range(10):
                L.append(i)
                T.appendd(i)

            res = T.split(j)
            self.check_split(L, res, j)

    def test_split_small(self):
        T = AVLTreeList()
        T.appendd('a')
        L = ['a']
        res = T.split(0)
        self.check_split(L, res, 0)

        for i in range(2):
            T = AVLTreeList()
            T.appendd('a')
            T.appendd('b')
            L = ['a', 'b']
            res = T.split(i)
            self.check_split(L, res, i)

    def test_split_big(self):
        for j in range(100):
            if j % 10 == 0:
                L = []
                T = AVLTreeList()

                for i in range(100):
                    L.append(i*17)
                    T.appendd(i*17)

                res = T.split(j)
                self.check_split(L, res, j)

    def test_split_big2(self):
        T = AVLTreeList()
        L = []
        for i in range(2000):
            L.append(i*17)
            T.appendd(i*17)

        res = T.split(1319)
        self.check_split(L, res, 1319)

    def test_search_after_split(self):
        for j in range(100):
            if j % 10 == 0:
                L = []
                T = AVLTreeList()

                for i in range(100):
                    L.append(i)
                    T.appendd(i)

                res = T.split(j)

                T1 = res[0]
                L1 = L[:j]
                T2 = res[2]
                L2 = L[j+1:]

                for i in range(100):
                    if i % 3 == 0:
                        T1.insert(T1.length()//2, i+100)
                        L1.insert(len(L1)//2, i+100)
                        T2.insert(T2.length()//2, i+100)
                        L2.insert(len(L2)//2, i+100)
                    elif i % 3 == 1:
                        T1.insert(0, i+100)
                        L1.insert(0, i+100)
                        T2.insert(0, i+100)
                        L2.insert(0, i+100)
                    else:
                        T1.delete(T1.length()//2)
                        L1.pop(len(L1)//2)
                        T2.delete(T2.length()//2)
                        L2.pop(len(L2)//2)
                    for j in range(len(L1)):
                        self.assertEqual(T1.search(L1[j]), j)
                    for j in range(len(L2)):
                        self.assertEqual(T2.search(L2[j]), j)

                    self.assertEqual(-1, T1.search(-20))
                    self.assertEqual(-1, T2.search(-20))

    def test_num_of_balnce_ops(self):
        T = AVLTreeList()
        self.assertEqual(T.appendd(3), 0)
        self.assertEqual(T.insert(0, 1), 1)
        self.assertEqual(T.insert(1, 2), 3)

    # def test_successor_and_predeccessor(self):
    #     T = AVLTreeList()
    #     T.append(0)
    #     T.append(1)

    #     self.assertEqual(T.getSuccessorOf(T.getRoot()).getValue(), 1)
    #     self.assertEqual(T.getSuccessorOf(
    #         T.getRoot().getRight()), None)
    #     self.assertEqual(T.getPredecessorOf(
    #         T.getRoot().getRight()).getValue(), 0)


if __name__ == '__main__':
    unittest.main()
