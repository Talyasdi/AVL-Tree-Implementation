#username - Talyasdi
#id1      - 206962359
#name1    - Tal Yasdi
#id2      - 207188285
#name2    - Meital Shpigel

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
	@returns: False if self is a virtual node, True otherwise.
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

		return (self.treeSelect(self.root, i+1)).value  # Complexity - O(logn)

	"""inserts val at position i in the list

	@type i: int
	@pre: 0 <= i <= self.length()
	@param i: The intended index in the list to which we insert val
	@type val: str
	@param val: the value we inserts
	@rtype: list
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""
	def insert(self, i, val): #not finished
		new_node = AVLNode(val)
		new_node.setHeight(0)#need to build func for update height
		new_node.setSize(1)#need to build func for update size
		left_son= AVLNode("None")
		right_son=AVLNode("None")
		new_node.setLeft(left_son)
		new_node.setRight(right_son)
		left_son.setParent(new_node)
		right_son.setParent(new_node)
		if i == self.length():
			self.max.right =  new_node
			self.max =  new_node
		elif i<n:
			current_node_in_i = self.treeSelect(i+1)
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
			temp_node.setSize(temp_node.getLeft().getSize()+temp_node.getRight().getSize()+1)
			bf_temp_node = temp_node.getLeft().getHeight()-temp_node.getRight().getHeight()
			old_height = temp_node.getHeight()
			curr_new_height= max(temp_node.getLeft().getHeight(), temp_node.getRight().getHeight())+1
			if abs(bf_temp_node)<2 and curr_new_height == old_height:
				break
			elif abs(bf_temp_node)<2:
				temp_node.setHeight(curr_new_height)
				temp_node = temp_node.parent
			elif abs(bf_temp_node)==2:
				rotate_and_fix()
				break
		while temp_node.isRealNode():
			temp_node.setSize(temp_node.getLeft().getSize() + temp_node.getRight().getSize() + 1)
			temp_node = temp_node.parent








		return -1


	"""deletes the i'th item in the list

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: The intended index in the list to be deleted
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""
	def delete(self, i):
		return -1

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
		return None

	"""concatenates lst to self

	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@rtype: int
	@returns: the absolute value of the difference between the height of the AVL trees joined
	"""
	def concat(self, lst):
		return None

	"""searches for a *value* in the list

	@type val: str
	@param val: a value to be searched
	@rtype: int
	@returns: the first index that contains val, -1 if not found.
	"""
	def search(self, val):  # Complexity - O(n) - we go through each edge twice at most (based on recitation - proof at documentary file)
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
		return None

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
		if curr_node.isRealNode:
			if not curr_node.left.isRealNode:
				return curr_node
			else:
				curr_node = curr_node.left
				while curr_node.left.isRealNode:
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
		if curr_node.isRealNode:
			if not curr_node.right.isRealNode:
				return curr_node
			else:
				curr_node = curr_node.right
				while curr_node.right.isRealNode:
					curr_node = curr_node.right

				return curr_node

		# if the node does not have a left child
		else:
			while not self.isRightChild(curr_node):
				curr_node = curr_node.parent

			return curr_node.parent

	"""Auxiliary Function - general rotate - checks which rotation needs to be made and calls it

				@rtype: None
			

				"""

	def rotate_and_fix(self, node):  # Complexity
		return None

	"""Auxiliary Function - returns true if the node is a right child of its parent
	
			@rtype: boolean value
			@returns: true if the node is a right child of its parent
	
			"""

	def isRightChild(self, node):  # Complexity - O(1) - access to a pointer
		parent = node.parent
		return parent.right == node


