
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

    def insert(self, i, val):  # Complexity - O(logn)
        rebalanceOp = 0

        # Creating node with the value to insert, with two virtual children
        new_node = AVLNode(val)
        new_node.setHeight(0)
        new_node.setSize(1)
        left_son = AVLNode("None")
        right_son = AVLNode("None")
        new_node.setLeft(left_son)
        new_node.setRight(right_son)
        left_son.setParent(new_node)
        right_son.setParent(new_node)

        # Inserting to an empty list - new node is the root, min and max
        if self.empty():
            self.root.setLeft(new_node)
            new_node.setParent(self.root)
            self.root = new_node
            self.min = new_node
            self.max = new_node
            return rebalanceOp

        # Inserting at the end of the list - fixing the max field
        if i == self.length():
            self.max.right = new_node
            new_node.setParent(self.max)
            self.max = new_node

        # Inserting somewhere in the middle of the list or at the beginning
        elif i < self.length():
            current_node_in_i = self.treeSelect(self.root, i + 1) # In order to insert before the current i'th node
            if not current_node_in_i.left.isRealNode():
                current_node_in_i.setLeft(new_node)
                new_node.setParent(current_node_in_i)
            else:
                pred_current_node_in_i = self.predeccesor(current_node_in_i)
                pred_current_node_in_i.setRight(new_node)
                new_node.setParent(pred_current_node_in_i)

            # If we Insert the node to the beginning - we need to fix the min field
            if i == 0:
                self.min = new_node
        temp_node = new_node.parent

        # A loop for fixing fields and rotating if needed, starting from the parent of the inserted node
        while temp_node.isRealNode():
            self.updateSize(temp_node)
            bf_temp_node = self.balanceFactor(temp_node)
            old_height = temp_node.getHeight()
            curr_new_height = max(temp_node.getLeft().getHeight(), temp_node.getRight().getHeight()) + 1

            # If the height hasn't changed and the BF is ok - terminate
            if abs(bf_temp_node) < 2 and curr_new_height == old_height:
                break

            # If the height has changed and the BF is ok - fix the height and continue to parent
            elif abs(bf_temp_node) < 2:
                temp_node.setHeight(curr_new_height)
                rebalanceOp += 1
                temp_node = temp_node.parent

            # If the BF is not ok - rotate and fix fields, then terminate
            elif abs(bf_temp_node) == 2:
                rebalanceOp += self.rotateAndFix(temp_node)
                break

        # Continue fixing the size field until we reach the root
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

    def delete(self, i): # Complexity - O(logn)

        # If we try to delete a node in a non-existing index - return -1
        if (i > self.length() - 1) or (i < 0):
            return -1

        rebalanceOp = 0
        is_two_children = False
        node_to_delete = self.treeSelect(self.root, i+1) # Finding the node to delete
        node_to_delete_height = node_to_delete.getHeight()
        left_child = node_to_delete.getLeft()
        right_child = node_to_delete.getRight()

        # Checks if we deleted min and fixing the min field
        if i == 0 and self.length() > 1:
            self.min = self.successor(node_to_delete)

        # Checks if we deleted max and fixing the max field
        if i == self.length()-1 and self.length() > 1:
            self.max = self.predeccesor(node_to_delete)

        # If node_to_delete is a leaf
        if node_to_delete.getSize() == 1:
            # If the tree has only one node (we deleted the root) - making the list an empty list
            if node_to_delete == self.root:
                node_to_delete.getParent().setRight(None)
                node_to_delete.getParent().setLeft(None)
                self.root = node_to_delete.getParent()
                self.min = node_to_delete.getParent()
                self.max = node_to_delete.getParent()
                parent_node = node_to_delete.getParent()
            else:
                parent_node = self.deleteLeaf(node_to_delete)

        # If node_to_delete has one child
        elif (left_child.isRealNode() and (not right_child.isRealNode())) or (right_child.isRealNode() and (not left_child.isRealNode())):
            parent_node = self.deleteNodeWithOneChild(node_to_delete)
            # If we deleted the root - fixing the root field to be the one child of the root
            if not parent_node.isRealNode():
                if parent_node.getLeft().isRealNode():
                    self.root = parent_node.getLeft()
                    self.min = parent_node.getLeft()
                    self.max = parent_node.getLeft()
                else:
                    self.root = parent_node.getRight()
                    self.min = parent_node.getRight()
                    self.max = parent_node.getRight()

        # If node_to_delete has two children
        else:
            parent_node = self.deleteNodeWithTwoChildren(node_to_delete)
            is_two_children = True
            # If we deleted the root - fixing the root field
            if not parent_node.isRealNode():
                if parent_node.getLeft().isRealNode():
                    self.root = parent_node.getLeft()
                else:
                    self.root = parent_node.getRight()

        # A loop for fixing fields and rotating if needed, starting from the parent of the physically deleted node
        while parent_node.isRealNode():
            self.updateSize(parent_node)
            bf_parent_node = self.balanceFactor(parent_node)

            # Edge case- if we deleted a node with two children, old height is considered differently at first iteration
            if is_two_children:
                old_height = node_to_delete_height
                is_two_children = False
            else:
                old_height = parent_node.getHeight()
            curr_new_height = max(parent_node.getLeft().getHeight(), parent_node.getRight().getHeight()) + 1

            # If the height hasn't changed and the BF is ok - terminate
            if abs(bf_parent_node) < 2 and curr_new_height == old_height:
                break

            # If the height has changed and the BF is ok - fix the height and continue to parent
            elif abs(bf_parent_node) < 2:
                parent_node.setHeight(curr_new_height)
                rebalanceOp += 1
                parent_node = parent_node.parent

            # If the BF is not ok - rotate and fix fields, then continue to the next relevant node
            elif abs(bf_parent_node) == 2:
                rebalanceOp += self.rotateAndFix(parent_node)
                parent_node = parent_node.parent.parent

        # Continue fixing fields until we reach the root
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

    def listToArray(self): # Complexity - O(n)
        res_array = []

        # If we make array to an empty list
        if self.empty():
            return res_array

        curr_node = self.min

        # Going through all nodes from the beginning to the end in-order
        while curr_node != self.max:
            res_array.append(curr_node.value)
            curr_node = self.successor(curr_node)
        res_array.append(curr_node.value) # Inserting the max node's value into the array
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

    def split(self, i): # Complexity - O(logn)
        tree1 = AVLTreeList()
        tree2 = AVLTreeList()

        # If we split by the first node at the list
        if i == 0:
            split_node_val = self.min.value
            self.delete(i)
            return [tree1, split_node_val, self]

        # If we split by the last node at the list
        elif i == self.length()-1:
            split_node_val =self.max.value
            self.delete(i)
            return [self, split_node_val, tree2]

        else:
            # Finding the node from which we split
            split_node = self.treeSelect(self.getRoot(), i+1)
            split_node_val = split_node.value

            # Keeping pointers to the new min and max of the two new trees
            min_tree1 = self.min
            max_tree1 = self.predeccesor(split_node)
            min_tree2 = self.successor(split_node)
            max_tree2 = self.max

            # If split_node has left child
            if split_node.getLeft().isRealNode():
                tree1.getRoot().setLeft(split_node.getLeft())
                split_node.getLeft().setParent(tree1.getRoot())
                tree1.root = split_node.getLeft()

            # If split_node has right child
            if split_node.getRight().isRealNode():
                tree2.getRoot().setLeft(split_node.getRight())
                split_node.getRight().setParent(tree2.getRoot())
                tree2.root = split_node.getRight()

            # Disconnect split_node from its children and its parent after saving its parent
            split_node.setRight(None)
            split_node.setLeft(None)
            is_right_child = self.isRightChild(split_node)
            curr_node = split_node.getParent()
            split_node.setParent(None)

            # A loop for joining subtrees along the way to the root
            while curr_node.isRealNode():
                curr_parent = curr_node.getParent()
                next_is_right_child = self.isRightChild(curr_node)

                # If we go up from a right child
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

                # If we go up from a left child
                else:
                    help_tree2 = AVLTreeList()
                    if curr_node.getRight().isRealNode():
                        help_tree2.getRoot().setLeft(curr_node.getRight())
                        curr_node.getRight().setParent(help_tree2.getRoot())
                        help_tree2.root = curr_node.getRight()
                    else:
                        curr_node.getRight().setParent(None)
                    tree2.join(curr_node,help_tree2)

                is_right_child = next_is_right_child
                curr_node = curr_parent

            # Disconnecting the virtual parent of the original root
            curr_node.setRight(None)
            curr_node.setLeft(None)

            # Fixing the min and max fields of the new trees
            tree1.min = min_tree1
            tree1.max = max_tree1
            tree2.min = min_tree2
            tree2.max = max_tree2

        return [tree1,split_node_val, tree2]


    """concatenates lst to self
    
    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def concat(self, lst): # Complexity - O(logn)
        new_min = self.min  # Keeping pointer to the min of the new list
        height_diff = abs(self.getRoot().getHeight() - lst.getRoot().getHeight())

        # If we concat lst to an empty list (self is empty)
        if self.empty():
            self.root = lst.root
            self.min = lst.min
            self.max = lst.max
            return height_diff

        # If we concat an empty list to self
        if lst.empty():
            return height_diff

        # Keeping pointers to the max of the new list and to the node we want to connect with
        new_max = lst.max
        max_node = self.max
        self.delete(self.length()-1)
        self.join(max_node, lst)

        # Fixing pointers of self
        self.max = new_max
        self.min = new_min
        return height_diff


    """searches for a *value* in the list
    
    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    """

    def search(self, val):  # Complexity - O(n)
        curr_node = self.min
        index = 0

        # Going through all nodes in order to look for val
        while curr_node != self.max:
            if curr_node.value == val:
                return index
            curr_node = self.successor(curr_node)
            index += 1

        # Checking if val is the value of the last node in list
        if curr_node.value == val:
            return index
        # If we haven't found val - return -1
        else:
            return -1

    """returns the root of the tree representing the list
    
    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """

    def getRoot(self): # Complexity - O(1)
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
        #  If the node has a right child
        if curr_node.isRealNode():
            # If the right node does not have a left child , then it is the successor
            if not curr_node.left.isRealNode():
                return curr_node
            else:
                curr_node = curr_node.left
                # Finding the successor by going left till we reach a node without a left child
                while curr_node.left.isRealNode():
                    curr_node = curr_node.left

                return curr_node

        # If the node does not have a right child
        else:
            # Finding the successor by going up until we reach a node which is a left child
            while self.isRightChild(curr_node):
                curr_node = curr_node.parent

            return curr_node.parent

    """Auxiliary Function - returns the node with rank(node) - 1 (the node at the previous index)
    
            @rtype: AVLNode
            @returns: the node at the previous index 
    
            """

    def predeccesor(self, node):  # Complexity - O(logn) - linear in the height of the tree
        curr_node = node.left
        #  If the node has a left child
        if curr_node.isRealNode():
            # If the left node does not have a right child , then it is the predecessor
            if not curr_node.right.isRealNode():
                return curr_node
            else:
                # Finding the predecessor by going right till we reach a node without a right child
                curr_node = curr_node.right
                while curr_node.right.isRealNode():
                    curr_node = curr_node.right

                return curr_node

        # If the node does not have a left child
        else:
            # Finding the predecessor by going up until we reach a node which is a right child
            while not self.isRightChild(curr_node):
                curr_node = curr_node.parent

            return curr_node.parent

    """Auxiliary Function - general rotate - checks which rotation needs to be made and calls it
    
                        @rtype: integer
                        @returns: number of rotations 
    
                        """

    def rotateAndFix(self, node):  # Complexity - O(1)
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

    def rotateRight(self, parent):  # Complexity - O(1)
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

        # If we change the root - fix the root field
        if self.root == parent:
            self.root = child

        # Update fields of the involved nodes
        self.updateSize(parent)
        self.updateHeight(parent)
        self.updateSize(child)
        self.updateHeight(child)

    """Auxiliary Function - rotates left
    
    """

    def rotateLeft(self, parent):  # Complexity - O(1)
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

        # If we change the root - fix the root field
        if self.root == parent:
            self.root = child

        # Update fields of the involved nodes
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

    def balanceFactor(self, node): # Complexity - O(1) - access to pointers
        return node.getLeft().getHeight() - node.getRight().getHeight()

    """Auxiliary Function - updates the size of a node
    
        """

    def updateSize(self, node): # Complexity - O(1) - access to pointers
        node.setSize(node.getLeft().getSize() + node.getRight().getSize() + 1)

    """Auxiliary Function - updates the height of a node
    
        """

    def updateHeight(self, node): # Complexity - O(1) - access to pointers
        node.setHeight(max(node.getLeft().getHeight(), node.getRight().getHeight()) + 1)

    """Auxiliary Function - deletes the leaf it gets
    
                @rtype: AVLNode
                @returns: the AVLNode from which the node has physically changed
    
            """
    def deleteLeaf(self, node_to_delete):  # Complexity - O(1)
        parent_res = node_to_delete.getParent()

        # Connecting virtual node to the parent of the leaf and vice versa (instead of the leaf)
        if self.isRightChild(node_to_delete):
            node_to_delete.getParent().setRight(node_to_delete.getRight())
            node_to_delete.getRight().setParent(node_to_delete.getParent())
        else:
            node_to_delete.getParent().setLeft(node_to_delete.getLeft())
            node_to_delete.getLeft().setParent(node_to_delete.getParent())

        # Disconnecting the node we deleted from the list
        node_to_delete.setParent(None)
        node_to_delete.setRight(None)
        node_to_delete.setLeft(None)
        return parent_res

    """Auxiliary Function - deletes the node it gets
    
                @rtype: AVLNode
                @returns: the AVLNode from which the node has physically changed
    
            """
    def deleteNodeWithOneChild(self, node_to_delete):  # Complexity
        # If node_to_delete has only right child
        # Connecting this child to the parent of the deleted node and vice versa
        if node_to_delete.getRight().isRealNode() and (not node_to_delete.getLeft().isRealNode()):
            if self.isRightChild(node_to_delete):
                node_to_delete.getParent().setRight(node_to_delete.getRight())
            else:
                node_to_delete.getParent().setLeft(node_to_delete.getRight())
            node_to_delete.getRight().setParent(node_to_delete.getParent())

        # If node_to_delete has only left child
        # Connecting this child to the parent of the deleted node and vice versa
        elif (not node_to_delete.getRight().isRealNode()) and node_to_delete.getLeft().isRealNode():
            if self.isRightChild(node_to_delete):
                node_to_delete.getParent().setRight(node_to_delete.getLeft())
            else:
                node_to_delete.getParent().setLeft(node_to_delete.getLeft())
            node_to_delete.getLeft().setParent(node_to_delete.getParent())

        return_node = node_to_delete.getParent()

        # Disconnecting the node we deleted from the list
        node_to_delete.setParent(None)
        node_to_delete.setRight(None)
        node_to_delete.setLeft(None)

        return return_node

    """Auxiliary Function - deletes the node it gets
    
                    @rtype: AVLNode
                    @returns: the AVLNode from which the node has physically changed
    
                """
    def deleteNodeWithTwoChildren(self, node_to_delete):  # Complexity - O(logn)
        # Finding the successor in order to replace the deleted node by its successor
        successor_to_delete = self.successor(node_to_delete)
        # Deleting the successor
        if successor_to_delete.getSize() == 1:
            parent_of_successor = self.deleteLeaf(successor_to_delete)
        else:
            parent_of_successor = self.deleteNodeWithOneChild(successor_to_delete)

        # Replacing pointers in order to "put" the successor in the place of the deleted node
        if self.isRightChild(node_to_delete):
            node_to_delete.getParent().setRight(successor_to_delete)
        else:
            node_to_delete.getParent().setLeft(successor_to_delete)
        successor_to_delete.setParent(node_to_delete.getParent())

        successor_to_delete.setRight(node_to_delete.getRight())
        node_to_delete.getRight().setParent(successor_to_delete)
        successor_to_delete.setLeft(node_to_delete.getLeft())
        node_to_delete.getLeft().setParent(successor_to_delete)

        # Edge case - if the parent of the successor is the node we deleted
        if parent_of_successor == node_to_delete:
            parent_of_successor = successor_to_delete

        # If we deleted the root - fixing the root field
        if node_to_delete == self.getRoot():
            self.root = successor_to_delete

        # Disconnecting the node we deleted from the list
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

        # If two trees have the same height - the connecting node is the new root of the joined tree
        if self.getRoot().getHeight() == tree.getRoot().getHeight():
            connecting_node.setLeft(self.getRoot())
            connecting_node.setRight(tree.getRoot())
            if not (self.empty() and tree.empty()):
                connecting_node.setParent(self.root.getParent())
                if self.isRightChild(self.root):
                    self.root.getParent().setRight(connecting_node)
                else:
                    self.root.getParent().setLeft(connecting_node)

            # If the two lists are empty the joined list contains only the connecting node
            else:
                virtual = AVLNode("None")
                connecting_node.setParent(virtual)
                virtual.setLeft(connecting_node)
            self.getRoot().setParent(connecting_node)
            tree.getRoot().setParent(connecting_node)

            # Fixing the root field and the root's fields
            self.root = connecting_node
            self.updateSize(self.getRoot())
            self.updateHeight(self.getRoot())

        # If the trees do not have same height
        else:
            # If self is higher than tree
            if self.getRoot().getHeight() > tree.getRoot().getHeight():
                lower_tree_height = tree.getRoot().getHeight()

                # Linking connecting_node to shorter tree
                connecting_node.setRight(tree.getRoot())
                tree.getRoot().setParent(connecting_node)

                # Finding the left child of connecting_node
                curr_node = self.getRoot()
                while curr_node.getHeight() > lower_tree_height :
                    curr_node = curr_node.getRight()

                # Linking connecting_node to higher tree
                connecting_node.setParent(curr_node.getParent())
                curr_node.getParent().setRight(connecting_node)
                connecting_node.setLeft(curr_node)
                curr_node.setParent(connecting_node)

            # If tree is higher than self
            else:
                lower_tree_height = self.getRoot().getHeight()
                tree_is_bigger = True

                # Linking connecting_node to shorter tree
                connecting_node.setLeft(self.getRoot())
                self.getRoot().setParent(connecting_node)

                # Finding the right child of connecting_node
                curr_node = tree.getRoot()
                while curr_node.getHeight() > lower_tree_height:
                    curr_node = curr_node.getLeft()

                # Linking connecting_node to higher tree
                connecting_node.setParent(curr_node.getParent())
                curr_node.getParent().setLeft(connecting_node)
                connecting_node.setRight(curr_node)
                curr_node.setParent(connecting_node)

            # Fixing the fields of connecting node
            self.updateSize(connecting_node)
            self.updateHeight(connecting_node)

            parent_node = connecting_node.getParent()
            # A loop for fixing fields and rotating if needed, starting from the parent of the connecting node
            while parent_node.isRealNode():
                self.updateSize(parent_node)
                bf_parent_node = self.balanceFactor(parent_node)
                old_height = parent_node.getHeight()
                curr_new_height = max(parent_node.getLeft().getHeight(), parent_node.getRight().getHeight()) + 1

                # If the height hasn't changed and the BF is ok - terminate
                if abs(bf_parent_node) < 2 and curr_new_height == old_height:
                    break

                # If the height has changed and the BF is ok - fix the height and continue to parent
                elif abs(bf_parent_node) < 2:
                    parent_node.setHeight(curr_new_height)
                    parent_node = parent_node.parent

                # If the BF is not ok - rotate and fix fields, then continue to the next relevant node
                elif abs(bf_parent_node) == 2:
                    if tree_is_bigger:
                        tree.rotateAndFix(parent_node)
                    else:
                        self.rotateAndFix(parent_node)
                    parent_node = parent_node.parent.parent

            # Continue fixing fields until we reach the root
            while parent_node.isRealNode():
                self.updateSize(parent_node)
                parent_node = parent_node.parent

            # If tree is higher than self - fixing self's root to be tree's root
            if tree_is_bigger:
                self.root = tree.root

