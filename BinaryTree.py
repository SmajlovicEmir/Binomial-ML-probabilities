import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
import itertools


class Node:

    def __init__(self, previous_nodes=None, next_nodes=None, data=None):
        if previous_nodes is not None:
            self.previous_nodes = previous_nodes
        else:
            self.previous_nodes = []
        if next_nodes is not None:
            self.next_nodes = next_nodes
        else:
            self.next_nodes = []
        self.data = data
        self.coo = ()

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def get_previous_nodes(self):
        return self.previous_nodes

    def set_previous_nodes(self, nodes):
        self.previous_nodes.append(nodes)

    def get_next_nodes(self):
        return self.next_nodes

    def set_next_nodes(self, nodes):
        self.next_nodes.append(nodes)


class BinaryTree:
    def __init__(self, depth=10):
        self.tree = []
        self.depth = depth

    def create_tree(self):
        for i in range(1, self.depth + 1):
            self.tree.append([Node(data=i)] * (1 << i))
        len_all_nodes = ([len(x) for x in self.tree])
        for i in range(1, len(len_all_nodes)):
            counter = 0
            for j in range(len_all_nodes[i]):
                if j % 2 == 0 and j > 0:
                    counter = counter + 1
                if i > 0:
                    self.tree[i - 1][counter].set_next_nodes(self.tree[i][j])


class BinomialTree(BinaryTree):

    def __init__(self, depth=10):
        super().__init__(depth)
        self.flat_tree = []
        self.print_coo = []
        self.sorted_coo = []
        self.create_tree()
        self.set_plot_coo()
        self.row_coo = []
        self.col_coo = []
        self.y_coo_sorted=[]

    def create_tree(self):
        print("create tree called")
        self.tree.append([Node(data=0)])
        for s in range(1, self.depth):
            self.tree.append([])
            for j in range(s + 1):
                self.tree[s].append(Node(data=s))
        len_all_nodes = ([len(x) for x in self.tree])
        for k in range(len(len_all_nodes) - 1):
            for m in range(len_all_nodes[k]):
                self.tree[k][m].set_next_nodes(self.tree[k + 1][m])
                self.tree[k][m].set_next_nodes(self.tree[k + 1][m + 1])
        self.flat_tree = itertools.chain(*self.tree)
        self.flat_tree = list(self.flat_tree)

    def set_tree_node(self, data, i, j):
        self.tree[i][j].set_data(data)

    def get_tree_node(self, i, j):
        return self.tree[i][j]

    def set_plot_coo(self):
        number_of_rows = self.depth * 2 - 1
        for i in range(number_of_rows):
            count = 0
            bound = 0
            for j in range(self.depth):
                x = i if (number_of_rows - self.depth - bound) <= i <= (number_of_rows - self.depth + bound) else " "
                bound += 1
                if x != " ":
                    x = (i + j + self.depth) % 2
                    if x == 1:
                        self.print_coo.append((j, i))
                        count += 1
                if x != 1:
                    x = " "

    def set_up(self):
        self.row_coo = [coo[0] for coo in self.print_coo]
        self.col_coo = [coo[1] for coo in self.print_coo]
        self.y_coo_sorted = np.sort(np.array(self.col_coo))
        temp_row_coo = []

        for x in self.row_coo:
            temp_row_coo.append(x)

        sort_for_mesh = sorted(zip(temp_row_coo, self.y_coo_sorted))
        self.sorted_coo = sort_for_mesh
        for node, coo in zip(self.flat_tree, self.sorted_coo):
            node.coo = coo

    def plot_tree(self):
        self.set_up()
        plt.plot(np.array(self.row_coo), self.y_coo_sorted, "ob")
        plt.yticks(np.arange(self.depth * 2 - 1))
        plt.xticks(np.arange(self.depth))

        temp_row_coo = []

        for x in self.row_coo:
            temp_row_coo.append(x)

        sort_for_plot = sorted(zip(temp_row_coo, self.y_coo_sorted))
        self.sorted_coo = sort_for_plot
        for i, j in zip(sort_for_plot, self.flat_tree):
            plt.annotate(j.get_data(), (i[0] - .05, i[1] + .25))

        for x in range(len(self.tree) - 1):
            for y in range(x + 1):
                alpha1 = binom(x, y) / 2**x
                alpha2 = binom(x, y + 1) / 2**x
                plt.plot([self.tree[x][y].coo[0], self.tree[x + 1][y].coo[0]],
                         [self.tree[x][y].coo[1], self.tree[x + 1][y].coo[1]], "k-")
                plt.plot([self.tree[x][y].coo[0], self.tree[x + 1][y + 1].coo[0]],
                         [self.tree[x][y].coo[1], self.tree[x + 1][y + 1].coo[1]], "k-")

        plt.show()

    def print_tree(self):
        self.set_up()
        mesh = np.full((len(self.tree), 2 * len(self.tree) - 1), " ")
        for xy, z in zip(self.sorted_coo, self.flat_tree):
            mesh[xy[0]][xy[1]] = z.get_data()

        mesh = mesh.T
        print(mesh)



