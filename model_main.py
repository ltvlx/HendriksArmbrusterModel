import codecs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class HendriksArmbrusterSimulator:
    # V_costs -- price for one truck of capacity V
    # B_costs -- cost coefficient for backlog
    # h_costs -- cost coefficient for warehousing
    V_costs = 0.0
    B_costs = 0.0
    h_costs =  []

    # S - number of suppliers, W - number of warehouses, D - number of distributors
    S, W, D = 0, 0, 0
    # lists of their indexes
    suppliers    = []
    warehouses   = []
    distributors = []
    # number of nodes and edges in network
    n_nodes = 0
    n_edges = 0
    # list of tuples with indexes of all edges
    x_node_pairs = []
    # adjacency matrix
    mtr_adj = None

    # Current timestep
    t = 0
    
    # Time series of delivery volumes, backlog, warehouse inventories, and demand
    x_t = []
    b_t = []
    y_t = []
    D_t = []

        
    def __init__(self):
        """
            During the initialization, setup is read from 'setup.txt' file.
            First row there contains S, W, D - numbers of suppliers, warehouses and distributors.
            Next S+W+D lines contain rows of adjecency matrix, which should be integers separated by whitespaces.

            Next, after converting these values and storing them, initial values of x_t, b_t, and y_t are generated.
        """
        with codecs.open('setup.txt', 'r') as finp:
            finp.readline() # S, W, D
            self.S, self.W, self.D = map(int, finp.readline().split())
            self.n_nodes = self.S + self.W + self.D
            self.suppliers = [i for i in range(self.S)]
            self.warehouses = [i for i in range(self.S, self.S + self.W)]
            self.distributors = [i for i in range(self.S + self.W, self.S + self.W + self.D)]

            finp.readline() # Adjecency matrix
            adj = []
            for _ in range(self.S + self.W + self.D):
                adj.append(list(map(int, finp.readline().split())))

            adj = np.array(adj, dtype=int)
            assert((adj.transpose() == adj).all())
            self.mtr_adj = adj

            for i in range(self.n_nodes):
                for j in range(i+1, self.n_nodes):
                    if self.mtr_adj[i][j] > 0:
                        self.n_edges += 1
                        self.x_node_pairs.append((i, j))

            finp.readline() # V delivery cost, B backlog cost
            self.V_costs, self.B_costs = map(float, finp.readline().split())

            finp.readline() # h inventory keeping costs; [0, ..., W-1]
            self.h_costs = list(map(float, finp.readline().split()))


        self.x_t = pd.Series([[0.0] * self.n_edges])
        self.y_t = pd.Series([[0.0] * self.W])
        self.b_t = pd.Series([[0.0] * self.D])
        self.D_t = pd.Series([[0.0] * self.D])

        # print(self.mtr_adj)
        # print(self.n_nodes, self.n_edges)
        # print(self.x_node_pairs)
        # print(self.V_costs, self.B_costs, self.h_costs)
        

    def __get_edges_number(self):
        res = 0
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                if self.mtr_adj[i][j] > 0:
                    res += 1
        return res


    def first_iteration_temp(self):
        # _x = pd.Series([[i for i in range(self.n_edges)]])
        self.t += 1
        _x = pd.Series([[4.5, 1.0, 1.9, 2.1, 0.5, 0.9, 2.3, 2.7, 1.4]])
        self.x_t = self.x_t.append(_x, ignore_index=True)

        self.y_t = self.y_t.append(pd.Series([[9.4]]), ignore_index=True)
        self.b_t = self.b_t.append(pd.Series([[1.1, 2.1, 3.1]]), ignore_index=True)
        self.D_t = self.D_t.append(pd.Series([[6.5, 3.7, 4.0]]), ignore_index=True)

        """
            # print(self.x_t.at[1])
            # print("x_t vector:")
            # print(self.x_t)
            # print("y_t vector:")
            # print(self.y_t)
            # print("b_t vector:")
            # print(self.b_t)
            # print("D_t vector:")
            # print(self.D_t)
        """

        mtr_Q, vec_c, c0 = self.make_QCc()

        from solver import solve_MIQP
        x, val = solve_MIQP(mtr_Q, vec_c, c0)

        print(mtr_Q)
        print(x, val)


    def make_QCc(self):
        """
            The goal of current subroutine is to create 
            matrix Q, vector C, and constant term c0 
            for the MIQP optimization problem, solved later.
            f(x) = x'.Q.x + C'.x + c0

            input: adjacency matrix; numbers of suppliers, warehouses and distributors
            return: Q, C

            mtr_adj -- weighted adjacency matrix of the network, 
                    weigts ~ geographical distance between nodes

                n_nodes -- number of nodes in this network
            x_node_pairs -- tuples (i, j) of integers for existing x_i,j routes
                n_edges -- number of those routes

            x -- variables in MIQP (not presented here); They basically reflect 
                    volumes of shipments via corresponing routes from the x_nodes
            c0 -- constant term in f(x)
            vec_c -- coefficients of linear terms in f(x)
            mtr_Q -- (default? -check if they can change) coefficients of quadratic terms in f(x)
        """
        t = self.t
        S = self.S
        W = self.W


        # resulting variables:
        c0 = 0
        vec_c = []
        mtr_Q = np.zeros((self.n_edges, self.n_edges), dtype=float)

        # for i in range(self.n_nodes):
        #     for j in range(i+1, self.n_nodes):
        for i, j in self.x_node_pairs:
            c_ij = self.mtr_adj[i][j]
            # Linear part from transportation costs
            vec_c.append(c_ij * self.V_costs)

        for i in self.distributors:
            c0 += self.B_costs * ((self.b_t.at[t][i - S - W] + self.D_t.at[t][i - S - W])**2)
            sup_nodes = []
            for j in range(self.n_nodes):
                w = self.mtr_adj[i][j]
                if w != 0:
                    sup_nodes.append(j)
            
            routes_i = []
            for j in sup_nodes:
                a = (j, i) # if j < i else (i, j) # not required as long as distributors are the last rows in mtr_adj
                k = self.x_node_pairs.index(a)
                routes_i.append(k)
                # Linear part from backlog costs (b(t) equations)
                vec_c[k] -= self.B_costs * (self.b_t.at[t][i-S-W] + self.D_t.at[t][i-S-W])
            
            for u in routes_i:
                for v in routes_i:
                    # Quadratic part from backlog costs (b(t) equations)
                    if u != v:
                        mtr_Q[u][v] = 0.5 * self.B_costs
                    else:
                        mtr_Q[u][v] = self.B_costs


        for w in self.warehouses:
            c0 += self.h_costs[w-S] * self.y_t.at[t][w-S]
            for i in self.suppliers:
                if self.mtr_adj[w][i] != 0:
                    a = (i, w)
                    k = self.x_node_pairs.index(a)
                    c0 += self.h_costs[w-S] * self.x_t.at[t][k]

            for j in self.distributors:
                if self.mtr_adj[w][j] != 0:
                    a = (w, j)
                    k = self.x_node_pairs.index(a)
                    vec_c[k] -= self.h_costs[w-S]

        return mtr_Q, vec_c, c0 # np.matrix(vec_c).transpose()


    def draw_network(self):
        import networkx as nx
        G = nx.from_numpy_matrix(self.mtr_adj)
        elables = {}
        width = [G[u][v]['weight'] / 5. for u,v in G.edges()]
        for (u,v) in G.edges():
            elables[(u,v)] = str(G[u][v]['weight']) # "%d  (%d,%d)"%(G[u][v]['weight'], u, v)
        
        pos = {0: [0.0,  0.0], 1: [0.0, -1.0], 2: [0.0, -2.0], 3: [1.0, -0.5], 4: [2.0,  0.0], 5: [2.0, -1.0], 6: [2.0, -2.0]}
        # pos = nx.spring_layout(G, weight='weight')

        suppliers = [i for i in range(self.S)]
        warehouses = [i for i in range(self.S, self.S + self.W)]
        distributors = [i for i in range(self.S + self.W, self.S + self.W + self.D)]

        nx.draw_networkx_nodes(G, pos=pos, nodelist=suppliers,    node_size=1000, node_color='C0')
        nx.draw_networkx_nodes(G, pos=pos, nodelist=warehouses,   node_size=1000, node_color='C1')
        nx.draw_networkx_nodes(G, pos=pos, nodelist=distributors, node_size=1000, node_color='C2')

        nx.draw_networkx_edges(G, pos=pos, node_color='C0', width=width)
        nx.draw_networkx_labels(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=elables, label_pos=0.8)
        plt.axis('off')
        plt.show()

            
A = HendriksArmbrusterSimulator()
A.first_iteration_temp()
# A.draw_network()
# mtr_Q, vec_c, c0 = A.make_QCc()

# print("Output Q, C, and c0:\n", mtr_Q, "\n", vec_c, "\n", c0)
