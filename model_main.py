import codecs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from solver import solve_MIQP


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

    # Time series of generated supply, delivery volumes, backlog, warehouse inventories, and demand
    S_t = []
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
            assert(adj.shape == (self.S + self.W + self.D, self.S + self.W + self.D))
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

        self.S_t = pd.Series([[0.0] * self.S])
        self.x_t = pd.Series([[0.0] * self.n_edges])
        self.y_t = pd.Series([[0.0] * self.W])
        self.b_t = pd.Series([[0.0] * self.D])
        self.D_t = pd.Series([[0.0] * self.D])
        

    def __get_edges_number(self):
        res = 0
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                if self.mtr_adj[i][j] > 0:
                    res += 1
        return res


    def simulate(self, n_iter):
        print("Starting simulation process.")
        for _ in range(n_iter):
            self.make_iteration()
        print("Simulation done.")


    def make_iteration(self):
        self.t += 1
        self.__generate_supply_demand()
        mtr_Q, vec_c, c0 = self.__make_QCc()
        const_S, const_W = self.__make_constr()

        print("t = %d"%self.t, end=', ')
        x, val = solve_MIQP(mtr_Q, vec_c, c0, const_S, const_W)
        self.x_t = self.x_t.append(pd.Series([list(x)]), ignore_index=True)
        self.__update_y_b()
        print("Cost = %6.2f"%val)


    def __generate_supply_demand(self):
        _D, _S = [], []
        mu, sigma = 10.0, 3.0
        for _ in range(self.D):
            _D.append(max(np.random.normal(mu, sigma), 0.0))
        for _ in range(self.S):
            _S.append(max(np.random.normal(mu, sigma), 0.0))
        self.S_t = self.S_t.append(pd.Series([_S]), ignore_index=True)
        self.D_t = self.D_t.append(pd.Series([_D]), ignore_index=True)


    def __update_y_b(self):
        # Recalculate backlog and warehouse inventories after optimization
        _y = list(self.y_t.at[self.t-1])
        for w in range(self.S, self.S+self.W):
            for i in range(self.S):
                if self.mtr_adj[i, w] > 0:
                    a = (i, w)
                    k = self.x_node_pairs.index(a)
                    # Insert y_tau here ## y(t-tau+1)
                    _y[w-self.S] += self.x_t.at[self.t-1][k]
            for j in range(self.S+self.W, self.n_nodes):
                if self.mtr_adj[w, j] > 0:
                    a = (w, j)
                    k = self.x_node_pairs.index(a)
                    _y[w-self.S] -= self.x_t.at[self.t][k]
        self.y_t = self.y_t.append(pd.Series([_y]), ignore_index=True)

        _b = list(self.b_t.at[self.t-1])
        for j in range(self.S+self.W, self.S+self.W+ self.D):
            _b[j-self.S-self.W] += self.D_t.at[self.t][j-self.S-self.W]
            for i in range(self.S+self.W):
                if self.mtr_adj[i, j] > 0:
                    a = (i, j)
                    k = self.x_node_pairs.index(a)
                    _b[j-self.S-self.W] -= self.x_t.at[self.t][k]
        self.b_t = self.b_t.append(pd.Series([_b]), ignore_index=True)


    def __make_QCc(self):
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

        for i, j in self.x_node_pairs:
            c_ij = self.mtr_adj[i][j]
            # Linear part from transportation costs
            vec_c.append(c_ij * self.V_costs)

        for i in self.distributors:
            c0 += self.B_costs * ((self.b_t.at[t-1][i - S - W] + self.D_t.at[t][i - S - W])**2)
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
                vec_c[k] -= 2 * self.B_costs * (self.b_t.at[t-1][i-S-W] + self.D_t.at[t][i-S-W])
            
            for u in routes_i:
                for v in routes_i:
                    # Quadratic part from backlog costs (b(t) equations)
                    mtr_Q[u][v] = self.B_costs

        for w in self.warehouses:
            # Insert y_tau here
            c0 += self.h_costs[w-S] * self.y_t.at[t-1][w-S]
            for i in self.suppliers:
                if self.mtr_adj[w][i] != 0:
                    a = (i, w)
                    k = self.x_node_pairs.index(a)
                    c0 += self.h_costs[w-S] * self.x_t.at[t-1][k]

            for j in self.distributors:
                if self.mtr_adj[w, j] != 0:
                    a = (w, j)
                    k = self.x_node_pairs.index(a)
                    vec_c[k] -= self.h_costs[w-S]

        return mtr_Q, vec_c, c0


    def __make_constr(self):
        # All produced supply must be shipped
        mtr_S = np.zeros((self.S, self.n_edges))
        k = 0
        for i in range(self.S):
            for j in range(self.S, self.n_nodes):
                if self.mtr_adj[i, j] > 0:
                    mtr_S[i, k] = 1
                    k += 1
        vec_S = self.S_t.at[self.t]

        # Warehouse inventories cannot be negative
        mtr_W = np.zeros((self.W, self.n_edges))
        for i in range(self.W):
            for j in range(self.S + self.W, self.n_nodes):
                if self.mtr_adj[self.S+i, j] > 0:
                    mtr_W[i, k] = 1
                    k += 1
        vec_W = [0.0]*self.W
        for w in range(self.W):
            # Insert y_tau here
            vec_W[w] += self.y_t.at[self.t-1][w]
            for i in range(self.S):
                if self.mtr_adj[i, self.S+w] > 0:
                    a = (i, self.S+w)
                    k = self.x_node_pairs.index(a)
                    vec_W[w] += self.x_t.at[self.t-1][k]
        return (mtr_S, vec_S), (mtr_W, vec_W)

    ###############################################################################
    #   Printing / Writing

    def write_history(self):
        self.__write_deliveries()
        self.__write_supply_demand()
        self.__write_inventory_backlog()


    def __write_deliveries(self):
        with codecs.open('history_xt.dat', 'w') as fout:
            fout.write("Variables = t, ")
            for i in range(self.n_edges):
                fout.write("x_%d_%d"%(self.x_node_pairs[i]))
                if i+1 < self.n_edges:
                    fout.write(", ")
                else:
                    fout.write("\n")

            for t in range(self.t+1):
                fout.write("%5d "%t)
                for i in range(self.n_edges):
                    fout.write("%7.2f "%self.x_t.at[t][i])
                fout.write("\n")


    def __write_supply_demand(self):
        with codecs.open('history_St_Dt.dat', 'w') as fout:
            fout.write("Variables = t, ")
            for i in range(self.S):
                fout.write("S_%d, "%i)
            for j in range(self.D):
                fout.write("D_%d"%j)
                if j+1 < self.D:
                    fout.write(", ")
                else:
                    fout.write("\n")

            for t in range(self.t+1):
                fout.write("%5d "%t)
                for i in range(self.S):
                    fout.write("%7.2f "%self.S_t.at[t][i])
                for j in range(self.D):
                    fout.write("%7.2f "%self.D_t.at[t][j])
                fout.write("\n")                


    def __write_inventory_backlog(self):
        with codecs.open('history_bt_yt.dat', 'w') as fout:
            fout.write("Variables = t, ")
            for w in range(self.W):
                fout.write("y_%d, "%w)
            for j in range(self.D):
                fout.write("b_%d"%j)
                if j+1 < self.D:
                    fout.write(", ")
                else:
                    fout.write("\n")

            for t in range(self.t+1):
                fout.write("%5d "%t)
                for w in range(self.W):
                    fout.write("%7.2f "%self.y_t.at[t][w])
                for j in range(self.D):
                    fout.write("%7.2f "%self.b_t.at[t][j])
                fout.write("\n")       


    def draw_network(self):
        import networkx as nx
        G = nx.from_numpy_matrix(self.mtr_adj)
        elables = {}
        width = [G[u][v]['weight'] / 5. for u,v in G.edges()]
        for (u,v) in G.edges():
            elables[(u,v)] = str(G[u][v]['weight']) # "%d  (%d,%d)"%(G[u][v]['weight'], u, v)
        
        pos = self.__make_pos()

        suppliers = [i for i in range(self.S)]
        warehouses = [i for i in range(self.S, self.S + self.W)]
        distributors = [i for i in range(self.S + self.W, self.S + self.W + self.D)]

        nx.draw_networkx_nodes(G, pos=pos, nodelist=suppliers,    node_size=1000, node_color='C0', node_shape='s')
        nx.draw_networkx_nodes(G, pos=pos, nodelist=warehouses,   node_size=1000, node_color='C1', node_shape='p')
        nx.draw_networkx_nodes(G, pos=pos, nodelist=distributors, node_size=1000, node_color='C2', node_shape='o')

        nx.draw_networkx_edges(G, pos=pos, node_color='C0', width=width)
        nx.draw_networkx_labels(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=elables, label_pos=0.8)
        plt.axis('off')
        plt.show()


    def draw_current_timestep(self):
        import networkx as nx
        G = nx.from_numpy_matrix(self.mtr_adj)
        elables = {}
        width = [G[u][v]['weight'] / 5. for u,v in G.edges()]
        for (u,v) in G.edges():
            k = self.x_node_pairs.index((u,v))
            elables[(u,v)] = "%4.2f"%(self.x_t.at[self.t][k])

        node_labels = {}
        for i in range(self.S):
            node_labels[i] = "%4.2f"%self.S_t.at[self.t][i]
        for w in range(self.S, self.S+self.W):
            node_labels[w] = "%4.2f\n%4.2f"%(self.y_t.at[self.t-1][w-self.S], self.y_t.at[self.t][w-self.S])
        for j in range(self.S+self.W, self.S+self.W+self.D):
            node_labels[j] = "[%3.1f]\n%4.2f\n%4.2f"%(self.D_t.at[self.t][j-self.S-self.W], 
                self.b_t.at[self.t-1][j-self.S-self.W], self.b_t.at[self.t][j-self.S-self.W])

        pos = self.__make_pos()

        suppliers = [i for i in range(self.S)]
        warehouses = [i for i in range(self.S, self.S + self.W)]
        distributors = [i for i in range(self.S + self.W, self.S + self.W + self.D)]

        nx.draw_networkx_nodes(G, pos=pos, nodelist=suppliers,    node_size=2000, node_color='C0', node_shape='s')
        nx.draw_networkx_nodes(G, pos=pos, nodelist=warehouses,   node_size=2000, node_color='C1', node_shape='p')
        nx.draw_networkx_nodes(G, pos=pos, nodelist=distributors, node_size=2000, node_color='C2', node_shape='o')

        nx.draw_networkx_edges(G, pos=pos, node_color='C0', width=width)
        nx.draw_networkx_labels(G, pos=pos, labels=node_labels)
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=elables, label_pos=0.75)
        plt.axis('off')
        plt.show()
            

    def __make_pos(self):
        swd = [self.S, self.W, self.D]
        n_max = max(swd)
        l_max = n_max - 1
        pos = {}
        k = 0
        x = 0.
        for j in range(3):
            n = swd[j]
            h_shift = 0.
            
            if n <= n_max-2:
                h_x = l_max / (n + 1)
                h_shift = h_x 

            elif n == n_max-1:
                h_x = l_max / n
                h_shift = 0.5
            else:
                h_x = l_max / (n - 1)
            if j == 1:
                h_shift -= h_x / 3

            for i in range(n):
                pos[k] = [x, l_max / 2 - h_shift - h_x*i]
                k += 1
            x += 1.
        return pos

        



np.random.seed(1)
A = HendriksArmbrusterSimulator()
A.simulate(5)
# A.draw_current_timestep()
# A.draw_network()
A.write_history()
