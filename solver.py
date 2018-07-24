import cplex
import numpy as np



def make_names(n):
    if n < 10:
        names = ['x_'+str(i) for i in range(n)]
    elif n < 100:
        names = ['x_0'+str(i) for i in range(10)] + ['x_'+str(i) for i in range(10, n)]
    elif n < 1000:
        names = ['x_00'+str(i) for i in range(10)] + ['x_0'+str(i) for i in range(10, 100)] \
                + ['x_'+str(i) for i in range(100, n)]
    elif n < 10000:
        names = ['x_000'+str(i) for i in range(10)] + ['x_00'+str(i) for i in range(10, 100)] \
                + ['x_0'+str(i) for i in range(100, 1000)] + ['x_'+str(i) for i in range(1000, n)]
    
    return names



def solve_MIQP(mtr_Q, vec_c, c0):
    n = len(vec_c)
    x_names = make_names(n)
    
    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.variables.add(names = x_names, lb=[0.0]*n, ub=[cplex.infinity]*n)

    linear_part = []
    for _x, _c in zip(x_names, vec_c):
        linear_part.append((_x, _c))
    # print(linear_part)
    problem.objective.set_linear(linear_part)

    Qc = np.array(mtr_Q, dtype=float)
    for i in range(n):
        Qc[i][i] *= 2
    quadratic_part = []
    for row in Qc:
        quadratic_part.append([x_names, list(row)])
    # print(quadratic_part)

    problem.objective.set_quadratic(quadratic_part)

    problem.write("MIQP_output.lp")

    print("-"*80)
    problem.solve()

    # print("-"*80)
    # # solution.get_status() returns an integer code
    # print("Solution status = %d:"%problem.solution.get_status(), end=' ')
    # # the following line prints the corresponding string
    # print(problem.solution.status[problem.solution.get_status()])
    # print("Solution value  = ", problem.solution.get_objective_value() )# + c0)

    # print("-"*80)


    # print("-"*80)
    # numrows = problem.linear_constraints.get_num()
    # numcols = problem.variables.get_num()
    # print(numrows, numcols)

    # slack = problem.solution.get_linear_slacks()
    # pi = problem.solution.get_dual_values()
    # x = problem.solution.get_values()
    # dj = problem.solution.get_reduced_costs()
    # for i in range(numrows):
    #     print("Row %d:  Slack = %10f  Pi = %10f" % (i, slack[i], pi[i]))
    # for j in range(numcols):
    #     print("Column %d:  Value = %10f Reduced cost = %10f" % (j, x[j], dj[j]))
    
    x = problem.solution.get_values()
    return np.array(x), problem.solution.get_objective_value()+c0

    


# mtr_Q = np.array([
#     [1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 20, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 10, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1]
#     ], dtype=float)

# vec_c = [10.0, 10.0, -123.4, -153.3, -164.8, -123.4, -153.3, -168.3, -126.9] #[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0]

# c0 = 3313.78

# print(mtr_Q)
# print(vec_c)
# x = solve_MIQP(mtr_Q, vec_c, c0)

# vec_c = np.matrix(vec_c)
# res = x.dot(mtr_Q.dot(x)) + float(vec_c.dot(x)) #+ c0
# print(res)
# print(x)


