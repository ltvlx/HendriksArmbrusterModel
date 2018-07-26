import cplex
import numpy as np



def make_names(n, key='x'):
    if n < 10:
        names = [key + '_'+str(i) for i in range(n)]
    elif n < 100:
        names = [key + '_0'+str(i) for i in range(10)] + [key + '_'+str(i) for i in range(10, n)]
    elif n < 1000:
        names = [key + '_00'+str(i) for i in range(10)] + [key + '_0'+str(i) for i in range(10, 100)] \
                + [key + '_'+str(i) for i in range(100, n)]
    elif n < 10000:
        names = [key + '_000'+str(i) for i in range(10)] + [key + '_00'+str(i) for i in range(10, 100)] \
                + [key + '_0'+str(i) for i in range(100, 1000)] + [key + '_'+str(i) for i in range(1000, n)]
    return names



def solve_MIQP(mtr_Q, vec_c, c0, const_S, const_W):
    n = len(vec_c)
    x_names = make_names(n)
    
    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.variables.add(names = x_names, lb=[0.0]*n, ub=[cplex.infinity]*n)

    linear_part = []
    for _x, _c in zip(x_names, vec_c):
        linear_part.append((_x, _c))
    problem.objective.set_linear(linear_part)


    quadratic_part = []
    for row in mtr_Q:
        quadratic_part.append([x_names, list(2.0*row)])
    problem.objective.set_quadratic(quadratic_part)

    # Make constraints
    constr_expr, constr_name, constr_rhs, constr_sens = [], [], [], ""
    S, W = len(const_S[1]), len(const_W[1])
    constr_name += make_names(S, 's')
    constr_expr += [[x_names, list(row)] for row in const_S[0]]
    constr_rhs += const_S[1]
    constr_sens += "E"*S
    constr_name += make_names(W, 'wh')
    constr_expr += [[x_names, list(row)] for row in const_W[0]]
    constr_rhs += const_W[1]
    constr_sens += "L"*W
    problem.linear_constraints.add(lin_expr=constr_expr, senses=constr_sens, rhs=constr_rhs, names=constr_name)

    problem.write("MIQP_output.lp")

    # problem.set_log_stream(None)
    problem.set_results_stream(None)
    problem.solve()

    print("MIQP solved, status:", problem.solution.status[problem.solution.get_status()])
    x = np.array(problem.solution.get_values())
    res = problem.solution.get_objective_value() + c0
    return x, res


def __test_solver():
    mtr_Q = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=float)
    vec_c = [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0, -90.0, -100.0]

    c0 = 3313.78

    const_S = (np.array([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.]
                        ]), 
               [100., 200.])
    
    const_W = (np.array([[0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]]), [100.0])

    x, val = solve_MIQP(mtr_Q, vec_c, c0, const_S, const_W)

    vec_c = np.matrix(vec_c)
    res = x.dot(mtr_Q.dot(x)) + float(vec_c.dot(x)) + c0
    print("%8.2f - matrix multiplication"%res)
    print("%8.2f - cplex result"%val)
    print(" difference = %4.2f"%(100*abs(res-val)/abs(res))+'%')
    print(x)


if __name__ == "__main__":
    __test_solver()