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
    problem.objective.set_linear(linear_part)


    quadratic_part = []
    for row in mtr_Q:
        quadratic_part.append([x_names, list(2.0*row)])

    problem.objective.set_quadratic(quadratic_part)

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

    x, val = solve_MIQP(mtr_Q, vec_c, c0)

    vec_c = np.matrix(vec_c)
    res = x.dot(mtr_Q.dot(x)) + float(vec_c.dot(x)) + c0
    print("%8.2f - matrix multiplication"%res)
    print("%8.2f - cplex result"%val)
    print(" difference = %4.2f"%(100*abs(res-val)/abs(res))+'%')
    print(x)


if __name__ == "__main__":
    __test_solver()