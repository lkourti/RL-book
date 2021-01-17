import numpy as np

def get_transition_matrix_nt(n: int) -> np.ndarray:
    transition_matrix_nt = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            transition_matrix_nt[i,j] = 1 / (n-i)
    return transition_matrix_nt

def solve_frog_puzzle(n: int) -> float:
    transition_matrix_nt = get_transition_matrix_nt(n)
    q = np.identity(n) - transition_matrix_nt
    ones = np.ones(n)
    t = np.linalg.solve(q, ones)
    return round(t[0], 2)

if __name__ == '__main__':
    n = 20
    num_jumps = solve_frog_puzzle(n)
    output = "For n = {}, the expected number of jumps is {}.".format(n, num_jumps)
    print(output)