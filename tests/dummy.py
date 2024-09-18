import pykokkos as pk
from mypy import api

@pk.workunit
def y_init(i, y_view):
    y_view[i] = 1

@pk.workunit
def matrix_init(j, cols, A_view):
    for i in range(cols):
        A_view[j * cols + i] = 1

@pk.workunit
def yAx(j, acc, cols, y_view, x_view, A_view):
    temp2 = 0
    reveal_type(temp2)
    for i in range(cols):
        temp2 += A_view[j * cols + i] * x_view[i]

    acc += y_view[j] * temp2

workunit_str = """
def yAx(j, acc, cols, y_view, x_view, A_view):
    temp2 = 0
    reveal_type(temp2)
    for i in range(cols):
        temp2 += A_view[j * cols + i] * x_view[i]

    acc += y_view[j] * temp2
"""

def run() -> None:
    N: int = 5# Rows
    M: int = 5 # Cols
    nrepeat: int = 1 
    print(f"Total size S = {N * M} N = {N} M = {M}")

    y = pk.View([N], pk.double)
    x = pk.View([M], pk.double)
    A = pk.View([N * M], pk.double)

    p = pk.RangePolicy(0, N)
    pk.parallel_for(p, y_init, y_view=y)
    pk.parallel_for(pk.RangePolicy(0, M), y_init, y_view=x)
    pk.parallel_for(p, matrix_init, cols=M, A_view=A)

    timer = pk.Timer()

    for i in range(nrepeat):
        result = pk.parallel_reduce(p, yAx, cols=M, y_view=y, x_view=x, A_view=A)

    timer_result = timer.seconds()

    print(f"Computed result for {N} x {M} is {result}")
    solution = N * M

    if result != solution:
        pk.printf("Error: result (%lf) != solution (%lf)\n", result, solution)

    print(f"N({N}) M({M}) nrepeat({nrepeat}) problem(MB) time({timer_result}) bandwidth(GB/s)")


def infer_types() -> None:
    command = ["-c", workunit_str]
    result = api.run(command)
    print(result[0])
    print(result[1])
    print(result[2])

if __name__ == "__main__":
    infer_types()
    # run()
