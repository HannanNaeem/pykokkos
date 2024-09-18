import pykokkos as pk
from mypy import api
import mypy.main as main
import mypy.build as build
from mypy.visitor import NodeVisitor

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

def yAx(j: int, acc: int, cols: int, y_view: List, x_view: List, A_view: List):
    temp2 = 0
    for i in range(cols):
        temp2 += A_view[j * cols + i] * x_view[i]

    acc += y_view[j] * temp2
"""
example_str = """

def yAx(j: int):
    x = 2
    reveal_type(x)
    return x + j

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

    # run type checking using MyPy Api
    command = ["-c", example_str]
    result = api.run(command)
    print(result[0]) # Normal report
    print(result[1]) # Error report
    print(result[2]) # Exit Status

    # we actually want to be able to infer a particular variable
    build_source, options = main.process_options(["-c", example_str])
    options.preserve_asts = True
    options.fine_grained_incremental = True
    options.ignore_missing_imports = True
    options.export_types = True

    result = build.build(build_source, options)
    print(result.graph['__main__'].tree.names['yAx'].node.body.body)
    print()
    # visitor = type_visitor.TypeVisitor
    # visitor.visit(result.graph['__main__'].tree.names['yAx'].node)

    # print(result.types[result.graph['__main__'].tree.names['yAx'].node])
    # print(result.graph['__main__'].tree.names['yAx'].node.body.body)
    for node in result.graph['__main__'].tree.names['yAx'].node.body.body:
        print(dir(node))
        print(node)
        for val in node.lvalues:
            print(val)
            try:
                print(result.types[val])
            except:
                print("Failed to find result")
            break


if __name__ == "__main__":
    infer_types()
    # run()
