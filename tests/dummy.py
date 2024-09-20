import pykokkos as pk
from mypy import api
import mypy.main as main
import mypy.build as build
from mypy.lookup import lookup_fully_qualified
from mypy.nodes import AssignmentStmt

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
    temp3 = j + 1.0
    for i in range(cols):
        temp2 += A_view[j * cols + i] * x_view[i]

    acc += y_view[j] * temp2
"""
example_str = """

def yAx(j: int):
    x = 2
    x + j
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
    command = ["-c", workunit_str]
    result = api.run(command)
    print(result[0]) # Normal report
    print(result[1]) # Error report
    print(result[2]) # Exit Status

    # we actually want to be able to infer a particular variable
    build_source, options = main.process_options(["-c", workunit_str])
    # don't let the AST flush/clear
    options.preserve_asts = True
    # don't cache
    options.fine_grained_incremental = True
    options.ignore_missing_imports = True
    # export entire dict of inferred types: dict[parse_tree_nodes, types]
    # the keys are always NameExpr nodes or Func etc but never stmt node
    options.export_types = True
    # visit unannotated nodes as well
    options.check_untyped_defs = True

    result = build.build(build_source, options)

    # below is the function body node
    # print(result.graph['__main__'].tree.names['yAx'].node.body.body)

    # print()

    # TODO 
    # Lets try to harness the built in visitor(s)
    # visitor = type_visitor.TypeVisitor
    # visitor.visit(result.graph['__main__'].tree.names['yAx'].node)

    # A workunit should be a function type
    print(result.graph['__main__'].tree.names['yAx'].type)
    # print the tree
    print(result.graph['__main__'].tree.names['yAx'].node.body)
    # printing out type for the first and only lvalue for the first Stmt (assignment stmt) in body
    # print(result.types[result.graph['__main__'].tree.names['yAx'].node.body.body[0].lvalues[0]])


    # for now lets say we are interested in Assignment statements (that can be definitions/redefinitions)
    # we want to get those annotations.
    print("\nInferred types:")
    for node in result.graph['__main__'].tree.names['yAx'].node.body.body:
        # We can also get the node from the fully qualified name (but this is a SymbolTableNode)
        # print(lookup_fully_qualified("__main__.yAx", result.files, raise_on_missing=True).node)

        if isinstance(node, AssignmentStmt):
            # get type
            print(node.lvalues[0].name, "->", result.types[node.lvalues[0]])

    # For pykokkos we want:
    """
        - Annotations in the args. This serves as an anchor to propagate type information down
            - These annotations must be copied over from pykokkos parse tree to mypy parse tree
            - Or we regenerate python source code for workunit with the args_type_inference and feed
                it to Mypy
        - Node visitor to visit assignment/relevant statements
        - The main problem here is this seems wasteful. Why maintain two different ASTs just to infer types,
            and copy the date in-between them.
        - It might be worth, just doing simple static analysis on the Pykokkos parse tree itself - given the
            the limited flexibility of calls we can make inside a workunit.
    """

if __name__ == "__main__":
    infer_types()
    # run()
