import functools


def compose(*functions):
    ''' evaluates functions from right to left.

        >>> add = lambda x, y: x + y
        >>> add3 = lambda x: x + 3
        >>> divide2 = lambda x: x/2
        >>> subtract4 = lambda x: x - 4
        >>> subtract1 = compose(add3, subtract4)
        >>> subtract1(1)
        0
        >>> compose(subtract1, add3)(4)
        6
        >>> compose(int, add3, add3, divide2)(4)
        8
        >>> compose(int, divide2, add3, add3)(4)
        5
        >>> compose(int, divide2, compose(add3, add3), add)(7, 3)
        8
    '''
    def inner(func1, func2):
        return lambda *x, **y: func1(func2(*x, **y))
    return functools.reduce(inner, functions)


def iter_doc(f, shuffled_indices):
    for i in shuffled_indices:
        for j, line in enumerate(f):
            if i == j:
                yield line
                f.seek(0)
                break
