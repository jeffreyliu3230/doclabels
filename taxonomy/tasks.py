from invoke import task


@task
def construct_taxonomy():
    from taxonomy import construct_taxonomy
    construct_taxonomy()


@task
def get_roots():
    from taxonomy import get_roots
    get_roots()


@task
def get_children(root_id):
    from taxonomy import get_children
    print(get_children(root_id))


@task
def grow_tree():
    from taxonomy import grow_tree
    grow_tree()


@task
def polish():
    from taxonomy import polish
    polish()
