funcs = dict()


def register(name):
    def decorator(func):
        funcs[name or func.__name__] = func
        return func

    if callable(name) and hasattr(name, "__name__"):
        name, f = None, name
        # noinspection PyTypeChecker
        decorator(f)
    else:
        return decorator


@register
def hello():
    print("hello")


@register("goodbye")
def bye():
    print("bye")


if __name__ == '__main__':
    print(funcs)
