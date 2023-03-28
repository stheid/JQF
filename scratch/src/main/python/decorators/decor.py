import functools

funcs = dict()


def register(name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            print(res)
            return res

        funcs[name or func.__name__] = wrapper

    if callable(name) and hasattr(name, "__name__"):
        name, f = None, name
        # noinspection PyTypeChecker
        decorator(f)
    else:
        return decorator


@register
def hello():
    print("hello")
    return "hi"


@register("goodbye")
def bye():
    print("bye")
    return "stuff"


if __name__ == '__main__':
    print(funcs)
    funcs["goodbye"]()
    funcs["hello"]()
