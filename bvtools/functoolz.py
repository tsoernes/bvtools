import inspect
import operator
import re
from functools import partial, reduce, wraps
from multiprocessing.pool import ThreadPool
from typing import Callable, Iterable, List, Set, TypeVar

V = TypeVar("V")


def curried(func: Callable) -> Callable:
    """Allows currying"""
    sig = inspect.signature(func)

    @wraps(func)
    def inner(*args, **kwargs):
        bind = sig.bind_partial(*args, **kwargs)
        for param in sig.parameters.values():
            if param.name not in bind.arguments and param.default is param.empty:
                # Some required arguments are missing; return a partial
                return partial(func, *args, **kwargs)
        return func(*args, **kwargs)

    return inner


_threadpools = dict()  # A threadpool for each function decorated with `allow_pooled_async` that has been called


def allow_pooled_async(threads: int = 1) -> Callable:
    """
    Allows threaded execution. Adds an `async_pool` parameter to the function.
    Tasks get queued with max `threads` running at once.

    Decorating a function with `allow_pooled_async` and then calling it with
    `async_pool=True` will make it return an multiprocessing.pool.AsyncResult object.
    """

    def deco(func: Callable) -> Callable:
        old_sig = inspect.signature(func)
        params = list(old_sig.parameters.values())
        param = inspect.Parameter(
            "async_pool", inspect.Parameter.KEYWORD_ONLY, default=False, annotation=bool
        )
        params.insert(len(params), param)
        sig = old_sig.replace(parameters=params)

        @wraps(func)
        def inner(*args, **kwargs):
            bind = sig.bind_partial(*args, **kwargs)
            bind.apply_defaults()
            if bind.arguments["async_pool"]:
                del bind.arguments["async_pool"]
                if func.__name__ in _threadpools:
                    pool = _threadpools[func.__name__]
                else:
                    pool = ThreadPool(threads)
                    threadpools[func.__name__] = pool
                return pool.apply_async(func, args=args, kwds=kwargs)
            else:
                return func(*args, **kwargs)

        inner.__signature__ = sig
        return inner

    return deco


def allow_async(func: Callable) -> Callable:
    """
    Allows threaded execution. Adds an `async_` parameter to the function. Decorating a function
    with `allow_async` and then calling it with `async_=True` will make it return an
    multiprocessing.pool.AsyncResult object.
    """
    old_sig = inspect.signature(func)
    params = list(old_sig.parameters.values())
    param = inspect.Parameter(
        "async_", inspect.Parameter.KEYWORD_ONLY, default=False, annotation=bool
    )
    params.insert(len(params), param)
    sig = old_sig.replace(parameters=params)

    @wraps(func)
    def inner(*args, **kwargs):
        bind = sig.bind_partial(*args, **kwargs)
        bind.apply_defaults()
        if bind.arguments["async_"]:
            del bind.arguments["async_"]
            pool = ThreadPool(1)
            return pool.apply_async(func, kwds=bind.arguments)
        else:
            return func(*args, **kwargs)

    inner.__signature__ = sig
    return inner


eq = curried(operator.eq)
lt = curried(operator.lt)
le = curried(operator.le)
gt = curried(operator.gt)
ge = curried(operator.ge)
neq = curried(operator.ne)
getitem = curried(operator.getitem)
itemgetter = operator.itemgetter
attrgetter = operator.attrgetter


def id_fn(x):
    return x


def noop(*args, **kwargs):  # pylint: disable=unused-argument
    pass


def fst(subscriptable):
    try:
        return subscriptable[0]
    except TypeError:
        return next(subscriptable)


def snd(subscriptable):
    try:
        return subscriptable[1]
    except TypeError:
        next(subscriptable)
        return next(subscriptable)


def singleton(cls_or_func):
    """Make a class or a function a Singleton class"""

    @wraps(cls_or_func)
    def wrapper_singleton(*args, **kwargs):
        if not wrapper_singleton.instance:
            wrapper_singleton.instance = cls_or_func(*args, **kwargs)
        return wrapper_singleton.instance

    wrapper_singleton.instance = None
    return wrapper_singleton


@curried
def str_startswith(
    prefix, string, start=None, end=None, case=True, regex=False
) -> bool:
    """
    If `case` is False then a case-insensitive comparison is performed.
    """
    if not case:
        prefix = prefix.lower()
        string = string.lower()
    if regex:
        if end:
            string = string[:end]
        if start:
            string = string[:start]
        return re.match(prefix, string) is not None
    return string.startswith(prefix, start, end)


@curried
def str_endswith(suffix, string, start=None, end=None, case=True, regex=False) -> bool:
    """
    If `case` is False then a case-insensitive comparison is performed.
    """
    if not case:
        suffix = suffix.lower()
        string = string.lower()
    if regex:
        if end:
            string = string[:end]
        if start:
            string = string[:start]
        return re.search(suffix + "$", string) is not None
    return string.endswith(suffix, start, end)


@curried
def str_contains(substring, string, case=True, regex=False) -> bool:
    """
    If `case` is False then a case-insensitive comparison is performed.
    """
    if not case:
        substring = substring.lower()
        string = string.lower()
    if regex:
        return re.search(substring, string) is not None
    return substring in string


@curried
def str_replace(old: str, new: str, string: str) -> str:
    return string.replace(old, new)


@curried
def not_(pred, *args, **kwargs):
    return not pred(*args, **kwargs)


def itemgetters(*items):
    """
    Return a callable that fetches nested items. For example:
    f = itemgetters('a', 'b')
    f({'a': {'b': 2}} == 2
    """

    def g(obj):
        ret = obj
        for item in items:
            ret = ret[item]
        return ret

    return g


@curried
def in_(collection: Iterable[V], item: V) -> bool:
    try:
        return item in collection
    except TypeError:
        return False


@curried
def has(item, collection) -> bool:
    try:
        return item in collection
    except TypeError:
        return False


@curried
def not_in(collection, item) -> bool:
    try:
        return item not in collection
    except TypeError:
        return False


@curried
def is_(b, a) -> bool:
    return a is b


@curried
def is_not(b, a) -> bool:
    return a is not b


@curried
def attr_eq(attr: str, val, obj) -> bool:
    return getattr(obj, attr) == val


@curried
def attr_neq(attr: str, val, obj) -> bool:
    return getattr(obj, attr) != val


@curried
def attr_is(attr: str, val, obj) -> bool:
    return getattr(obj, attr) is val


@curried
def attr_is_not(attr: str, val, obj) -> bool:
    return getattr(obj, attr) is not val


@curried
def itemget_eq(key, val, obj) -> bool:
    return obj[key] == val


@curried
def itemget_neq(key, val, obj) -> bool:
    return obj[key] != val


@curried
def itemget_is(key, val, obj) -> bool:
    return obj[key] is val


@curried
def itemget_is_not(key, val, obj) -> bool:
    return obj[key] is not val


def avg(iterable):
    total = 0
    n = 0
    for i in iterable:
        n += 1
        total += i
    return total / n


@curried
def list_append(item, li) -> None:
    if isinstance(li, list):
        li.append(item)


@curried
def list_append_unique(item, li) -> None:
    if isinstance(li, list):
        if item not in li:
            li.append(item)


@curried
def list_extend(items, li) -> None:
    if isinstance(li, list):
        li.extend(items)


@curried
def list_remove(value, li: List) -> None:
    if isinstance(li, list):
        try:
            li.remove(value)
        except ValueError:
            pass


@curried
def set_union(set1, set2) -> Set:
    return set1.union(set2)


@curried
def set_isdisjoint(set1, set2) -> Set:
    return set1.isdisjoint(set2)


@curried
def set_isunion(set1, set2) -> Set:
    return not set1.isdisjoint(set2)


@curried
def set_issubset(set1, set2) -> bool:
    return set1.issubset(set2)


@curried
def dict_get(*keys, default=None, di={}):
    for key in keys:
        if key in di:
            return di[key]
    return default


def chainf(*funcs):
    """
    Chain functions
    """

    @wraps(funcs[0])
    def inner(*args, **kwargs):
        res = funcs[0](*args, **kwargs)
        for func in funcs:
            res = func(res)
        return res

    return inner


def rsetattr(obj, attr, val):
    """
    Allows chained setattr, e.g. 'x.y'
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
    Allows chained setattr, e.g. 'x.y'
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))
