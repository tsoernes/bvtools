import atexit
import inspect
import itertools
import logging
import operator
import re
from argparse import ArgumentError
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce, wraps
from multiprocessing.pool import AsyncResult, ThreadPool
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
)

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
P = ParamSpec("P")
R = TypeVar("R")
V = TypeVar("V")


class Subscriptable(Protocol[T]):
    def __getitem__(self, index: int) -> T: ...


def curried(func: Callable) -> Callable:
    # def curried(func: Callable[P, T]) -> Callable[U, T]:
    """
    Decorator that enables currying of the decorated function.

    Currying transforms a function that takes multiple arguments into a series of functions
    that each take a single argument. This allows partial application of the function's arguments.

    Args:
        func: The function to be curried.

    Returns:
        A curried version of the input function.

    Example:
        >>> @curried
        ... def add(x: int, y: int) -> int:
        ...     return x + y
        >>> add(1)(2)
        3
        >>> add(1, 2)
        3
    """
    sig = inspect.signature(func)

    @wraps(func)
    def inner(*args: P.args, **kwargs: P.kwargs) -> T | Callable:
        bind = sig.bind_partial(*args, **kwargs)
        for param in sig.parameters.values():
            if param.name not in bind.arguments and param.default is param.empty:
                return partial(func, *args, **kwargs)
        return func(*args, **kwargs)

    return inner


# Global thread pool executor for async operations
_thread_executor = ThreadPoolExecutor()
atexit.register(lambda: _thread_executor.shutdown(wait=True))

# Global thread pools for pooled async operations
_thread_pools: dict[str, ThreadPool] = {}


@atexit.register
def cleanup_thread_pools() -> None:
    """Cleanup function to properly shut down all thread pools."""
    logger.info("Cleaning up thread pools...")
    for name, pool in _thread_pools.items():
        logger.debug(f"Shutting down thread pool for {name}")
        pool.close()
        pool.join()
    logger.info("Thread pools cleanup completed")


def allow_async(
    threads: int = 1,
) -> Callable[[Callable[P, R]], Callable[P, R | AsyncResult[R]]]:
    """
    Decorator that enables threaded execution using a thread pool.

    Args:
        threads: Maximum number of threads to use in the pool.

    Returns:
        A decorated function that accepts an additional `async_` parameter.
        When `async_=True`, returns an AsyncResult.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R | AsyncResult[R]]:
        old_sig = inspect.signature(func)
        params = list(old_sig.parameters.values())
        async_param = inspect.Parameter(
            "async_", inspect.Parameter.KEYWORD_ONLY, default=False, annotation=bool
        )
        params.append(async_param)
        sig = old_sig.replace(parameters=params)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | AsyncResult[R]:
            bind = sig.bind_partial(*args, **kwargs)
            bind.apply_defaults()

            is_async = bind.arguments.pop("async_")

            if is_async:
                if func.__name__ not in _thread_pools:
                    _thread_pools[func.__name__] = ThreadPool(threads)
                return _thread_pools[func.__name__].apply_async(
                    func, args=args, kwds=kwargs
                )

            return func(*args, **kwargs)

        wrapper.__signature__ = sig
        return wrapper

    return decorator


def allow_await(func: Callable[P, R]) -> Callable[P, R | Awaitable[R]]:
    """
    Decorator that enables async/await execution.

    Returns:
        A decorated function that accepts an additional `awaitable` parameter.
        When `awaitable=True`, returns an Awaitable.
    """
    old_sig = inspect.signature(func)
    params = list(old_sig.parameters.values())
    awaitable_param = inspect.Parameter(
        "awaitable", inspect.Parameter.KEYWORD_ONLY, default=False, annotation=bool
    )
    params.append(awaitable_param)
    sig = old_sig.replace(parameters=params)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | Awaitable[R]:
        bind = sig.bind_partial(*args, **kwargs)
        bind.apply_defaults()

        awaitable = bind.arguments.pop("awaitable")

        if awaitable:
            return cast(Awaitable[R], _thread_executor.submit(func, *args, **kwargs))

        return func(*args, **kwargs)

    wrapper.__signature__ = sig
    return wrapper


eq = curried(operator.eq)
lt = curried(operator.lt)
le = curried(operator.le)
gt = curried(operator.gt)
ge = curried(operator.ge)
neq = curried(operator.ne)
getitem = curried(operator.getitem)
itemgetter = operator.itemgetter
attrgetter = operator.attrgetter


def id_fn[T](x: T) -> T:
    """Identity function that returns its argument unchanged."""
    return x


def noop(*_args: Any, **_kwargs: Any) -> None:
    """No operation. Function that does nothing and returns None."""
    pass


def fst[T](subscriptable: Sequence[T] | Iterable[T]) -> T:
    """
    Returns the first element of a sequence or iterable.

    Args:
        subscriptable: A sequence or iterable containing elements.

    Returns:
        The first element.

    Raises:
        StopIteration: If the iterable is empty.
        TypeError: If the input is neither a sequence nor an iterable.
    """
    try:
        return subscriptable[0]  # type: ignore
    except (TypeError, AttributeError):
        try:
            return next(iter(subscriptable))
        except (TypeError, StopIteration) as e:
            logger.error(f"Failed to get first element: {e}")
            raise


def snd[T](subscriptable: Sequence[T] | Iterable[T]) -> T:
    """
    Returns the second element of a sequence or iterable.

    Args:
        subscriptable: A sequence or iterable containing elements.

    Returns:
        The second element.

    Raises:
        StopIteration: If the iterable has fewer than 2 elements.
        TypeError: If the input is neither a sequence nor an iterable.
    """
    try:
        return subscriptable[1]  # type: ignore
    except (TypeError, AttributeError):
        try:
            iterator = iter(subscriptable)
            next(iterator)  # Skip first element
            return next(iterator)
        except (TypeError, StopIteration) as e:
            logger.error(f"Failed to get second element: {e}")
            raise


def singleton[T](cls_or_func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to implement the singleton pattern.
    """

    @wraps(cls_or_func)
    def wrapper_singleton(*args: Any, **kwargs: Any) -> T:
        if not hasattr(wrapper_singleton, "instance"):
            wrapper_singleton.instance = cls_or_func(*args, **kwargs)
        return wrapper_singleton.instance  # type: ignore

    return wrapper_singleton


@curried
def str_startswith(
    prefix: str,
    string: str,
    start: int | None = None,
    end: int | None = None,
    case: bool = True,
    regex: bool = False,
) -> bool:
    """Check if string starts with prefix."""
    if not case:
        prefix = prefix.lower()
        string = string.lower()
    if regex:
        if end:
            string = string[:end]
        if start:
            string = string[start:]
        return re.match(prefix, string) is not None
    return string.startswith(prefix, start, end)


@curried
def str_endswith(
    suffix: str,
    string: str,
    start: int | None = None,
    end: int | None = None,
    case: bool = True,
    regex: bool = False,
) -> bool:
    """Check if string ends with suffix."""
    if not case:
        suffix = suffix.lower()
        string = string.lower()
    if regex:
        if end:
            string = string[:end]
        if start:
            string = string[start:]
        return re.search(suffix + "$", string) is not None
    return string.endswith(suffix, start, end)


@curried
def str_contains(
    substring: str, string: str, case: bool = True, regex: bool = False
) -> bool:
    """Check if string contains substring."""
    if not case:
        substring = substring.lower()
        string = string.lower()
    if regex:
        return re.search(substring, string) is not None
    return substring in string


@curried
def str_replace(old: str, new: str, string: str) -> str:
    """Replace old with new in string."""
    return string.replace(old, new)


@curried
def not_(pred: Callable[..., bool], *args: Any, **kwargs: Any) -> bool:
    """Returns the negation of calling pred with the given arguments."""
    return not pred(*args, **kwargs)


def itemgetters(*items: str) -> Callable[[Mapping[str, T]], T]:
    """
    Return a callable that fetches nested items.

    Example:
        >>> f = itemgetters('a', 'b')
        >>> f({'a': {'b': 2}})
        2
    """

    def getter(obj: Mapping[str, Any]) -> Any:
        result = obj
        for item in items:
            result = result[item]
        return result

    return getter


@curried
def in_[T](collection: Iterable[T], item: T) -> bool:
    """Check if item is in collection."""
    try:
        return item in collection
    except TypeError:
        return False


@curried
def has[T](item: T, collection: Iterable[T]) -> bool:
    """Check if collection has item."""
    try:
        return item in collection
    except TypeError:
        return False


@curried
def not_in[T](collection: Iterable[T], item: T) -> bool:
    """Check if item is not in collection."""
    try:
        return item not in collection
    except TypeError:
        return False


@curried
def is_(b: Any, a: Any) -> bool:
    """Check if a is b."""
    return a is b


@curried
def is_not(b: Any, a: Any) -> bool:
    """Check if a is not b."""
    return a is not b


@curried
def attr_eq(attr: str, val: Any, obj: Any) -> bool:
    """Check if object's attribute equals value."""
    return getattr(obj, attr) == val


@curried
def attr_neq(attr: str, val: Any, obj: Any) -> bool:
    """Check if object's attribute does not equal value."""
    return getattr(obj, attr) != val


@curried
def attr_is(attr: str, val: Any, obj: Any) -> bool:
    """Check if object's attribute is value."""
    return getattr(obj, attr) is val


@curried
def attr_is_not(attr: str, val: Any, obj: Any) -> bool:
    """Check if object's attribute is not value."""
    return getattr(obj, attr) is not val


@curried
def itemget_eq(key: Any, val: Any, obj: Mapping[Any, Any]) -> bool:
    """Check if object[key] equals value."""
    return obj[key] == val


@curried
def itemget_neq(key: Any, val: Any, obj: Mapping[Any, Any]) -> bool:
    """Check if object[key] does not equal value."""
    return obj[key] != val


@curried
def itemget_is(key: Any, val: Any, obj: Mapping[Any, Any]) -> bool:
    """Check if object[key] is value."""
    return obj[key] is val


@curried
def itemget_is_not(key: Any, val: Any, obj: Mapping[Any, Any]) -> bool:
    """Check if object[key] is not value."""
    return obj[key] is not val


def avg(iterable: Iterable[float | int]) -> float:
    """Calculate the average of an iterable of numbers."""
    total = 0
    count = 0
    for num in iterable:
        count += 1
        total += num
    if count == 0:
        raise ValueError("Cannot calculate average of empty iterable")
    return total / count


@curried
def list_append[T](item: T, lst: list[T]) -> None:
    """Append item to list."""
    lst.append(item)


@curried
def list_append_unique[T](item: T, lst: list[T]) -> None:
    """Append item to list if it's not already present."""
    if item not in lst:
        lst.append(item)


@curried
def list_extend[T](items: Iterable[T], lst: list[T]) -> None:
    """Extend list with items if it is a list."""
    lst.extend(items)


@curried
def list_remove[T](value: T, lst: list[T]) -> None:
    """Remove value from list if present."""
    try:
        lst.remove(value)
    except ValueError:
        pass


@curried
def set_union[T](set1: set[T], set2: set[T]) -> set[T]:
    """Return the union of two sets."""
    return set1.union(set2)


@curried
def set_isdisjoint[T](set1: set[T], set2: set[T]) -> bool:
    """Check if two sets are disjoint."""
    return set1.isdisjoint(set2)


@curried
def set_isunion[T](set1: set[T], set2: set[T]) -> bool:
    """Check if two sets have a non-empty intersection."""
    return not set1.isdisjoint(set2)


@curried
def set_issubset[T](set1: set[T], set2: set[T]) -> bool:
    """Check if set1 is a subset of set2."""
    return set1.issubset(set2)


@curried
def dict_get(*keys: Any, default: Any = None, di: dict[Any, Any] = {}) -> Any:
    """
    Get value from dictionary using multiple possible keys.
    Returns the first matching value or default if none found.
    """
    for key in keys:
        if key in di:
            return di[key]
    return default


def dict_set(
    di: dict[Any, Any],
    *item_pairs,
    **items,
) -> Any:
    """
    Update / set the key-value pairs given a as iterator of pairs
    or as keyword-value arguments.


    >>> dict_set({"alpha": 1}, (("beta", 2), ("ceta", 3)), ceta=4, gamma=5)
    {'alpha': 1, 'beta': 2, 'ceta': 4, 'gamma': 5}

    """
    di.update(*item_pairs)
    di.update(items)
    return di


def chainf(*funcs: Callable) -> Callable:
    """
    Chain functions, passing the result of each to the next.
    """

    @wraps(funcs[0])
    def inner(*args: Any, **kwargs: Any) -> Any:
        res = funcs[0](*args, **kwargs)
        for func in funcs[1:]:
            res = func(res)
        return res

    return inner


@curried
def rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    """
    Recursively get nested attributes.

    Example:
        >>> rgetattr(obj, 'attr.subattr')
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))


@curried
def rsetattr(obj: Any, attr: str, val: Any) -> None:
    """
    Recursively set nested attributes.

    Allows chained setattr, e.g. 'x.y'
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


@curried
def rget_attr_or_key(object, *attr_or_key: str) -> Any:
    """
    Recursively get attrs or keys

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        >>> class Obj:
        >>>     di: dict
        >>> obj = Obj({"level1": {"level2": "value"}})
        >>> value = get_attr_or_key(obj, "di", "level1", "level2")

    """
    if len(attr_or_key) == 1:
        attr_or_key = tuple(attr_or_key[0].split("."))
    for attr_key in attr_or_key:
        if isinstance(object, Mapping):
            object = object[attr_key]
        else:
            object = getattr(object, attr_key)
    return object


def rset_attr_or_key(object, value, attr_or_key: str, *attr_or_keys: str) -> Any:
    """
    Recursively set attrs or keys

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        >>> class Obj:
        >>>     di: dict
        >>> obj = Obj({"level1": {"level2": "value"}})
        >>> obj = rset_attr_or_key(obj, "newvalue", "di", "level1", "level2")i
        Obj(di={'level1': {'level2': 'newvalue'}})
    """
    # If there are more keys, recurse
    if attr_or_keys:
        next_attr_or_key = attr_or_keys[0]
        next_object = (
            getattr(object, attr_or_key)
            if hasattr(object, attr_or_key)
            else object[attr_or_key]
        )
        # Recursively call with the next attribute/key
        new_next_object = rset_attr_or_key(
            next_object, value, next_attr_or_key, *attr_or_keys[1:]
        )
        # Set the modified object back
        if hasattr(object, attr_or_key):
            setattr(object, attr_or_key, new_next_object)
        else:
            object[attr_or_key] = new_next_object
    else:
        # Set the final value
        if hasattr(object, attr_or_key):
            setattr(object, attr_or_key, value)
        else:
            object[attr_or_key] = value
    return object
