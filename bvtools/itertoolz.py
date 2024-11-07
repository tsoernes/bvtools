import collections
import random
from concurrent.futures import ThreadPoolExecutor
from itertools import chain, islice
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from typing import (Any, Callable, Collection, Generator, Iterable, Iterator,
                    Sized, TypeVar, overload)

from bvtools.functoolz import curried

K = TypeVar("K")
L = TypeVar("L")
V = TypeVar("V")
W = TypeVar("W")
X = TypeVar("X")


@curried
def map_mp(
    fn: Callable[[V], W], seq: Iterable[V], n_concurrent: int = cpu_count()
) -> list[W]:
    """
    Apply a function to each item in a sequence using multiple processes.

    Args:
        fn: A function to apply to each item.
        seq: An iterable sequence of items.
        n_concurrent: The number of concurrent processes to use.

    Returns:
        A list of results from applying the function to each item.
    """
    with Pool(processes=n_concurrent) as pool:
        return pool.map(fn, seq)


@curried
def map_mt(
    fn: Callable[[V], W], seq: Iterable[V], n_concurrent: int = cpu_count()
) -> Iterator[W]:
    """
    Apply a function to each item in a sequence using multiple threads.

    Args:
        fn: A function to apply to each item.
        seq: An iterable sequence of items.
        n_concurrent: The number of concurrent threads to use.

    Returns:
        An iterator of results from applying the function to each item.
    """
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        return pool.map(fn, seq)


@curried
def starmap_mp(
    fn: Callable[..., W], seq: Iterable[Iterable[Any]], n_concurrent: int = cpu_count()
) -> list[W]:
    """
    Apply a function to each item in a sequence of tuples using multiple processes.

    Args:
        fn: A function to apply to each tuple.
        seq: An iterable sequence of tuples.
        n_concurrent: The number of concurrent processes to use.

    Returns:
        A list of results from applying the function to each tuple.
    """
    with Pool(processes=n_concurrent) as pool:
        return pool.starmap(fn, seq)


@curried
def _unstar(fn, seq):
    """
    Unpack a tuple and apply a function to its elements.

    Args:
        fn: A function to apply.
        seq: A sequence of arguments to unpack.

    Returns:
        The result of applying the function to the unpacked arguments.
    """
    return fn(*seq)


@curried
def starmap_mt(
    fn: Callable[..., W], seq: Iterable[Iterable[Any]], n_concurrent: int = cpu_count()
) -> Iterator[W]:
    """
    Apply a function to each item in a sequence of tuples using multiple threads.

    Args:
        fn: A function to apply to each tuple.
        seq: An iterable sequence of tuples.
        n_concurrent: The number of concurrent threads to use.

    Returns:
        An iterator of results from applying the function to each tuple.
    """
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        return pool.map(_unstar(fn), seq)


@curried
def itemmap(fn: Callable[[K, V], W], di: dict[K, V]) -> dict[K, W]:
    """
    Map a function over each (key, value) pair in a dictionary.

    Args:
        fn: A function to apply to each key-value pair.
        di: A dictionary to map over.

    Returns:
        A new dictionary with original keys pointing to function results.
    """
    return {key: fn(key, val) for key, val in di.items()}


@curried
def itemmap_mp(
    fn: Callable[[K, V], W], di: dict[K, V], n_concurrent: int = cpu_count()
) -> dict[K, W]:
    """
    Map a function over each (key, value) pair in a dictionary using multiple processes.

    Args:
        fn: A function to apply to each key-value pair.
        di: A dictionary to map over.
        n_concurrent: The number of concurrent processes to use.

    Returns:
        A new dictionary with original keys pointing to function results.
    """
    results_flat = starmap_mp(fn, di.items(), n_concurrent)
    return dict(zip(di.keys(), results_flat))


@curried
def itemmap_mt(
    fn: Callable[[K, V], W], di: dict[K, V], n_concurrent: int = cpu_count()
) -> dict[K, W]:
    """
    Map a function over each (key, value) pair in a dictionary using multiple threads.

    Args:
        fn: A function to apply to each key-value pair.
        di: A dictionary to map over.
        n_concurrent: The number of concurrent threads to use.

    Returns:
        A new dictionary with original keys pointing to function results.
    """
    results_flat = starmap_mt(fn, di.items(), n_concurrent)
    return dict(zip(di.keys(), results_flat))


@curried
def valuemap(fn: Callable[[V], W], di: dict[K, V]) -> dict[K, W]:
    """
    Map a function over each value in a dictionary.

    Args:
        fn: A function to apply to each value.
        di: A dictionary to map over.

    Returns:
        A new dictionary with original keys pointing to function results.
    """
    return {k: fn(v) for k, v in di.items()}


@curried
def valuemap_mp(
    fn: Callable[[V], W], di: dict[K, V], n_concurrent: int = cpu_count()
) -> dict[K, W]:
    """
    Map a function over each value in a dictionary using multiple processes.

    Args:
        fn: A function to apply to each value.
        di: A dictionary to map over.
        n_concurrent: The number of concurrent processes to use.

    Returns:
        A new dictionary with original keys pointing to function results.
    """
    results_flat = map_mp(fn, di.values(), n_concurrent)
    return dict(zip(di.keys(), results_flat))


@curried
def valuemap_mt(
    fn: Callable[[V], W], di: dict[K, V], n_concurrent: int = cpu_count()
) -> dict[K, W]:
    """
    Map a function over each value in a dictionary using multiple threads.

    Args:
        fn: A function to apply to each value.
        di: A dictionary to map over.
        n_concurrent: The number of concurrent threads to use.

    Returns:
        A new dictionary with original keys pointing to function results.
    """
    results_flat = map_mt(fn, di.values(), n_concurrent)
    return dict(zip(di.keys(), results_flat))


@curried
def keymap(fn: Callable[[K], W], di: dict[K, V]) -> dict[W, V]:
    """
    Map a function over each key in a dictionary.

    Args:
        fn: A function to apply to each key.
        di: A dictionary to map over.

    Returns:
        A new dictionary with transformed keys and original values.
    """
    return {fn(k): v for k, v in di.items()}


@curried
def keymap_mp(
    fn: Callable[[K], W], di: dict[K, V], n_concurrent: int = cpu_count()
) -> dict[W, V]:
    """
    Map a function over each key in a dictionary using multiple processes.

    Args:
        fn: A function to apply to each key.
        di: A dictionary to map over.
        n_concurrent: The number of concurrent processes to use.

    Returns:
        A new dictionary with transformed keys and original values.
    """
    results_flat = map_mp(fn, di.keys(), n_concurrent)
    return dict(zip(results_flat, di.values()))


@curried
def keymap_mt(
    fn: Callable[[K], W], di: dict[K, V], n_concurrent: int = cpu_count()
) -> dict[W, V]:
    """
    Map a function over each key in a dictionary using multiple threads.

    Args:
        fn: A function to apply to each key.
        di: A dictionary to map over.
        n_concurrent: The number of concurrent threads to use.

    Returns:
        A new dictionary with transformed keys and original values.
    """
    results_flat = map_mt(fn, di.keys(), n_concurrent)
    return dict(zip(results_flat, di.values()))


@curried
def listmap_dict(
    fn: Callable[[K, V], W], di: dict[K, Collection[V]], n_concurrent: int = cpu_count()
) -> dict[K, list[W]]:
    """
    Map a function over each (key, iterable element) pair in a dictionary.

    Args:
        fn: A function to apply to each key and element.
        di: A dictionary of iterables to map over.
        n_concurrent: The number of concurrent processes to use.

    Returns:
        A new dictionary with lists of function results for each key.
    """
    results_flat = starmap_mp(
        fn, ((k, v) for k, vs in di.items() for v in vs), n_concurrent
    )
    results_di = {}
    n_added = 0
    for key, inp in di.items():
        ninp = len(inp)
        results_di[key] = results_flat[n_added : n_added + ninp]
        n_added += ninp
    return results_di


@curried
def flatmap_dict(
    fn: Callable[[K, V], Iterable[W]],
    di: dict[K, Collection[V]],
    n_concurrent: int = cpu_count(),
) -> dict[K, list[W]]:
    """
    Apply a function that returns an iterable over each (key, list element) pair in a dictionary.

    Args:
        fn: A function to apply to each key and element.
        di: A dictionary of iterables to flatmap over.
        n_concurrent: The number of concurrent processes to use.

    Returns:
        A new dictionary with lists of flattened function results for each key.
    """
    results_flat = starmap_mp(
        fn, ((k, v) for k, vs in di.items() for v in vs), n_concurrent
    )

    results_di = {}
    n_added = 0
    for key, inp in di.items():
        ninp = len(inp)
        results_di[key] = list(chain(*results_flat[n_added : n_added + ninp]))
        n_added += ninp
    return results_di


@curried
def map_to_dict(fn: Callable[[K], V], seq: Iterable[K]) -> dict[K, V]:
    """
    Apply a function to each item in a sequence and return a dictionary.

    Args:
        fn: A function to apply to each item.
        seq: An iterable sequence of items.

    Returns:
        A dictionary with items as keys and function results as values.
    """
    return {k: fn(k) for k in seq}


@curried
def map_to_dict_mp(
    fn: Callable[[K], V], seq: Iterable[K], n_concurrent: int = cpu_count()
) -> dict[K, V]:
    """
    Apply a function to each item in a sequence using multiple processes and return a dictionary.

    Args:
        fn: A function to apply to each item.
        seq: An iterable sequence of items.
        n_concurrent: The number of concurrent processes to use.

    Returns:
        A dictionary with items as keys and function results as values.
    """
    seq = seq if isinstance(seq, list) else list(seq)
    results_flat = map_mp(fn, seq, n_concurrent)
    return dict(zip(seq, results_flat))


@curried
def map_to_dict_mt(
    fn: Callable[[K], V], seq: Iterable[K], n_concurrent: int = cpu_count()
) -> dict[K, V]:
    """
    Apply a function to each item in a sequence using multiple threads and return a dictionary.

    Args:
        fn: A function to apply to each item.
        seq: An iterable sequence of items.
        n_concurrent: The number of concurrent threads to use.

    Returns:
        A dictionary with items as keys and function results as values.
    """
    seq = seq if isinstance(seq, list) else list(seq)
    results_flat = map_mt(fn, seq, n_concurrent)
    return dict(zip(seq, results_flat))


@curried
def dict_sample(di: dict[K, V], n_samples: int = 1) -> dict[K, V]:
    """
    Return a randomly selected subset from the given dictionary.

    Args:
        di: A dictionary to sample from.
        n_samples: The number of samples to return.

    Returns:
        A dictionary containing the sampled key-value pairs.
    """
    return {k: di[k] for k in random.sample(di.keys(), n_samples)}


def dict_max(di: dict[K, V]) -> tuple[K, V]:
    """
    Return the key-item pair with the maximum item value from a dictionary.

    Args:
        di: A dictionary to find the maximum in.

    Returns:
        A tuple containing the key and the maximum value.
    """
    return max(di.items(), key=itemgetter(1))


def dict_pos(di: dict[K, float]) -> dict[K, float]:
    """
    Return key-item pairs with positive item values from a dictionary.

    Args:
        di: A dictionary to filter.

    Returns:
        A new dictionary containing only key-value pairs with positive values.
    """
    return {k: v for k, v in di.items() if v > 0}


@curried
def map_dict_get(di: dict[K, V], iterable: Iterable[K]) -> list[V]:
    """
    Retrieve values from a dictionary for a given iterable of keys.

    Args:
        di: A dictionary to retrieve values from.
        iterable: An iterable of keys to look up.

    Returns:
        A list of values corresponding to the keys in the iterable.
    """
    return [di[x] for x in iterable if x in di]


class IxGetter:
    """
    Initialize with a dictionary; access dictionary items by numerical index (zero copy).
    """

    def __init__(self, di: dict[K, V]):
        self.dik = list(di.keys())
        self.div = list(di.values())

    def __getitem__(self, ix: int) -> V:
        """
        Retrieve the value at a specific index.

        Args:
            ix: The index of the value to retrieve.

        Returns:
            The value at the specified index.
        """
        return self.div[ix]

    def __len__(self) -> int:
        """
        Return the number of items in the dictionary.

        Returns:
            The number of items.
        """
        return len(self.div)

    def key(self, ix) -> K:
        """
        Retrieve the key at a specific index.

        Args:
            ix: The index of the key to retrieve.

        Returns:
            The key at the specified index.
        """
        return self.dik[ix]

    def keys(self) -> list[K]:
        """
        Retrieve all keys in the dictionary.

        Returns:
            A list of keys.
        """
        return self.dik

    def items(self):
        """
        Retrieve all key-value pairs in the dictionary.

        Returns:
            An iterator of key-value pairs.
        """
        return zip(self.dik, self.div)


def rename_dict_keys(di: dict, mapping: dict, allow_missing: bool = True) -> None:
    """
    Rename keys in a dictionary based on a mapping.

    Args:
        di: A dictionary to modify.
        mapping: A dictionary mapping old keys to new keys.
        allow_missing: Whether to allow missing keys in the mapping.

    Returns:
        None; modifies the dictionary in place.
    """
    for old_key, new_key in mapping.items():
        if allow_missing and old_key in di:
            di[new_key] = di.pop(old_key)
        else:
            di[new_key] = di.pop(old_key)


def sort_di_values_by_key(di: dict[K, V]) -> list[V]:
    """
    Sort a dictionary by its keys and return its values.

    Args:
        di: A dictionary to sort.

    Returns:
        A list of values sorted by their corresponding keys.
    """
    return [x[1] for x in sorted(di.items(), key=lambda tup: tup[0])]


@curried
def filter_di(pred: Callable[[K], bool], di: dict[K, V]) -> dict[K, V]:
    """
    Filter a dictionary by a predicate on its keys.

    Args:
        pred: A predicate function to apply to the keys.
        di: A dictionary to filter.

    Returns:
        A new dictionary containing only key-value pairs that satisfy the predicate.
    """
    return {k: v for k, v in di.items() if pred(k)}


@curried
def filter_di_vals(pred: Callable[[V], bool], di: dict[K, V]) -> dict[K, V]:
    """
    Filter a dictionary by a predicate on its values.

    Args:
        pred: A predicate function to apply to the values.
        di: A dictionary to filter.

    Returns:
        A new dictionary containing only key-value pairs that satisfy the predicate.
    """
    return {k: v for k, v in di.items() if pred(v)}


@curried
def filter_di_items(pred: Callable[[K, V], bool], di: dict[K, V]) -> dict[K, V]:
    """
    Filter a dictionary by a predicate on its key-value pairs.

    Args:
        pred: A predicate function to apply to the key-value pairs.
        di: A dictionary to filter.

    Returns:
        A new dictionary containing only key-value pairs that satisfy the predicate.
    """
    return {k: v for k, v in di.items() if pred(k, v)}


@curried
def lfilter(pred: Callable[[V], bool], iterable: Iterable[V]) -> list[V]:
    """
    Filter an iterable based on a predicate and return a list.

    Args:
        pred: A predicate function to apply to the items.
        iterable: An iterable to filter.

    Returns:
        A list of items that satisfy the predicate.
    """
    return [x for x in iterable if pred(x)]


@curried
def lfilterfalse(pred: Callable[[V], bool], iterable: Iterable[V]) -> list[V]:
    """
    Filter an iterable based on the negation of a predicate and return a list.

    Args:
        pred: A predicate function to apply to the items.
        iterable: An iterable to filter.

    Returns:
        A list of items that do not satisfy the predicate.
    """
    return [x for x in iterable if not pred(x)]


@curried
def lfiltermap(
    pred: Callable[[V], bool], fn: Callable[[V], W], iterable: Iterable[V]
) -> list[W]:
    """
    Filter an iterable based on a predicate, then map a function over the filtered items.

    Args:
        pred: A predicate function to apply to the items.
        fn: A function to apply to the filtered items.
        iterable: An iterable to filter and map.

    Returns:
        A list of results from applying the function to the filtered items.
    """
    return [fn(x) for x in iterable if pred(x)]


@curried
def filtermap(
    pred: Callable[[V], bool], fn: Callable[[V], W], iterable: Iterable[V]
) -> Iterator[W]:
    """
    Filter an iterable based on a predicate, then map a function over the filtered items.

    Args:
        pred: A predicate function to apply to the items.
        fn: A function to apply to the filtered items.
        iterable: An iterable to filter and map.

    Returns:
        An iterator of results from applying the function to the filtered items.
    """
    return map(fn, filter(pred, iterable))


@curried
def lmapfilter(
    fn: Callable[[V], W], pred: Callable[[W], bool], iterable: Iterable[V]
) -> list[W]:
    """
    Map a function over an iterable, then filter the results based on a predicate.

    Args:
        fn: A function to apply to the items.
        pred: A predicate function to apply to the mapped results.
        iterable: An iterable to map and filter.

    Returns:
        A list of results that satisfy the predicate.
    """
    return [x for x in map(fn, iterable) if pred(x)]


@curried
def mapfilter(
    fn: Callable[[V], W], pred: Callable[[W], bool], iterable: Iterable[V]
) -> Iterator[W]:
    """
    Map a function over an iterable, then filter the results based on a predicate.

    Args:
        fn: A function to apply to the items.
        pred: A predicate function to apply to the mapped results.
        iterable: An iterable to map and filter.

    Returns:
        An iterator of results that satisfy the predicate.
    """
    return filter(pred, map(fn, iterable))


@curried
def lmapfilterfalse(
    fn: Callable[[V], W], pred: Callable[[W], bool], iterable: Iterable[V]
) -> list[W]:
    """
    Map a function over an iterable, then filter the results based on the negation of a predicate.

    Args:
        fn: A function to apply to the items.
        pred: A predicate function to apply to the mapped results.
        iterable: An iterable to map and filter.

    Returns:
        A list of results that do not satisfy the predicate.
    """
    return [x for x in map(fn, iterable) if not pred(x)]


@curried
def mapfilterfalse(
    fn: Callable[[V], W], pred: Callable[[W], bool], iterable: Iterable[V]
) -> Iterator[W]:
    """
    Map a function over an iterable, then retain results that do not satisfy a predicate.

    Args:
        fn: A function to apply to the items.
        pred: A predicate function to apply to the mapped results.
        iterable: An iterable to map and filter.

    Returns:
        An iterator of results that do not satisfy the predicate.
    """
    return filter(lambda x: not pred(x), map(fn, iterable))


@curried
def anymap(fn: Callable[[V], W], iterable: Iterable[V]) -> bool:
    """
    Apply a function to each item in an iterable and check if any result is truthy.

    Args:
        fn: A function to apply to the items.
        iterable: An iterable to check.

    Returns:
        True if any result is truthy, otherwise False.
    """
    return any(map(fn, iterable))


@curried
def allmap(fn: Callable[[V], W], iterable: Iterable[V]) -> bool:
    """
    Apply a function to each item in an iterable and check if all results are truthy.

    Args:
        fn: A function to apply to the items.
        iterable: An iterable to check.

    Returns:
        True if all results are truthy, otherwise False.
    """
    return all(map(fn, iterable))


@curried
def smap(fn: Callable[[V], W], iterable: Iterable[V]) -> set[W]:
    """
    Apply a function to each item in an iterable and return a set of results.

    Args:
        fn: A function to apply to the items.
        iterable: An iterable to map.

    Returns:
        A set of results from applying the function to each item.
    """
    return set(map(fn, iterable))


@curried
def lmap(fn: Callable[[V], W], iterable: Iterable[V]) -> list[W]:
    """
    Apply a function to each item in an iterable and return a list of results.

    Args:
        fn: A function to apply to the items.
        iterable: An iterable to map.

    Returns:
        A list of results from applying the function to each item.
    """
    return [fn(x) for x in iterable]


@curried
def lmap2(
    fn1: Callable[[W], X], fn2: Callable[[V], W], iterable: Iterable[V]
) -> list[X]:
    """
    Apply one function after another over each item in an iterable and return a list of results.

    Args:
        fn1: A function to apply to the results of fn2.
        fn2: A function to apply to the items.
        iterable: An iterable to map.

    Returns:
        A list of results from applying fn1 to the results of fn2.
    """
    return [fn1(fn2(x)) for x in iterable]


@curried
def map2(
    fn1: Callable[[W], X], fn2: Callable[[V], W], iterable: Iterable[V]
) -> Generator[X, None, None]:
    """
    Apply one function after another over each item in an iterable and return a generator of results.

    Args:
        fn1: A function to apply to the results of fn2.
        fn2: A function to apply to the items.
        iterable: An iterable to map.

    Returns:
        A generator of results from applying fn1 to the results of fn2.
    """
    return (fn1(fn2(x)) for x in iterable)


@curried
def sub_dict(di, *keys):
    """
    Return a subset of the dictionary containing only the specified keys.

    Args:
        di: A dictionary to subset.
        keys: Keys to include in the subset.

    Returns:
        A new dictionary containing only the specified keys.
    """
    return {k: di[k] for k in keys if k in di}


@curried
def sub_dict_inv(di, *keys):
    """
    Return a subset of the dictionary excluding the specified keys.

    Args:
        di: A dictionary to subset.
        keys: Keys to exclude from the subset.

    Returns:
        A new dictionary containing all keys except the specified ones.
    """
    return {k: di[k] for k in di.keys() if k not in keys}


@curried
def remove_dict_keys(di, *keys, allow_missing=False) -> None:
    """
    Remove specified keys from a dictionary.

    Args:
        di: A dictionary to modify.
        keys: Keys to remove from the dictionary.
        allow_missing: Whether to allow missing keys without raising an error.

    Returns:
        None; modifies the dictionary in place.
    """
    if allow_missing:
        keys = set(keys) & di.keys()
    for k in keys:
        del di[k]


@curried
def sub_list(li, *keys):
    """
    Return a list containing only the specified items.

    Args:
        li: A list to subset.
        keys: Items to include in the subset.

    Returns:
        A new list containing only the specified items.
    """
    return [c for c in li if c in keys]


@curried
def sub_list_inv(li, *keys):
    """
    Return a list excluding the specified items.

    Args:
        li: A list to subset.
        keys: Items to exclude from the subset.

    Returns:
        A new list containing all items except the specified ones.
    """
    return [c for c in li if c not in keys]


def tail(iterable: Iterable[V]) -> Iterator[V]:
    """
    Skip the first item in an iterable.

    Args:
        iterable: An iterable to process.

    Returns:
        An iterator starting from the second item.
    """
    iter_ = iter(iterable)
    next(iter_, False)
    return iter_


def peek(iterable: Iterable[V]) -> tuple[V, Iterable[V]]:
    """
    Return the first item of the iterable, and the iterable itself, unchanged.

    Args:
        iterable: An iterable to peek into.

    Returns:
        A tuple containing the first item and the original iterable.
    """
    if isinstance(iterable, list):
        return iterable[0], iterable
    it = iter(iterable)
    el = next(it)
    return el, chain([el], it)


@curried
def chunks(iterable, size):
    """
    Split an iterable into chunks of a specified size.

    Args:
        iterable: An iterable to split.
        size: The size of each chunk.

    Yields:
        Chunks of the specified size from the iterable.
    """
    it = iter(iterable)
    item = list(islice(it, size))
    while item:
        yield item
        item = list(islice(it, size))


@curried
def map_partitions(
    fn: Callable[[Iterable[V]], Iterable[W]],
    iterable: Iterable[V],
    iter_len_hint: int | None = None,
    partition_size: int | None = None,
    n_concurrent: int = cpu_count(),
) -> list[W]:
    """
    Split the iterable into chunks and apply a function to each chunk in parallel.

    Args:
        fn: A function to apply to each chunk.
        iterable: An iterable to process.
        iter_len_hint: An optional hint for the length of the iterable.
        partition_size: An optional size for each partition.
        n_concurrent: The number of concurrent processes to use.

    Returns:
        A list of results from applying the function to each chunk.
    """
    if not partition_size:
        if not iter_len_hint:
            if isinstance(iterable, Sized):
                iter_len_hint = len(iterable)
            else:
                iterable = list(iterable)
                iter_len_hint = len(iterable)
        partition_size = iter_len_hint // n_concurrent

    with Pool(processes=n_concurrent) as pool:
        result_chunks = pool.map(fn, chunks(iterable, partition_size))
    return [result for chunk in result_chunks for result in chunk]


@curried
def find(
    pred: Callable[[W], bool],
    iterable: Iterable[V],
    key: Callable[[V], W] | None = None,
    default: V | None = None,
) -> V:
    """
    Retrieve the first item from an iterable that matches a predicate.

    Args:
        pred: A predicate function to apply to the items.
        iterable: An iterable to search.
        key: An optional function to transform items before applying the predicate.
        default: A default value to return if no item matches.

    Returns:
        The first matching item, or the default value if none match.
    """
    if key is not None:
        return next(filter(pred, map(key, iterable)), default)
    return next(filter(pred, iterable), default)


@overload
def find_all(
    pred: Callable[[V], bool],
    iterable: Iterable[V],
) -> list[V]:
    """
    Retrieve all items from an iterable that match a predicate.

    Args:
        pred: A predicate function to apply to the items.
        iterable: An iterable to search.

    Returns:
        A list of all matching items.
    """
    ...


@overload
def find_all(
    pred: Callable[[W], bool] | Callable[[V], bool],
    iterable: Iterable[V],
    key: Callable[[V], W],
) -> list[V]:
    """
    Retrieve all items from an iterable that match a predicate, with an optional key function.

    Args:
        pred: A predicate function to apply to the items.
        iterable: An iterable to search.
        key: A function to transform items before applying the predicate.

    Returns:
        A list of all matching items.
    """
    ...


def find_all(
    pred: Callable,
    iterable: Iterable[V],
    key: Any | None = None,
) -> list[V]:
    """
    Retrieve all items from an iterable that match a predicate.

    Args:
        pred: A predicate function to apply to the items.
        iterable: An iterable to search.
        key: An optional function to transform items before applying the predicate.

    Returns:
        A list of all matching items.
    """
    if key is not None:
        return list(filter(pred, map(key, iterable)))
    return list(filter(pred, iterable))


@curried
def consume(iterator: Iterator, n: int | None = None):
    """
    Advance the iterator by a specified number of steps.

    Args:
        iterator: An iterator to consume.
        n: The number of steps to advance. If None, consume entirely.

    Returns:
        The iterator after advancing.
    """
    if n == 0:
        return iterator
    if n is None:
        collections.deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)


@curried
def find_ix(
    pred: Callable[[V], bool] | Callable[[W], bool],
    iterable: Iterable[V],
    key: Callable[[V], W] | None = None,
    default: X = None,
) -> int | X:
    """
    Retrieve the index of the first item in an iterable that matches a predicate.

    Args:
        pred: A predicate function to apply to the items.
        iterable: An iterable to search.
        key: An optional function to transform items before applying the predicate.
        default: A default value to return if no item matches.

    Returns:
        The index of the first matching item, or the default value if none match.
    """
    if key is not None:
        for ix, val in enumerate(iterable):
            if pred(key(val)):
                return ix
    else:
        for ix, val in enumerate(iterable):
            if pred(val):
                return ix
    return default


@curried
def find_ixs(
    pred: Callable[[W], bool],
    iterable: Iterable[V],
    key: Callable[[V], W] | None = None,
) -> list[int]:
    """
    Retrieve the indices of all items in an iterable that match a predicate.

    Args:
        pred: A predicate function to apply to the items.
        iterable: An iterable to search.
        key: An optional function to transform items before applying the predicate.

    Returns:
        A list of indices of all matching items.
    """
    hits = []
    if key is not None:
        for ix, val in enumerate(iterable):
            if pred(key(val)):
                hits.append(ix)
    else:
        for ix, val in enumerate(iterable):
            if pred(val):
                hits.append(ix)
    return hits


def nested_children_of(xx, include_keys=True) -> set:
    """
    Retrieve all nested children from a structure, optionally including keys.

    Args:
        xx: The structure to analyze (can be a list, tuple, or dict).
        include_keys: Whether to include keys in the result if the structure is a dictionary.

    Returns:
        A set of all nested children.
    """
    res = set()

    def _add(yy):
        if isinstance(yy, (list, tuple)):
            for x in yy:
                _add(x)
        elif isinstance(yy, dict):
            if include_keys:
                for k, v in yy.items():
                    res.add(k)
                    _add(v)
            else:
                for v in yy.values():
                    _add(v)
        else:
            res.add(yy)

    _add(xx)
    return res


@curried
def split_by_pred(
    data: Iterable[V], pred: Callable[[V], bool]
) -> tuple[list[V], list[V]]:
    """
    Split an iterable into two lists based on a predicate function.

    Args:
        data: An iterable to split.
        pred: A predicate function to apply to the items.

    Returns:
        A tuple of two lists: one with items that satisfy the predicate and one with items that do not.
    """
    yes, no = [], []
    for d in data:
        if pred(d):
            yes.append(d)
        else:
            no.append(d)
    return (yes, no)


class FilterCount:
    """
    An iterator that counts how many items satisfy a predicate while iterating.
    """

    def __init__(self, function, iterable):
        self.function = function
        self.iterable = iter(iterable)
        self.count_true, self.count_false = 0, 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Retrieve the next item from the iterable that satisfies the predicate.

        Returns:
            The next item that satisfies the predicate.

        Raises:
            StopIteration: If no more items satisfy the predicate.
        """
        nxt = next(self.iterable)
        while not self.function(nxt):
            self.count_false += 1
            nxt = next(self.iterable)

        self.count_true += 1
        return nxt


@curried
def map_nth(func, n: int, iterable):
    return map(lambda seq: seq[:n] + type(seq)([func(seq[n])]) + seq[n + 1 :], iterable)


@curried
def lmap_nth(func, n: int, iterable):
    return list(map_nth(func, n, iterable))


@curried
def map_fst(func, iterable):
    return map_nth(func, 0, iterable)


@curried
def map_snd(func, iterable):
    return map_nth(func, 1, iterable)


@curried
def map_third(func, iterable):
    return map_nth(func, 2, iterable)


@curried
def lmap_fst(func, iterable):
    return list(map_nth(func, 0, iterable))


@curried
def lmap_snd(func, iterable):
    return list(map_nth(func, 1, iterable))


@curried
def lmap_third(func, iterable):
    return list(map_nth(func, 2, iterable))
