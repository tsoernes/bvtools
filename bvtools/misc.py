import inspect
import re
from typing import Callable

from betterpathlib import Path


def load_root_dotenv(
    name: Path | str | None = None,
    ignore_env_re: str | None = "example",
    override=True,
    strict=True,
    verbose=False,
) -> bool:
    """
    Locate the directory of the caller function. Look for files with '.env' suffix.
    If there are any, that `ignore_env_re` do not match, then load it. If not,
    iterate upwards the directory tree until one is found.

    """
    from dotenv import load_dotenv as _load_dotenv

    if name:
        env_path = name
    else:
        stack = inspect.stack()
        if len(stack) >= 3:
            # Go back two frames to get the caller of this function
            frame = inspect.stack()[2]
            caller_file = Path(frame.filename).resolve()
            current = caller_file.parent
            # print(f"Got {caller_file=} and {current=} from stack[2]")
            if caller_file.name == "interactiveshell.py":
                # print(f"Running in REPL. Attempting to find root from cwd")
                current = Path.cwd().resolve()
        elif len(stack) == 2:
            # Go back a frame to get the caller of this function
            frame = inspect.stack()[1]
            caller_file = frame.filename
            current = Path(caller_file).resolve().parent
            # print(f"Got {caller_file=} and {current=} from stack[1]")
        else:
            current = Path.cwd().resolve()
            # print(f"Got directory={current} from cwd")

        env_path = None
        while True:
            env_paths = current.glob("*.env")
            if ignore_env_re:
                env_paths = filter(
                    lambda p: not re.search(ignore_env_re, p.name), env_paths
                )
            env_paths = list(env_paths)
            if env_paths:
                if env_path := next(
                    filter(lambda p: p.name == ".env", iter(env_paths)), None
                ):
                    # env_path = env_path
                    break
                env_path = env_paths[0]
                break
            parent = current.parent
            if parent == current:
                raise ValueError("Reached root, could not find any '*.env' files")
            current = parent

    loaded = _load_dotenv(dotenv_path=env_path, override=override, verbose=verbose)
    if strict and not loaded:
        raise FileNotFoundError(".env")
    return loaded


def confirm_action(
    desc="Really execute?",
    yes_func: Callable | None = None,
    no_func: Callable | None = None,
    enter_is_yes=False,
) -> bool:
    """
    Return True if user confirms with 'Y' input
    """
    if desc == "Really execute?" and yes_func:
        desc = f"Really execute {yes_func.__name__}?"
    inp = None
    yes_inputs = ["y", "yes"]
    allowed_inputs = ["n", "no"] + yes_inputs
    if enter_is_yes:
        allowed_inputs.append("")
        yes_inputs.append("")
    while inp not in allowed_inputs:
        inp = input(desc + " Y/N: ").lower()
    yes = inp in yes_inputs
    if yes and yes_func:
        return yes_func()
    if not yes and no_func:
        return no_func()
    return yes
