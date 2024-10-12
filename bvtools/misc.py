import inspect
from typing import Callable

from betterpathlib import Path


def git_root() -> Path:
    """Find the Path of the file of the calling function, and traverse upwards the directory tree
    until a '.git' directory is found. This is usually the project root of the calling function.
    """
    stack = inspect.stack()
    if len(stack) >= 3:
        # Go back two frames to get the caller of this function
        frame = inspect.stack()[2]
        caller_file = frame.filename
        current = Path(caller_file).resolve().parent
        # print(f"Got {caller_file=} and {current=} from stack[2]")
    elif len(stack) == 2:
        # Go back a frame to get the caller of this function
        frame = inspect.stack()[1]
        caller_file = frame.filename
        current = Path(caller_file).resolve().parent
        # print(f"Got {caller_file=} and {current=} from stack[1]")
    else:
        current = Path.cwd().resolve()
        # print(f"Got directory={current} from cwd")

    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            raise ValueError("Reached root, could not find '.git' folder")
        current = parent


def load_root_dotenv(name=".env", override=True, strict=True, verbose=False) -> bool:
    """Loads the '.env' file at the callers project root."""
    from dotenv import load_dotenv as _load_dotenv

    root = git_root()
    dotenv_path = root / name
    loaded = _load_dotenv(dotenv_path=dotenv_path, override=override, verbose=verbose)
    if strict and not loaded:
        raise FileNotFoundError(dotenv_path)
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
