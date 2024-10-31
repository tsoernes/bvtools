import inspect
import subprocess
import shlex
import platform
import os
import re
from typing import Callable

from betterpathlib import Path


def load_root_dotenv(
    name: Path | str | None = None,
    ignore_env_re: str | None = "example",
    override: bool = True,
    strict: bool = True,
    verbose: bool = False,
) -> bool:
    """
    Load the environment variables from a .env file located in the caller's directory or its parents.

    Args:
        name (Path | None): Specific path to the .env file.
        ignore_env_re (str | None): Regex pattern to ignore certain .env files.
        override (bool): Whether to override existing environment variables.
        strict (bool): If True, raises an error if no .env file is found.
        verbose (bool): If True, prints loading messages.

    Returns:
        bool: True if the .env file was loaded successfully, False otherwise.

    Raises:
        ValueError: If no .env file is found and strict is True.

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
    desc: str = "Really execute?",
    yes_func: Callable | None = None,
    no_func: Callable | None = None,
    enter_is_yes: bool = False,
) -> bool:
    """
    Prompt the user for confirmation before executing an action.

    Args:
        desc (str): Description of the action to confirm.
        yes_func (Callable | None): Function to call if the user confirms.
        no_func (Callable | None): Function to call if the user declines.
        enter_is_yes (bool): If True, pressing Enter counts as a 'yes'.

    Returns:
        bool: True if the action is confirmed, False otherwise.
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


def open_file_with_default_program(path: str | Path) -> None:
    path = str(path)
    if platform.system() == 'Darwin':
        # MacOS
        os.system(f'open "{path}"')
    elif platform.system() == 'Windows':
        os.startfile(path)
    else:
        # Linux
        os.system(f'xdg-open "{path}"')
