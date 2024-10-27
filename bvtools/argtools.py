import argparse


class ArgumentParserWithHelp(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs
        )

    def add_argument(self, *args, help="_", **kwargs):
        """
        Set help to '_' as default to force ArgumentDefaultsHelpFormatter
        to display defaults if no help is given.
        """
        return super().add_argument(*args, help=help, **kwargs)
