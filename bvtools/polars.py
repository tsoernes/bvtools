"""A collection of tools and utilities for Polars"""

import polars as pl
import polars.selectors as cs
from typing import TypeVar, Sequence, Any

T = TypeVar('T', pl.Series, pl.DataFrame)

def drop_empty_list_columns(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """
    Drop columns with just empty lists.

    Args:
        df (pl.DataFrame): Input DataFrame.
        verbose (bool, optional): If True, print information about dropped columns. Defaults to True.

    Returns:
        pl.DataFrame: DataFrame with empty list columns removed.
    """
    empty_cols = [col for col in df.columns if isinstance(df[col].dtype, pl.List) and df[col].list.len().sum() == 0]

    if empty_cols and verbose:
        for col in empty_cols:
            print(f"Dropped empty list column {col}")

    return df.drop(empty_cols) if empty_cols else df


def drop_empty_list_columns_in_place(df: pl.DataFrame, verbose: bool = True) -> None:
    """
    Drop columns with just empty lists in-place.

    Args:
        df (pl.DataFrame): Input DataFrame (modified in-place).
        verbose (bool, optional): If True, print information about dropped columns. Defaults to True.
    """
    for col in df.columns:
        if isinstance(df[col].dtype, pl.List):
            if df[col].list.len().sum() == 0:
                df.drop_in_place(col)
                if verbose:
                    print(f"Dropped empty list column {col}")


def empty_string_to_null(
    df: T, columns: str | list[str] | None = None, verbose: bool = True
) -> T:
    """
    Replace empty strings with nulls in specified columns or all string columns.

    Args:
        df: Input Series or DataFrame.
        columns: Column(s) to process. If None, all string columns are processed.
        verbose: If True, print information about replacements. Defaults to True.

    Returns:
        Series or DataFrame with empty strings replaced by nulls.
    """
    if isinstance(df, pl.Series):
        is_series = True
        df = df.to_frame()
    else:
        is_series = False

    if isinstance(columns, str):
        columns = [columns]

    string_cols = columns if columns else df.select(cs.by_dtype(pl.String)).columns

    # Create a mask for empty strings
    masks = {col: df[col].str.len_bytes().eq(0) for col in string_cols}

    # Apply the mask to set empty strings to null
    df = df.with_columns(
            pl.when(masks[col]).then(None).otherwise(pl.col(col)).alias(col)
            for col in string_cols
    )

    if verbose:
        for col in string_cols:
            if n_empty := masks[col].sum():
                print(f"Replaced {n_empty} empty strings with null in column {col}")

    return df.to_series(0) if is_series else df


def empty_list_to_null(
        df: T,
    columns: str | list[str] | None = None,
    verbose: bool = True
) -> T:
    """
    Convert all empty lists in the given column(s) to nulls.

    Args:
        df (Union[pl.Series, pl.DataFrame]): Input Series or DataFrame.
        columns (Optional[Union[str, List[str]]]): Column(s) to process. If None, all list columns are processed.
        verbose (bool, optional): If True, print information about conversions. Defaults to True.

    Returns:
        Union[pl.Series, pl.DataFrame]: Series or DataFrame with empty lists replaced by nulls.
    """

    if isinstance(df, pl.Series):
        is_series = True
        df = df.to_frame()
    else:
        is_series = False

    if columns is None:
        columns = [col for col in df.columns if isinstance(df[col].dtype, pl.List)]
    elif isinstance(columns, str):
        columns = [columns]

    df = df.with_columns(
        pl.when(pl.col(column).list.len().eq(0))
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
        for column in columns
    )

    if verbose:
        for column in columns:
            if column not in df.columns:
                print(f"Warning: Column '{column}' not found in DataFrame")

    return df.to_series(0) if is_series else df


def unnest_struct_columns(
    df: pl.DataFrame,
    columns: str | list[str] | None = None,
    recurse: bool = True,
    sep: str = ".",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Flatten/unnest struct (from nested JSON or dicts) columns in the DataFrame.

    Args:
        df (pl.DataFrame): Input DataFrame.
        columns (Optional[Union[str, List[str]]]): Column(s) to unnest. If None, all struct columns are processed.
        recurse (bool, optional): If True, recursively unnest nested structs. Defaults to True.
        sep (str, optional): Separator for nested column names. Defaults to ".".
        verbose (bool, optional): If True, print information about unnested columns. Defaults to True.

    Returns:
        pl.DataFrame: DataFrame with unnested columns.
    """
    # Identify JSON columns by checking for struct type
    if columns is None:
        struct_columns = [col for col in df.columns if isinstance(df[col].dtype, pl.Struct)]
    elif isinstance(columns, str):
        struct_columns = [columns]
    else:
        struct_columns = columns

    flattened_dfs = []
    remaining_cols = [col for col in df.columns if col not in struct_columns]

    # Keep non-struct columns
    if remaining_cols:
        flattened_dfs.append(df.select(remaining_cols))

    # Flatten each struct column
    for col in struct_columns:
        flattened = df.select(col).unnest(col)
        # Prefix column names with original column name
        new_names = {c: f"{col}{sep}{c}" for c in flattened.columns}
        flattened = flattened.rename(new_names)
        if recurse:
            flattened = unnest_struct_columns(flattened, recurse=True, sep=sep, verbose=verbose)
        if verbose:
            new_cols = ", ".join([f"'{c}'" for c in flattened.columns])
            print(f"Unnested '{col}' to {new_cols}")
        flattened_dfs.append(flattened)

    return pl.concat(flattened_dfs, how="horizontal")


def convert_date_columns(
    df: pl.DataFrame, columns: str | list[str], format: str = "%Y-%m-%d"
) -> pl.DataFrame:
    """
    Convert (parse) specified string column(s) to date type.

    Args:
        df (pl.DataFrame): Input DataFrame.
        columns (Union[str, List[str]]): Column(s) to convert.
        format (str, optional): Date format string. Defaults to "%Y-%m-%d".

    Returns:
        pl.DataFrame: DataFrame with converted date columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    df = df.with_columns(
        [pl.col(col).str.strptime(pl.Date, format=format).alias(col)]
        for col in columns
    )
    return df


def cast_columns(
        df: pl.DataFrame, columns: list[str], dtype, verbose: bool = True,
) -> pl.DataFrame:
    """
    Convert specified columns to given type.

    Args:
        df (pl.DataFrame): Input DataFrame.
        columns (List[str]): Columns to cast.
        dtype (pl.DataType): Target data type.
        verbose (bool, optional): If True, print warnings for missing columns. Defaults to True.

    Returns:
        pl.DataFrame: DataFrame with casted columns.
    """
    """Convert specified columns to given type."""
    columns_ = []
    for col in columns:
        if col in df.columns:
            columns_.append(col)
        elif verbose:
            print(f"{col=} not in df")
    df = df.with_columns([pl.col(col).cast(dtype).alias(col) for col in columns_])
    return df


def first_nonempty_value(s: pl.Series) -> Any:
    """Return the first value that is not Null, NaN or a String or a List of length 0"""
    if s.dtype.is_numeric():
        s =  s.drop_nulls().drop_nans()
        if len(s):
            return s.item(0)
        return pl.Null
    elif isinstance(s.dtype, pl.List):
        s = s.filter(s.list.len() > 0).drop_nulls()
        if len(s):
            return s.item(0)
        return pl.Null
    elif isinstance(s.dtype, pl.String):
        s = s.filter(s.str.len_bytes() > 0).drop_nulls()
        if len(s):
            return s.item(0)
        return pl.Null
    else:
        s = s.drop_nulls()
        if len(s):
            return s.item(0)
        return pl.Null


