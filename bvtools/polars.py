"""A collection of tools and utilities for Polars"""

import polars as pl
import polars.selectors as cs


def drop_empty_list_columns(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """Drop columns with just empty lists"""
    empty_cols = [col for col in df.columns if isinstance(df[col].dtype, pl.List) and df[col].list.len().sum() == 0]

    if empty_cols and verbose:
        for col in empty_cols:
            print(f"Dropped empty list column {col}")

    return df.drop(empty_cols) if empty_cols else df


def drop_empty_list_columns_in_place(df: pl.DataFrame, verbose: bool = True) -> None:
    """Drop columns with just empty lists"""
    for col in df.columns:
        if isinstance(df[col].dtype, pl.List):
            if df[col].list.len().sum() == 0:
                df.drop_in_place(col)
                if verbose:
                    print(f"Dropped empty list column {col}")


def empty_string_to_null(
    df: pl.Series | pl.DataFrame, columns: str | list[str] | None = None, verbose: bool = True
) -> pl.Series | pl.DataFrame:
    """
    Replace empty strings with nulls. If no column(s) are given then all suitable columns are processed.
    """
    if isinstance(df, pl.Series):
        is_series = True
        df = df.to_frame()
    else:
        is_series = False

    if isinstance(columns, str):
        columns = [columns]
    # Select string columns
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

    if is_series:
        df = df.to_series(0)

    return df


def empty_list_to_null(
    df: pl.DataFrame | pl.Series,
    columns: str | list[str] | None = None,
    verbose: bool = True
) -> pl.DataFrame | pl.Series:
    """Given a polars DataFrame, convert all empty lists in the given column(s) to nulls.
    If no columns are specified, the all list columns will be processed."""
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
                print(f"{column=} not in df")
    if is_series:
        df = df.to_series(0)
    return df


def unnest_struct_columns(
    df: pl.DataFrame,
    columns: str | list[str] | None = None,
    recurse: bool = True,
    sep: str = ".",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Flatten/unnest struct (from nested JSON or dicts) columns in the DataFrame.
    If no column(s) are given then all suitable columns are processed.
    """
    # Identify JSON columns by checking for struct type
    if columns is None:
        struct_columns = [col for col in df.columns if isinstance(df[col].dtype, pl.Struct)]
    elif isinstance(columns, str):
        struct_columns = [columns]
    else:
        struct_columns = columns

    # Process each JSON column
    flattened_dfs = []
    remaining_cols = [col for col in df.columns if col not in struct_columns]

    # Keep non-JSON columns
    if remaining_cols:
        flattened_dfs.append(df.select(remaining_cols))

    # Flatten each JSON column
    for col in struct_columns:
        flattened = df.select(pl.col(col)).unnest(col)
        # Prefix column names with original column name
        new_names = {c: f"{col}{sep}{c}" for c in flattened.columns}
        flattened = flattened.rename(new_names)
        if recurse:
            flattened = unnest_struct_columns(flattened, recurse=True)
        if verbose:
            new_cols = ", ".join([f"'{c}'" for c in flattened.columns])
            print(f"Unnested '{col}' to {new_cols}")
        flattened_dfs.append(flattened)

    # Combine all DataFrames horizontally
    return pl.concat(flattened_dfs, how="horizontal")


def convert_date_columns(
    df: pl.DataFrame, columns: str | list[str], format: str = "%Y-%m-%d"
) -> pl.DataFrame:
    """Convert (parse) specified string column(s) to date type."""
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
    """Convert specified columns to given type."""
    columns_ = []
    for col in columns:
        if col in df.columns:
            columns_.append(col)
        elif verbose:
            print(f"{col=} not in df")
    df = df.with_columns([pl.col(col).cast(dtype).alias(col) for col in columns_])
    return df
