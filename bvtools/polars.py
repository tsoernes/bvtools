#!/usr/bin/env ipython
"""A collection of tools and utilities for Polars"""

import polars as pl
import polars.selectors as cs


def drop_empty_list_columns(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """Drop columns with just empty lists"""
    for col in df.columns:
        if isinstance(df[col].dtype, pl.List):
            if df[col].list.len().sum() == 0:
                df = df.drop(col)
                if verbose:
                    print(f"Dropped empty list column {col}")
    return df


def drop_empty_list_columns_in_place(df: pl.DataFrame, verbose: bool = True) -> None:
    """Drop columns with just empty lists"""
    for col in df.columns:
        if isinstance(df[col].dtype, pl.List):
            if df[col].list.len().sum() == 0:
                df.drop_in_place(col)
                if verbose:
                    print(f"Dropped empty list column {col}")


def empty_string_to_null(
    df: pl.DataFrame, columns: str | list[str] | None = None, verbose: bool = True
) -> pl.DataFrame:
    """
    Replace empty strings with nulls. If no column(s) are given then all suitable columns are processed.
    """
    if isinstance(columns, str):
        columns = [columns]
    # Select string columns
    string_cols = columns if columns else df.select(cs.by_dtype(pl.String)).columns

    # Create a mask for empty strings
    masks = {col: df[col].str.len_bytes().eq(0) for col in string_cols}

    # Apply the mask to set empty strings to null
    df = df.with_columns(
        [
            pl.when(masks[col]).then(None).otherwise(pl.col(col)).alias(col)
            for col in string_cols
        ]
    )

    if verbose:
        for col in string_cols:
            if verbose:
                if n_empty := masks[col].sum():
                    print(f"Replaced {n_empty} empty strings with null in column {col}")

    return df


def empty_string_to_null_in_place(
    df: pl.DataFrame, columns: str | list[str] | None = None, verbose: bool = True
):
    """
    Replace empty strings with nulls, in_place.
    If no column(s) are given then all suitable columns are processed.
    """
    if isinstance(columns, str):
        columns = [columns]
    # Select string columns
    string_cols = columns if columns else df.select(cs.by_dtype(pl.String)).columns

    # Create a mask for empty strings
    masks = {col: df[col].str.len_bytes().eq(0) for col in string_cols}

    # Apply the mask to set empty strings to null
    for col in string_cols:
        df[col] = pl.when(masks[col]).then(None).otherwise(pl.col(col))
        if verbose:
            if n_empty := masks[col].sum():
                print(f"Replaced {n_empty} empty strings with null in column {col}")


def empty_list_to_null(df: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError("untested")
    # Convert empty lists to None
    df = df.with_columns(
        pl.when(cs.by_dtype(pl.List(pl.String)).list.len() > 0).then(
            cs.by_dtype(pl.List(pl.String))
        )
    )


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
    if columns:
        if isinstance(columns, str):
            struct_columns = [columns]
        else:
            struct_columns = columns
    else:
        struct_columns = [
            col for col in df.columns if isinstance(df[col].dtype, pl.Struct)
        ]

    # No JSON columns found
    if not struct_columns:
        return df

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
            print(f"Flattened '{col}' to {new_cols}")
        flattened_dfs.append(flattened)

    # Combine all DataFrames horizontally
    return pl.concat(flattened_dfs, how="horizontal")


def convert_date_columns(
    df: pl.DataFrame, date_columns: list[str], format: str = "%Y-%m-%d"
) -> pl.DataFrame:
    """Convert specified columns to date type."""
    for col in date_columns:
        if col in df.columns:
            df = df.with_columns(
                [pl.col(col).str.strptime(pl.Date, format=format).alias(col)]
            )
        else:
            print(f"{col=} not in df")
    return df


def cast_columns(
    df: pl.DataFrame, int_columns: list[str], dtype=pl.Int32
) -> pl.DataFrame:
    """Convert specified columns to given type."""
    for col in int_columns:
        if col in df.columns:
            df = df.with_columns([pl.col(col).cast(dtype).alias(col)])
        else:
            print(f"{col=} not in df")
    return df
