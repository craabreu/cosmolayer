import re

import pandas as pd


def parse_table(
    contents: str,
    row_regex: re.Pattern[str],
    section_regex: re.Pattern[str],
    schema: dict[str, type],
) -> pd.DataFrame:
    section_match = section_regex.search(contents)
    if not section_match:
        raise ValueError("Could not parse table information.")
    rows = [
        {
            title: converter(value)
            for (title, converter), value in zip(
                schema.items(), row_match.groups(), strict=True
            )
        }
        for row_match in row_regex.finditer(section_match.group(1))
    ]
    return pd.DataFrame(rows, columns=schema.keys())


def parse_value(contents: str, regex: re.Pattern[str]) -> float:
    match = regex.search(contents)
    if not match:
        raise ValueError("Could not parse value.")
    return float(match.group(1))
