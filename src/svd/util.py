#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from typing import List


def strip_prefixes_suffixes(word: str, prefixes: List[str], suffixes: List[str]) -> str:
    """
    Emulates the functionality provided by chaining `removeprefix` and `removesuffix`
    to a str object.

    :param word: String to strip prefixes and suffixes from.
    :param prefixes: List of prefixes to strip.
    :param suffixes: List of suffixes to strip.

    :return: String where prefixes and suffixes have been sequentially removed.
    """

    for prefix in prefixes:
        if word.startswith(prefix):
            word = word[len(prefix) :]

    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[: -len(suffix)]

    return word.strip("_")


def to_int(number: str) -> int:
    """
    Convert a string representation of an integer following the SVD format to its corresponding
    integer representation.

    :param number: String representation of the integer.

    :return: Decoded integer.
    """

    if number.startswith("0x"):
        return int(number, base=16)

    if number.startswith("#"):
        return int(number[1:], base=2)

    return int(number)
