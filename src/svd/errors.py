# Copyright (c) 2023 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Iterable, Union

from .path import EPathUnion


class SvdError(Exception):
    """Base class for errors raised by the library."""

    ...


class SvdParseError(SvdError):
    """Raised when an error occurs during SVD parsing."""

    ...


class SvdDefinitionError(SvdError, ValueError):
    """Raised when unrecoverable errors occur due to an invalid definition in the SVD file."""

    def __init__(self, bindings: Iterable[Any], explanation: str):
        bindings_str = "\n".join(f"  * {b!r}" for b in bindings)
        super().__init__(f"Invalid SVD file element(s):\n{bindings_str}\n{explanation}")


class SvdMemoryError(SvdError, BufferError):
    """Raised when an invalid memory operation was attempted on a SVD element."""

    ...


class SvdPathError(SvdError):
    """Raised when trying to access a nonexistent/invalid SVD path."""

    def __init__(
        self, path: Union[str, EPathUnion], source: Any, explanation: str = ""
    ) -> None:
        formatted_explanation = "" if not explanation else f" ({explanation})"
        message = (
            f"{source!s} does not contain an element '{path}'{formatted_explanation}"
        )

        super().__init__(message)


class SvdIndexError(SvdPathError, IndexError):
    """Raised when given an invalid index in an array instance."""

    ...


class SvdKeyError(SvdPathError, KeyError):
    """Raised when given an invalid child element name in a struct or register instance. """

    ...
