# Copyright (c) 2023 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

"""
Classes for refererencing SVD elements based on name.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from itertools import chain
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Self


# Type of path part items 
PartT = TypeVar("PartT")


class AbstractEPath(ABC, Sequence[PartT]):
    """
    A type used to describe paths of named SVD elements.
    """

    @abstractmethod
    def __init__(self, *parts: Union[PartT, Sequence[PartT]]) -> None:
        """
        :param parts: Path segments
        """
        ...

    @property
    @abstractmethod
    def parts(self) -> Tuple[PartT, ...]:
        """:return: Path components."""
        ...

    @property
    @abstractmethod
    def name(self) -> Optional[str]:
        """:return: Name of the element pointed to by the path, including any array indices."""
        ...

    @property
    @abstractmethod
    def stem(self) -> Optional[str]:
        """:return: Name of the element pointed to by the path, excluding any array indices"""
        ...

    @property
    @abstractmethod
    def parent(self) -> Optional[AbstractEPath]:
        """:return: Path to the parent element of this path, if it exists."""
        ...

    def join(self, *other: Union[PartT, Sequence[PartT]]) -> Self:
        """:return: The path resulting from appending other to the end of this path. """
        return self.__class__(*self.parts, *other)

    @overload
    def __getitem__(self, item: int, /) -> PartT:
        ...

    @overload
    def __getitem__(self, item: slice, /) -> Self:
        ...

    def __getitem__(self, item: Union[int, slice], /) -> Union[PartT, Self]:
        if isinstance(item, slice):
            return self.__class__(*self.parts[item])
        else:
            return self.parts[item]

    def __len__(self) -> int:
        return len(self.parts)

    def __hash__(self) -> int:
        return hash(self.parts)

    def __eq__(self, other: Any) -> bool:
        return self.parts == other

    @abstractmethod
    def __repr__(self) -> str:
        ...


class FEPath(AbstractEPath[str]):
    """
    Path to a flat SVD register level element.
    An FEPath like "REGISTER_A.REGISTER_B" refers to the element with name "REGISTER_B" with a
    parent with the name "REGISTER_A".

    FEPaths refer to SVD register level elements by name without considering array dimensions,
    corresponding directly to how the elements are structured in the SVD file.
    """

    __slots__ = "_parts"

    def __init__(self, *parts: Union[str, Sequence[str]]) -> None:
        split_parts: List[str] = []

        for part in parts:
            if isinstance(part, str):
                split_parts.extend(part.split("."))
            elif isinstance(part, FEPath):
                split_parts.extend(part)
            else:
                sub_parts = (p.split(".") for p in part)
                split_parts.extend(chain.from_iterable(sub_parts))

        if not split_parts:
            raise ValueError(f"Empty {self.__class__.__name__} not allowed")

        if any(not p for p in split_parts):
            raise ValueError(f"Invalid {self.__class__.__name__} parts: {parts}")

        self._parts: Tuple[str, ...] = tuple(split_parts)

    @property
    def parts(self) -> Tuple[str, ...]:
        return self._parts

    @property
    def name(self) -> str:
        return self.parts[-1]

    @property
    def parent(self) -> Optional[FEPath]:
        if len(self.parts) <= 1:
            return None
        return FEPath(*self.parts[:-1])

    @property
    def stem(self) -> str:
        return self.name

    def to_xpath(self) -> str:
        """:return: An XPath expression that can be used to locate XML elements having this path"""
        return "." + "".join((f"/*[name='{p}']" for p in self.parts))

    def __repr__(self) -> str:
        return ".".join(self.parts)


class EPath(AbstractEPath[Union[str, int]]):
    """
    Path to a SVD element.
    """

    __slots__ = "_parts"

    def __init__(self, *parts: Union[str, int, Sequence[Union[str, int]]]) -> None:
        processed_parts = self._process_parts(parts)

        if not processed_parts:
            raise ValueError(f"Empty {self.__class__.__name__} not allowed")

        self._parts: Tuple[Union[str, int], ...] = tuple(processed_parts)

    @property
    def parts(self) -> Tuple[Union[str, int], ...]:
        return self._parts

    @property
    def name(self) -> str:
        for i in reversed(range(len(self._parts))):
            if isinstance(self._parts[i], str):
                return self._format_parts(self._parts[i:])
        assert False

    @property
    def stem(self) -> str:
        for part in reversed(self._parts):
            if isinstance(part, str):
                return part
        assert False

    @property
    def parent(self) -> Optional[EPath]:
        if len(self._parts) <= 1:
            return None
        return EPath(*self._parts[:-1])

    @property
    def element_index(self) -> Optional[int]:
        """Index of the register in the parent array, if applicable."""
        if not isinstance(self[-1], int):
            return None
        return self[-1]

    def to_flat(self) -> FEPath:
        """Convert the regular path to the equivalent flat path."""
        return FEPath(*(p for p in self.parts if not isinstance(p, int)))

    def __repr__(self) -> str:
        """String representation of the path."""
        return self._format_parts(self.parts)

    def _process_parts(
        self,
        parts: Iterable[Union[str, int, Sequence[Union[str, int]]]],
        allow_seq: bool = True,
    ) -> List[Union[str, int]]:
        """Helper method for converting initialization arguments to a list of parts."""
        split_parts: List[Union[str, int]] = []

        for part in parts:
            if isinstance(part, str):
                if part.isalpha():
                    # Part is trivially correct; skip complex parsing
                    split_parts.append(part)
                else:
                    split_parts.extend(self._parse_path_str(part))

            elif isinstance(part, int):
                split_parts.append(part)

            elif allow_seq and isinstance(part, Sequence):
                if isinstance(part, EPath):
                    # Part is already processed; skip complex parsing
                    split_parts.extend(part.parts)
                else:
                    split_parts.extend(self._process_parts(part, False))

            else:
                raise TypeError(
                    f"Invalid {self.__class__.__name__} part {part} of type '{type(part)}'"
                )

        return split_parts

    def _parse_path_str(self, part: str) -> Iterable[Union[str, int]]:
        """Convert a string path to a list of parts."""
        parsed_parts: List[Union[str, int]] = []

        remaining = part
        subpart_match = re.match(r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)", remaining)
        if subpart_match is None:
            raise ValueError(f"Invalid {self.__class__.__name__} part '{part}'")

        parsed_parts.append(subpart_match["name"])
        remaining = remaining[subpart_match.end() :]

        while remaining:
            subpart_match = re.match(
                r"(?:(?:\[(?P<index>[0-9]+)\])|(?:\.(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)))",
                remaining,
            )
            if subpart_match is None:
                raise ValueError(f"Invalid {self.__class__.__name__} part '{part}'")

            remaining = remaining[subpart_match.end() :]

            if (index := subpart_match["index"]) is not None:
                parsed_parts.append(int(index, 10))
            else:
                parsed_parts.append(subpart_match["name"])

        return parsed_parts

    @staticmethod
    def _format_parts(parts: Iterable[Union[str, int]]) -> str:
        """Format parts as a string path."""
        formatted_parts: List[str] = []

        for part in parts:
            if isinstance(part, int):
                formatted_parts.append(f"[{part}]")
            else:
                if not formatted_parts:
                    formatted_parts.append(part)
                else:
                    formatted_parts.append(f".{part}")

        return "".join(formatted_parts)


# Union of SVD path types
EPathUnion = Union[EPath, FEPath]

# Generic type variable constrained to being a specific SVD path type
EPathType = TypeVar("EPathType", EPath, FEPath)
