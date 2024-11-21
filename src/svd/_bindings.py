# Copyright (c) 2023 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

"""
Various internal functionality used by the bindings module.
"""

from __future__ import annotations

import enum
import inspect
import typing
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)

from lxml import objectify
from typing_extensions import Self


class CaseInsensitiveStrEnum(enum.Enum):
    """String enum class that can be constructed from a case-insensitive string."""

    @classmethod
    def _missing_(cls, value: object) -> Optional[Self]:
        """Handler for string values with mismatched case."""
        if not isinstance(value, str):
            return None

        value_lower = value.lower()
        for member in cls:
            if member.value.lower() == value_lower:
                return member

        return None


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


def to_bool(value: str) -> bool:
    """
    Convert a string representation of a boolean following the SVD format to its corresponding
    boolean representation.

    :param value: String representation of the boolean.

    :return: Decoded boolean.
    """
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


class SvdElement(objectify.ObjectifiedElement):
    """Base class for all the SVD element classes."""

    TAG: str

    def __repr__(self) -> str:
        """
        A more informative string representation than the default one from lxml.
        This is mostly useful for the exception tracebacks that occur on parsing errors.
        """
        return self._repr()

    def _repr(
        self,
        props: Mapping[Any, Any] = MappingProxyType({}),
    ) -> str:
        """
        Default repr() implementation for the binding classes.
        """

        props = dict(props)

        parent = self.getparent()
        if parent is not None:
            ancestors_str = f" in {parent!r}"
            try:
                child_index = parent.index(self)
            except Exception:
                child_index = None
        else:
            ancestors_str = ""
            child_index = None

        child_index_str = f"({child_index})" if child_index is not None else ""
        props_str = f" {props}" if props else ""
        self_repr = f"[{self.tag}{child_index_str}{props_str}]"

        return f"{self_repr}{ancestors_str}"


class SvdIntElement(objectify.IntElement):
    """
    Element containing an SVD integer value.
    This class uses a custom parser to convert the value to an integer.
    """

    def _init(self) -> None:
        self._setValueParser(to_int)


class _Self:
    ...


# Sentinel value used for element properties where the result class is equal to the class of the
# parent object. This is required since self-referential class members are not allowed.
SELF_CLASS = _Self()


class _Missing:
    ...


# Sentinel value used to indicate that a default value is missing.
MISSING = _Missing()


O = TypeVar("O", bound=objectify.ObjectifiedElement)
T = TypeVar("T")


class Elem(Generic[T]):
    """Data descriptor class used to access a XML element."""

    def __init__(
        self,
        name: str,
        element_class: Union[Type[objectify.ObjectifiedElement], _Self],
        /,
        *,
        default: Union[T, _Missing] = MISSING,
        default_factory: Union[Callable[[], T], _Missing] = MISSING,
    ) -> None:
        """
        Create a data descriptor object that extracts an element from an XML node.
        Only one of default or default_factory can be set.

        :param name: Name of the element.
        :param element_class: Class to use for the extracted element.
        :param default: Default value to return if the element is not found.
        :param default_factory: Callable that returns the default value to return if the element is
                                not found.
        """
        if default != MISSING and default_factory != MISSING:
            raise ValueError("Cannot set both default and default_factory")

        self.name: str = name
        self.element_class: Type[
            objectify.ObjectifiedElement
        ] = element_class  # type:ignore
        self.default: Union[T, _Missing] = default
        self.default_factory: Union[Callable[[], T], _Missing] = default_factory

    @overload
    def __get__(self, node: Literal[None], owner: Optional[Type] = None) -> Self:
        ...

    @overload
    def __get__(self, node: O, owner: Optional[Type] = None) -> T:
        ...

    def __get__(self, node: Optional[O], owner: Any = None) -> Union[T, Self]:
        """Get the element value from the given node."""
        if node is None:
            # If the node argument is None, we are being accessed through the class object.
            # In that case, return the descriptor itself.
            return self

        try:
            svd_obj = node.__getattr__(self.name)
        except AttributeError:
            if not isinstance(self.default_factory, _Missing):
                return self.default_factory()
            if not isinstance(self.default, _Missing):
                return self.default
            raise

        if issubclass(self.element_class, objectify.ObjectifiedDataElement):
            return svd_obj.pyval  # type: ignore
        else:
            return svd_obj  # type: ignore


class Attr(Generic[T]):
    """Data descriptor used to access a XML attribute."""

    def __init__(
        self,
        name: str,
        /,
        *,
        converter: Optional[Callable[[str], T]] = None,
        default: Union[T, _Missing] = MISSING,
        default_factory: Union[Callable[[], T], _Missing] = MISSING,
    ) -> None:
        """
        Create a data descriptor object that extracts an attribute from an XML node.
        Only one of default or default_factory can be set.

        :param name: Name of the attribute.
        :param converter: Optional callable that converts the attribute value from a string to another
                        type.
        :param default: Default value to return if the element is not found.
        :param default_factory: Callable that returns the default value to return if the element is
                                not found.
        """
        if default != MISSING and default_factory != MISSING:
            raise ValueError("Cannot set both default and default_factory")

        self.name: str = name
        self.converter: Optional[Callable[[str], T]] = converter
        self.default: Union[T, _Missing] = default
        self.default_factory: Union[Callable[[], T], _Missing] = default_factory

    @overload
    def __get__(self, node: Literal[None], owner: Optional[Type] = None) -> Self:
        ...

    @overload
    def __get__(self, node: O, owner: Optional[Type] = None) -> T:
        ...

    def __get__(self, node: Optional[O], owner: Any = None) -> Union[T, Self]:
        """Get the attribute value from the given node."""
        if node is None:
            # If the node argument is None, we are being accessed through the class object.
            # In that case, return the descriptor itself.
            return self

        value = node.get(self.name)

        if value is None:
            if not isinstance(self.default_factory, _Missing):
                return self.default_factory()
            if not isinstance(self.default, _Missing):
                return self.default
            raise AttributeError(f"Attribute {self.name} was not found")

        if self.converter is None:
            return value  # type: ignore

        try:
            return self.converter(value)
        except Exception as e:
            raise ValueError(f"Error converting attribute {self.name}") from e


C = TypeVar("C", bound=SvdElement)


class BindingRegistry:
    """Simple container for XML binding classes."""

    def __init__(self) -> None:
        self._element_classes: List[Type[SvdElement]] = []

    def add(
        self,
        element_class: Type[C],
        /,
    ) -> Type[C]:
        """
        Add a class to the binding registry.
        This is intended to be used as a class decorator.
        """
        elem_props: Dict[str, Elem] = getattr(element_class, "_xml_elem_props", {})
        attr_props: Dict[str, Attr] = getattr(element_class, "_xml_attr_props", {})

        for name, prop in inspect.getmembers(element_class):
            if isinstance(prop, Elem):
                if prop.element_class == SELF_CLASS:
                    prop.element_class = element_class
                elem_props[name] = prop

            elif isinstance(prop, Attr):
                attr_props[name] = prop

        setattr(element_class, "_xml_elem_props", elem_props)
        setattr(element_class, "_xml_attr_props", attr_props)

        self._element_classes.append(element_class)

        return element_class

    @property
    def bindings(self) -> List[Type[SvdElement]]:
        """Get the list of registered bindings."""
        return self._element_classes


def get_binding_elem_props(
    klass: Type[objectify.ObjectifiedElement],
) -> Mapping[str, Elem]:
    """Get the XML element properties of a binding class."""
    try:
        return klass._xml_elem_props  # type: ignore
    except AttributeError as e:
        raise ValueError(f"Class {klass} is not a binding") from e


def make_enum_wrapper(
    enum_cls: Type[CaseInsensitiveStrEnum],
) -> Type[SvdElement]:
    """
    Factory for creating lxml.objectify.ObjectifiedDataElement wrappers around
    CaseInsensitiveStrEnum subclasses.
    """

    class EnumWrapper(SvdElement, objectify.ObjectifiedDataElement):
        @property
        def pyval(self) -> CaseInsensitiveStrEnum:
            return enum_cls(self.text)

        def __repr__(self) -> str:
            props: Dict[str, Any]

            try:
                enum_value = self.pyval
                props = {"pyval": str(enum_value)}
            except Exception:
                props = {"text": self.text}

            return super()._repr(props=props)

    return EnumWrapper


def iter_element_children(
    element: Optional[objectify.ObjectifiedElement], *tags: str
) -> Iterable[objectify.ObjectifiedElement]:
    """
    Iterate over the children of an lxml element, optionally filtered by tag.
    If the element is None, an empty iterator is returned.
    """
    if element is None:
        return iter(())

    child_iter = element.iterchildren(*tags)  # type: ignore
    return typing.cast(Iterable[objectify.ObjectifiedElement], child_iter)
