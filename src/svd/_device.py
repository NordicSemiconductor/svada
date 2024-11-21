# Copyright (c) 2023 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

"""
Various internal functionality used by the device module.
"""

from __future__ import annotations

import re
import typing
from abc import ABC
from collections import defaultdict
from time import perf_counter_ns
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Reversible,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import lxml.etree as ET

from . import bindings
from .path import EPath


def topo_sort_derived_peripherals(
    peripherals: Iterable[bindings.PeripheralElement],
) -> List[bindings.PeripheralElement]:
    """
    Topologically sort the peripherals based on 'derivedFrom' attributes using Kahn's algorithm.
    The returned list has the property that the peripheral element at index i does not derive from
    any of the peripherals at indices 0..(i - 1).

    :param peripherals: List of peripheral elements to sort
    :return: List of peripheral elements topologically sorted based on the 'derivedFrom' attribute.
    """

    sorted_peripherals: List[bindings.PeripheralElement] = []
    no_dep_peripherals: List[bindings.PeripheralElement] = []
    dep_graph: Dict[str, List[bindings.PeripheralElement]] = defaultdict(list)

    for peripheral in peripherals:
        if peripheral.derived_from is not None:
            dep_graph[peripheral.derived_from].append(peripheral)
        else:
            no_dep_peripherals.append(peripheral)

    while no_dep_peripherals:
        peripheral = no_dep_peripherals.pop()
        sorted_peripherals.append(peripheral)
        # Each peripheral has a maximum of one in-edge since they can only derive from one
        # peripheral. Therefore, once they are encountered here they have no remaining dependencies.
        no_dep_peripherals.extend(dep_graph[peripheral.name])
        dep_graph.pop(peripheral.name, None)

    if dep_graph:
        raise ValueError(
            "Unable to determine order in which peripherals are derived. "
            "This is likely caused either by a cycle in the "
            "'derivedFrom' attributes, or a 'derivedFrom' attribute pointing to a "
            "nonexistent peripheral."
        )

    return sorted_peripherals


def remove_registers(
    peripheral_element: bindings.PeripheralElement,
    remove: Mapping[str, Sequence[str]],
) -> None:
    """
    Remove clusters/registers from a peripheral by deleting the nodes from the XML tree itself.

    :param peripheral_element: Peripheral node to filter registers from.
    :param remove:
    """
    registers = peripheral_element._registers
    if registers is None:
        # Skip if the node has no <registers> node (permitted on derived peripherals)
        return

    for pattern_str, paths in remove.items():
        if re.fullmatch(pattern_str, peripheral_element.name) is None:
            continue

        for path in paths:
            xpath = "." + "".join((f"/*[name='{p}']" for p in path.split(".")))
            nodes = typing.cast(Iterable[ET._Element], registers.xpath(xpath))
            for node in nodes:
                if (parent := node.getparent()) is not None:
                    parent.remove(node)


def svd_element_repr(
    klass: type,
    name: str,
    /,
    *,
    address: Optional[int] = None,
    length: Optional[int] = None,
    content: Optional[int] = None,
    content_max_width: int = 32,
    bool_props: Iterable[Any] = (),
    kv_props: Mapping[Any, Any] = MappingProxyType({}),
) -> str:
    """
    Common pretty print function for SVD elements.

    :param klass: Class of the element.
    :param name: Name of the element.
    :param address: Address of the element.
    :param length: Address of the element.
    :param content: Length of the element.
    :param content_max_width: Available width of the element, used to zero-pad the value.
    :param bool_props: Additional arguments to include in the pretty print.
    :param kv_props: Additional keyword arguments to include in the pretty print.

    :return: Pretty printed string representing the element.
    """

    address_str: str = f" @ 0x{address:08x}" if address is not None else ""
    length_str: str = f"<{length}>" if length is not None else ""
    value_str: str

    if content is not None:
        leading_zeros: str = "0" * ((content_max_width - content.bit_length()) // 4)
        value_str = f" = 0x{leading_zeros}{content:x}"
    else:
        value_str = ""

    if bool_props or kv_props:
        bool_props_str: str = (
            f"{', '.join(f'{v!s}' for v in bool_props)}" if bool_props else ""
        )
        kv_props_str: str = (
            f"{', '.join(f'{k}: {v!s}' for k, v in kv_props.items())})"
            if kv_props
            else ""
        )
        props_str = f" ({bool_props_str}{', ' if kv_props else ''}{kv_props_str})"
    else:
        props_str = ""

    return (
        f"[{name}{length_str}{address_str}{value_str}{props_str} {{{klass.__name__}}}]"
    )


def strip_suffix(word: str, suffix: str) -> str:
    """
    Remove the given suffix from the word, if present.

    :param word: String to strip prefixes and suffixes from.
    :param suffix: Suffix to strip.

    :return: word without the suffix.
    """

    if word.endswith(suffix):
        word = word[: -len(suffix)]

    return word


ItemT = TypeVar("ItemT")


class ChildIter(Reversible[ItemT]):
    """Helper class used as a generic reversible iterator"""

    def __init__(self, keys: Reversible, getter: Callable[[Any], ItemT]) -> None:
        self._keys = keys
        self._getter = getter

    def __iter__(self) -> Iterator[ItemT]:
        for k in self._keys:
            yield self._getter(k)

    def __reversed__(self) -> Iterator[ItemT]:
        for k in reversed(self._keys):
            yield self._getter(k)


class LazyFixedMapping(Mapping[str, ItemT]):
    """
    A mapping that lazily constructs its values.
    The set of keys is fixed at construction time - this ensures consistent ordering during
    iteration.
    """

    def __init__(
        self,
        keys: Iterable[str],
        factory: Callable[[str], ItemT],
        **kwargs: Any,
    ) -> None:
        """
        :param keys: Keys contained in the mapping.
        :param factory: Factory function which is called to initialize
                        new elements.
        """
        self._storage: Dict[str, Optional[ItemT]] = {key: None for key in keys}
        self._factory = factory

        super().__init__(**kwargs)

    def __getitem__(self, key: Union[str, Sequence[Union[str, Any]]], /) -> ItemT:
        remaining_key: Sequence[Union[str, Any]]

        if isinstance(key, str):
            this_key = key
            remaining_key = ()
        else:
            this_key = key[0]
            remaining_key = key[1:] if len(key) > 1 else ()

        if not isinstance(key, str):
            raise TypeError(f"Key {key} has invalid type {type(key)}. Expected a str.")

        value = self._storage[this_key]
        if value is None:
            value = self._factory(key)
            self._storage[this_key] = value

        if remaining_key:
            if not hasattr(value, "__getitem__"):
                raise LookupError(
                    f"̈́Invalid key for {value!r}: {remaining_key}. "
                    "The object is not subscriptable."
                )
            else:
                return value[remaining_key]
        else:
            return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._storage)

    def __contains__(self, key: Any) -> bool:
        return key in self._storage

    def __len__(self) -> int:
        return len(self._storage)


class _End:
    ...


# Sentinel value for the end of an iterator
_END = _End()


def iter_merged(a: Iterable[ItemT], b: Iterable[ItemT], key: Callable[[ItemT], Any]) -> Iterator[ItemT]:
    """
    Iterator that merges two sorted iterables.
    It is assumed that both iterables are sorted - passing unsorted iterables will not
    cause an error.
    """

    iter_a = iter(a)
    iter_b = iter(b)

    item_a = next(iter_a, _END)
    item_b = next(iter_b, _END)

    while True:
        if not isinstance(item_a, _End) and not isinstance(item_b, _End):
            key_a = key(item_a)
            key_b = key(item_b)
            if key_a <= key_b:
                yield item_a
                item_a = next(iter_a, _END)
            else:
                yield item_b
                item_b = next(iter_b, _END)
        elif not isinstance(item_a, _End):
            yield item_a
            item_a = next(iter_a, _END)
        elif not isinstance(item_b, _End):
            yield item_b
            item_b = next(iter_b, _END)
        else:
            break
