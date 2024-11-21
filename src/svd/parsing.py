# Copyright (c) 2022 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses as dc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import lxml.etree as ET
from lxml import objectify

from . import bindings
from .device import Device
from .errors import SvdParseError

if TYPE_CHECKING:
    from ._bindings import SvdElement


@dataclass(frozen=True)
class Options:
    """Options to configure the SVD parsing behavior."""

    # Ignore overlapping clusters/registers described in the SVD file.
    # If set to False, an exception is raised when structures overlap.
    ignore_overlapping_structures: bool = False

    # Make the addressOffset in cluster elements be relative to the immediate parent
    # element instead of relative to the containing peripheral.
    parent_relative_cluster_address: bool = True

    # Cluster/register/field elements to remove from the XML document prior to parsing.
    # This can be used to remove outdated/deprecated elements from the device if they cause
    # issues with parsing.
    #
    # The value should be a dictionary mapping string peripheral name regex patterns to lists
    # containing the paths of elements to remove from the peripherals matching the pattern.
    # For example, passing the value
    # {"UART[0-9]": ["CONFIG.DEPRECATED"]}
    # would cause the element named "DEPRECATED" to be removed from the element named "CONFIG"
    # in peripherals whose name match "UART[0-9]" (UART0, UART1 etc.).
    #
    # Note: the element paths must match exactly the names used in the SVD document.
    # Note: no exception is raised if the paths given don't match anything in the SVD document.
    skip_registers: Mapping[str, Sequence[str]] = dc.field(
        default_factory=lambda: defaultdict(list)
    )


def parse(svd_path: Union[str, Path], options: Options = Options()) -> Device:
    """
    Parse a device described by a SVD file.

    :param svd_path: Path to the SVD file.
    :param options: Parsing options.

    :raises FileNotFoundError: If the SVD file does not exist.
    :raises SvdParseError: If an error occurred while parsing the SVD file.

    :return: Parsed `Device` representation of the SVD file.
    """

    t_parse_start = perf_counter_ns()

    svd_file = Path(svd_path)

    if not svd_file.is_file():
        raise FileNotFoundError(f"No such file: {svd_file.absolute()}")

    try:
        # Note: remove comments as otherwise these are present as nodes in the returned XML tree
        xml_parser = objectify.makeparser(remove_comments=True)
        class_lookup = _TwoLevelTagLookup(bindings.BINDINGS)
        xml_parser.set_element_class_lookup(class_lookup)

        with open(svd_file, "rb") as f:
            xml_device = objectify.parse(f, parser=xml_parser)

        t_parse = (perf_counter_ns() - t_parse_start) / 1_000_000
        device = Device(xml_device.getroot(), options=options, time_parse=t_parse)

    except Exception as e:
        raise SvdParseError(f"Error parsing SVD file {svd_file}") from e

    return device


class _TwoLevelTagLookup(ET.ElementNamespaceClassLookup):
    """
    XML element class lookup that uses two levels of tag names to map an XML element to a Python
    class. This two-level scheme is used to slightly optimize the time spent by the parser looking
    up looking up the class for an element (which is a sizeable portion of the time spent parsing).

    Element classes that can be uniquely identified by tag only are stored in the first level.
    This level uses the lxml ElementNamespaceClassLookup which is faster than the second level.
    The remaining element classes are assumed to be uniquely identified by a combination of
    the parent tag and the tag itself, and are stored in the second level.
    The second level uses the lxml PythonElementClassLookup which is slower.
    """

    def __init__(self, element_classes: List[Type[SvdElement]]):
        """
        :param element_classes: lxml element classes to add to the lookup table.
        """
        super().__init__()

        tag_classes: Dict[str, Set[type]] = defaultdict(set)
        two_tag_classes: Dict[Tuple[Optional[str], str], Set[type]] = defaultdict(set)

        for element_class in element_classes:
            tag = element_class.TAG
            tag_classes[tag].add(element_class)
            for prop in bindings.get_binding_elem_props(element_class).values():
                tag_classes[prop.name].add(prop.element_class)
                two_tag_classes[(tag, prop.name)].add(prop.element_class)

        one_tag: Set[str] = set()
        namespace = self.get_namespace(None)  # None is the empty namespace

        for tag, classes in tag_classes.items():
            if len(classes) == 1:
                # Add the class to the namespace
                # namespace is a decorator, so the syntax here is a little odd
                element_class = classes.pop()
                namespace(tag)(element_class)
                one_tag.add(tag)

        two_tag_lookup: Dict[Tuple[Optional[str], str], type] = {}

        for (parent_tag, field_name), classes in two_tag_classes.items():
            if field_name in one_tag:
                continue

            if len(classes) != 1:
                raise RuntimeError(
                    f"Multiple classes for ({parent_tag}, {field_name}): {classes}. "
                    "This should never happen, and likely indicates a bug in the way element "
                    "class lookup is implemented."
                )

            two_tag_lookup[(parent_tag, field_name)] = classes.pop()

        fallback_lookup = _SecondLevelTagLookup(two_tag_lookup)
        self.set_fallback(fallback_lookup)


class _SecondLevelTagLookup(ET.PythonElementClassLookup):
    """XML element class lookup table that uses two levels of tags to look up the class"""

    def __init__(
        self,
        lookup_table: Dict[
            Tuple[Optional[str], str], Type[objectify.ObjectifiedElement]
        ],
    ):
        """
        :param lookup_table: Lookup table mapping a tuple of (parent tag, tag) to an element class.
        """
        self._lookup_table = lookup_table

    def lookup(
        self, _document: Any, element: ET._Element
    ) -> Optional[Type[objectify.ObjectifiedElement]]:
        """Look up the Element class for the given XML element"""
        if (parent := element.getparent()) is not None:
            parent_tag = parent.tag
        else:
            parent_tag = None
        return self._lookup_table.get((parent_tag, element.tag))
