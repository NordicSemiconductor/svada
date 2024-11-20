#
# Copyright (c) 2024 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Optional

import svd


@dataclass
class BuildSelector:
    """Used to select select which parts of the device to write to an output."""

    class ContentStatus(str, enum.Enum):
        ANY = "any"
        WRITTEN = "written"
        MODIFIED = "modified"

    peripherals: Optional[list[str]] = None
    address_range: Optional[tuple[int, int]] = None
    content_status: ContentStatus = ContentStatus.ANY

    def is_periph_selected(self, periph: svd.Peripheral) -> bool:
        if self.peripherals and periph.name not in self.peripherals:
            return False

        if self.address_range is not None and not _ranges_overlap_inclusive(
            *periph.address_bounds, *self.address_range
        ):
            return False

        return True

    def is_addr_selected(self, address: int) -> bool:
        if self.address_range is not None and not (
            self.address_range[0] <= address <= self.address_range[1]
        ):
            return False
        return True

    def is_reg_selected(self, reg: svd.Register) -> bool:
        if self.address_range is not None:
            reg_addr_range = reg.address_range
            if not _ranges_overlap_inclusive(
                reg_addr_range.start, reg_addr_range.stop - 1, *self.address_range
            ):
                return False

        if (
            self.content_status == BuildSelector.ContentStatus.WRITTEN
            and not reg.written
        ):
            return False

        if (
            self.content_status == BuildSelector.ContentStatus.MODIFIED
            and not reg.modified
        ):
            return False

        return True

    def is_field_selected(self, field: svd.Field) -> bool:
        # svada does not have a way to check written status at this level currently
        if (
            self.content_status == BuildSelector.ContentStatus.MODIFIED
            and not field.modified
        ):
            return False

        return True


class DeviceBuilder:
    """Used to populate peripheral registers and output the register contents in various formats."""

    def __init__(
        self, device: svd.Device, enforce_svd_constraints: bool = True
    ) -> None:
        self._device = device
        self._enforce_svd_constraints = enforce_svd_constraints
        self._written_peripherals = {}
        self._cached_device_periph_map = []
        self._cached_periph_reg_maps = {}

    @property
    def device(self) -> svd.Device:
        """The device structure the builder was initialized with."""
        return self._device

    @property
    def written_peripherals(self) -> list[svd.Peripheral]:
        """The list of peripherals that have been written to as part of the API calls."""
        return list(self._written_peripherals.values())

    def build_bytes(self, selector: BuildSelector = BuildSelector()) -> bytearray:
        """Encode device content as bytes.

        :param selector: selected parts of the device.
        :returns: content bytes.
        """
        memory = self.build_memory(selector)
        out = bytearray()

        for (addr_a, value_a), (addr_b, value_b) in pairwise(memory.items()):
            if not out:
                out.append(value_a)

            num_empty = addr_b - addr_a - 1
            if num_empty > 0:
                # TODO: 0 may not always be valid
                out.extend([0] * num_empty)

            out.append(value_b)

        return out

    def build_memory(self, selector: BuildSelector = BuildSelector()) -> dict[int, int]:
        """Encode device content as a mapping between address and value.

        :param selector: selected parts of the device.
        :returns: content memory map.
        """
        memory = {}

        for peripheral in self._device.values():
            if not selector.is_periph_selected(peripheral):
                continue

            if selector.content_status == BuildSelector.ContentStatus.MODIFIED:
                # There's no good way to determine this in svada at the moment
                periph_modified_filter = set()
                for reg in peripheral.register_iter(leaf_only=True):
                    if reg.modified:
                        periph_modified_filter.update(reg.address_range)
            else:
                periph_modified_filter = None

            memory_iter = peripheral.memory_iter(
                absolute_addresses=True,
                written_only=(
                    selector.content_status == BuildSelector.ContentStatus.WRITTEN
                ),
            )

            for addr, val in memory_iter:
                if (
                    periph_modified_filter is not None
                    and addr not in periph_modified_filter
                ):
                    continue
                memory[addr] = val

        return memory

    def build_dict(self, selector: BuildSelector = BuildSelector()) -> dict:
        """Encode device content as a dictionary representation of the registers and content.

        :param selector: selected parts of the device.
        :returns: content dictionary.
        """
        config = {}

        for peripheral in self._device.values():
            if not selector.is_periph_selected(peripheral):
                continue

            cfg_periph = {}
            nonleaf_stack = [cfg_periph]
            for reg in peripheral.register_iter():
                reg_depth = len(reg.path)
                if reg_depth < len(nonleaf_stack):
                    # We are returning from a nested context.
                    # Prune any empty supertables on the way up.
                    for elem in list(nonleaf_stack[reg_depth:]):
                        for key, val in list(elem.items()):
                            if not val:
                                del elem[key]

                    nonleaf_stack = nonleaf_stack[:reg_depth]

                if not reg.leaf:
                    reg_table = {}
                    nonleaf_stack[reg_depth - 1][str(reg.path[-1])] = reg_table
                    nonleaf_stack.append(reg_table)
                    continue

                assert isinstance(reg, svd.Register)

                if not selector.is_reg_selected(reg):
                    continue

                if reg.fields:
                    reg_table = {}

                    for field_name, field in reg.fields.items():
                        if not selector.is_field_selected(field):
                            continue

                        try:
                            reg_table[field_name] = field.content_enum
                        except LookupError:
                            # Content does not match any defined enum
                            reg_table[field_name] = field.content

                    # Include if at least one field was selected
                    if reg_table:
                        nonleaf_stack[reg_depth - 1][str(reg.path[-1])] = reg_table
                else:
                    # No fields, just a value
                    nonleaf_stack[reg_depth - 1][str(reg.path[-1])] = reg.content

            # Prune empty tables from the top level config
            for key, val in list(cfg_periph.items()):
                if not val:
                    del cfg_periph[key]

            if cfg_periph:
                config[peripheral.name] = cfg_periph
            else:
                # Ensure a non-empty config file
                config[peripheral.name] = {}

        return config

    def apply_memory(self, content: dict[int, int]) -> DeviceBuilder:
        """Update device content based on a memory map.

        The content is assumed to be at byte granularity and sorted by ascending address.

        :param content: content memory map.
        :returns: builder instance.
        """
        periph_map = self._device_periph_map()
        reg_map = {}
        current_periph = None
        current_periph_range = range(-1, 0)
        current_periph_regs = {}

        map_iter = iter(content.items())

        while True:
            try:
                addr_0, val_0 = next(map_iter)
            except StopIteration:
                break

            if addr_0 not in current_periph_range:
                for periph_range, periph in periph_map:
                    if addr_0 in periph_range:
                        current_periph = periph
                        current_periph_range = periph_range
                        periph_id = _get_periph_id(periph)
                        if periph_id in reg_map:
                            current_periph_regs = reg_map[periph_id]
                        else:
                            current_periph_regs = self._periph_reg_map(current_periph)
                            reg_map[periph_id] = current_periph_regs
                        self._written_peripherals.setdefault(periph_id, periph)
                        break
                else:
                    # TODO: logger?
                    print(
                        f"Address 0x{addr_0:08x} does not correspond to any peripheral",
                        file=sys.stderr,
                    )
                    continue

            assert current_periph_regs is not None

            try:
                reg = current_periph_regs[addr_0]
            except KeyError:
                # TODO: logger?
                print(
                    f"Address 0x{addr_0:08x} is within the address range of {current_periph} "
                    f"[0x{current_periph_range.start:x}-0x{current_periph_range.stop:x}), but "
                    "does not correspond to any register in the peripheral",
                    file=sys.stderr,
                )
                continue

            reg_len = reg.bit_width // 8
            reg_content_bytes = [val_0]
            for i in range(reg_len - 1):
                try:
                    _, val_i = next(map_iter)
                    reg_content_bytes.append(val_i)
                except StopIteration:
                    raise ValueError(
                        f"Content for {reg} was only partially specified. "
                        f"Missing value for address 0x{addr_0 + i:08x}"
                    )

            # TODO: don't need to call this more than once
            if not self._enforce_svd_constraints:
                reg.unconstrain()

            reg.content = int.from_bytes(bytes(reg_content_bytes), byteorder="little")

        return self

    def apply_dict(self, config: dict[str, Any]) -> DeviceBuilder:
        """Populate device content from a dictionary representation of the registers and content.

        The dictionary structure should match the structure of the device peripherals and registers.
        Content can be set either at the register or field level.
        Field content can be set either using an enum name or with a numeric value.

        :param content: content dictionary.
        :returns: builder instance.
        """
        affected_periphs = []

        for periph_name, content in config.items():
            peripheral = self._device[periph_name]
            for reg_name, reg_value in content.items():
                self._reg_apply_dict(
                    peripheral[reg_name],
                    reg_value,
                )
            affected_periphs.append(peripheral)

        for periph in affected_periphs:
            self._written_peripherals.setdefault(_get_periph_id(periph), periph)

        return self

    def _reg_apply_dict(
        self,
        reg: svd.Array | svd.Struct | svd.Register | svd.Field,
        value: dict | int | str,
    ) -> None:
        match (reg, value):
            case (svd.Array(), dict()):
                for index_str, rest in value.items():
                    try:
                        index = int(index_str)
                    except ValueError:
                        raise ValueError(
                            f"{index_str} is not a valid index for {reg!r}"
                        )
                    self._reg_apply_dict(reg[index], rest)

            case (svd.Struct() | svd.Register(), dict()):
                for name, rest in value.items():
                    self._reg_apply_dict(reg[name], rest)

            case (svd.Register() | svd.Field(), int()) | (svd.Field(), str()):
                if not self._enforce_svd_constraints:
                    reg.unconstrain()
                reg.content = value

            case _:
                raise ValueError(f"{value} is not a valid value for {reg!r}")

    def _device_periph_map(self) -> list[tuple[range, svd.Peripheral]]:
        if self._cached_device_periph_map:
            return self._cached_device_periph_map

        for periph in self._device.values():
            start_addr, end_addr = periph.address_bounds
            self._cached_device_periph_map.append((range(start_addr, end_addr), periph))

        return self._cached_device_periph_map

    def _periph_reg_map(self, peripheral: svd.Peripheral) -> dict[int, svd.Register]:
        periph_id = _get_periph_id(peripheral)
        if periph_id in self._cached_periph_reg_maps:
            return self._cached_periph_reg_maps[periph_id]

        new_map = {reg.address: reg for reg in peripheral.register_iter(leaf_only=True)}
        self._cached_periph_reg_maps[periph_id] = new_map
        return new_map


def _get_periph_id(peripheral: svd.Peripheral) -> tuple:
    return peripheral.name, peripheral.base_address


def _ranges_overlap_inclusive(
    a_start: int, a_end: int, b_start: int, b_end: int
) -> bool:
    return a_end >= b_start and b_end >= a_start
