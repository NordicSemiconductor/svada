# Copyright (c) 2024 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from typing import Any, Optional

import svd


@dataclass
class BuildSelector:
    """Used to select select which parts of the device to write to an output."""

    class ContentStatus(str, enum.Enum):
        WRITTEN = "written"
        MODIFIED = "modified"

    peripherals: Optional[list[str]] = None
    address_range: Optional[tuple[int, int]] = None
    content_status: Optional[ContentStatus] = None

    def is_periph_selected(self, periph: svd.Peripheral) -> bool:
        if self.peripherals and periph.name not in self.peripherals:
            return False

        address_bounds = periph.address_bounds
        if self.address_range is not None and not _ranges_overlap_inclusive(
            address_bounds[0], address_bounds[1] - 1, *self.address_range
        ):
            return False

        return True

    def is_addr_selected(self, address: int) -> bool:
        return self.address_range is None or (
            self.address_range[0] <= address <= self.address_range[1]
        )

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


class ContentBuilder:
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

    def build_bytes(
        self, selector: BuildSelector = BuildSelector(), *, fill_value: int = 0
    ) -> bytearray:
        """Encode device content as bytes.

        :param selector: selected parts of the device.
        :param fill_value: value used to fill empty address ranges.
        :returns: content bytes.
        """
        memory = self.build_memory(selector)
        if not memory:
            return bytearray()

        start_addr = next(iter(memory))
        end_addr = next(reversed(memory))
        out = bytearray([fill_value]) * (end_addr - start_addr + 1)

        for addr, value in memory.items():
            out[addr - start_addr] = value

        return out

    def build_memory(self, selector: BuildSelector = BuildSelector()) -> dict[int, int]:
        """Encode device content as a mapping between address and value.

        :param selector: selected parts of the device.
        :returns: content memory map.
        """
        memory = {}

        self._log_selector_issues(selector)

        for peripheral in self._device.values():
            if (
                selector.content_status == BuildSelector.ContentStatus.WRITTEN
                and peripheral not in self.written_peripherals
            ):
                continue

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
                if not selector.is_addr_selected(addr):
                    continue
                memory[addr] = val

        if not memory:
            svd.log.warning("No part of the device was selected")

        return memory

    def build_dict(self, selector: BuildSelector = BuildSelector()) -> dict:
        """Encode device content as a dictionary representation of the registers and content.

        :param selector: selected parts of the device.
        :returns: content dictionary.
        """
        config = {}

        self._log_selector_issues(selector)

        for peripheral in self._device.values():
            if not selector.is_periph_selected(peripheral):
                continue

            cfg_periph = config.setdefault(peripheral.name, {})

            if (
                selector.content_status == BuildSelector.ContentStatus.WRITTEN
                and peripheral not in self.written_peripherals
            ):
                continue

            for reg in peripheral.register_iter(leaf_only=True):
                assert isinstance(reg, svd.Register)
                if not selector.is_reg_selected(reg):
                    continue

                reg_table = cfg_periph
                for part in reg.path.parts:
                    reg_table = reg_table.setdefault(str(part), {})

                for field_name, field in reg.fields.items():
                    if not selector.is_field_selected(field):
                        continue

                    try:
                        reg_table[field_name] = field.content_enum
                    except LookupError:
                        # Content does not match any defined enum
                        reg_table[field_name] = field.content

        if not config:
            svd.log.warning("No part of the device was selected")

        return config

    def apply_memory(self, content: dict[int, int]) -> ContentBuilder:
        """Update device content based on a memory map.

        The content is assumed to be at byte granularity and sorted by ascending address.

        :param content: content memory map.
        :returns: builder instance.
        """
        reg_map = {}
        current_periph = None
        current_periph_range = range(-1, 0)
        current_periph_regs = {}

        map_iter = iter(content.items())

        while True:
            try:
                # The first address, value in a potentially a multi-byte value
                addr_0, val_0 = next(map_iter)
            except StopIteration:
                break

            if addr_0 not in current_periph_range:
                for periph_range, periph in self._device_periph_map:
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
                    svd.log.warning(
                        f"Address 0x{addr_0:08x} does not correspond to any peripheral"
                    )
                    continue

            assert current_periph_regs is not None

            try:
                reg = current_periph_regs[addr_0]
            except KeyError:
                svd.log.warning(
                    f"Address 0x{addr_0:08x} is within the address range of {current_periph} "
                    f"[0x{current_periph_range.start:x}-0x{current_periph_range.stop:x}), but "
                    "does not correspond to any register in the peripheral"
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

    def apply_dict(self, config: dict[str, Any]) -> ContentBuilder:
        """Populate device content from a dictionary representation of the registers and content.

        The dictionary structure should match the structure of the device peripherals and registers.
        Content can be set either at the register or field level.
        Field content can be set either using an enum name or with a numeric value.

        :param content: content dictionary.
        :returns: builder instance.
        """
        for periph_name, content in config.items():
            peripheral = self._device[periph_name]
            for reg_name, reg_value in content.items():
                self._reg_apply_dict(
                    peripheral[reg_name],
                    reg_value,
                )
            self._written_peripherals.setdefault(_get_periph_id(peripheral), peripheral)

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

    @cached_property
    def _device_periph_map(self) -> list[tuple[range, svd.Peripheral]]:
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

    def _log_selector_issues(self, selector: BuildSelector) -> None:
        if selector.peripherals is not None:
            nonexistent_periphs = set(selector.peripherals) - set(self.device.keys())
            if nonexistent_periphs:
                svd.log.warning(
                    "Selector references peripherals that don't exist in the device: "
                    + ", ".join(nonexistent_periphs)
                )


def _get_periph_id(peripheral: svd.Peripheral) -> tuple:
    """Get a hashable unique ID for a peripheral."""
    return peripheral.name, peripheral.base_address


def _ranges_overlap_inclusive(
    a_start: int, a_end: int, b_start: int, b_end: int
) -> bool:
    return a_end >= b_start and b_end >= a_start
