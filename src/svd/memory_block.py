# Copyright (c) 2023 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import (
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import numpy.ma as ma
from numpy.typing import ArrayLike
from typing_extensions import Self

from .errors import SvdMemoryError

SIZE_TO_DTYPE: Mapping[int, np.dtype] = {
    1: np.dtype(np.uint8),
    2: np.dtype((np.dtype("<u2"), (np.uint8, 2))),
    4: np.dtype((np.dtype("<u4"), (np.uint8, 4))),
}


def _get_dtype_for_size(item_size: int) -> np.dtype:
    try:
        return SIZE_TO_DTYPE[item_size]
    except KeyError:
        raise SvdMemoryError(f"Unsupported item size: {item_size}")


IdxT = TypeVar("IdxT", int, slice)


class MemoryBlock:
    """
    A contiguous memory region at a given (offset, length).
    """

    class Builder:
        """
        Builder that can be used to construct a MemoryBlock in several steps.
        """

        def __init__(self) -> None:
            self._lazy_base_block: Optional[Callable[[], MemoryBlock]] = None
            self._offset: Optional[int] = None
            self._length: Optional[int] = None
            self._default_content: Optional[int] = None
            self._default_item_size: Optional[int] = None
            self._ops: List[Callable[[MemoryBlock], None]] = []

        def build(self) -> MemoryBlock:
            """
            Build the memory block based on the parameters set.

            :return: The built memory block.
            """
            if self._default_content is None or self._default_item_size is None:
                raise ValueError("Missing ")

            from_block: Optional[MemoryBlock] = (
                self._lazy_base_block() if self._lazy_base_block is not None else None
            )

            block = MemoryBlock(
                default_content=self._default_content,
                length=self._length,
                offset=self._offset,
                from_block=from_block,
            )

            for op in self._ops:
                op(block)

            return block

        def lazy_copy_from(self, lazy_block: Callable[[], MemoryBlock]) -> Self:
            """
            Use a different memory block as the base for this memory block.
            The lazy_block should be a callable that can be called in build() to get the base
            block.

            :param lazy_block: Callable that returns the base block.
            :return: The builder instance.
            """
            self._lazy_base_block = lazy_block
            return self

        def set_extent(self, offset: int, length: int) -> Self:
            """
            Set the offset and length of the memory block.
            This is required unless lazy_copy_from() is used.

            :param offset: Starting offset of the memory block.
            :param length: Length of the memory block, starting at the given offset.
            :return: The builder instance.
            """
            self._offset = offset
            self._length = length
            return self

        def set_default_content(self, default_content: int, item_size: int = 4) -> Self:
            """
            Set the default content (initial value) of the memory block.
            This is required.

            :param default_content: Default value.
            :param item_size: Size in bytes of default_content.
            :return: The builder instance.
            """
            self._default_content = default_content
            self._default_item_size = item_size
            return self

        def fill(self, start: int, end: int, content: int, item_size: int = 4) -> Self:
            """
            Fill the memory block address range [start, end) with a value.

            :param start: Start offset of the range to be filled.
            :param end: Exclusive end offset of the range to be filled.
            :param content: Value to fill with.
            :param item_size: Size in bytes of content.
            :return: The builder instance.
            """
            if start < end:
                self._ops.append(
                    partial(
                        MemoryBlock._fill,
                        start=start,
                        end=end,
                        value=content,
                        item_size=item_size,
                    )
                )
            return self

        def tile(self, start: int, end: int, times: int) -> Self:
            """
            Duplicate the values at the memory block address range [start, end) a number of times.
            The range is duplicated at the *times* positions following the range.

            :param start: Start offset of the range to be duplicated.
            :param end: Exclusive end offset of the range to be duplicated.
            :param times: Number of times to duplicate the range.
            :return: The builder instance.
            """
            if times > 1 and start < end:
                self._ops.append(
                    partial(MemoryBlock._tile, start=start, end=end, times=times)
                )
            return self

    def __init__(
        self,
        default_content: int,
        default_item_size: int = 4,
        offset: Optional[int] = None,
        length: Optional[int] = None,
        from_block: Optional[MemoryBlock] = None,
    ) -> None:
        """
        :param default_content:
        :param offset: Starting offset of the memory block. Required unless from_block is passed.
        :param length: Length in bytes of the memory block. Required unless from_block is passed.
        :param from_block: Memory block to use as the base for this memory block.
        """
        self._offset: int
        self._length: int

        default_dtype = SIZE_TO_DTYPE[default_item_size]

        if from_block is not None:
            # Shrink the offset and expand the length so that the base block is covered.
            if offset is not None:
                self._offset = min(offset, from_block._offset)
            else:
                self._offset = from_block._offset

            if length is not None:
                self._length = (
                    max(self._offset + length, from_block._offset + from_block._length)
                    - self._offset
                )
            else:
                self._length = from_block._length
        else:
            if offset is None or length is None:
                raise ValueError(
                    "offset and length are required when no from_block is given"
                )

            self._offset = offset
            self._length = length

        # Define the arrays used to represent the memory at [offset...offset + length]
        # To save some space, the offset is added on/subtracted in the API functions, so that the
        # array sizes are independent of the offset.
        # The address space defined in the SVD file may have discontinuities where there are no
        # elements. This is represented in the memory block by an address mask that masks out all
        # the unoccupied addresses.
        data = _numpy_full(
            self._length // default_item_size, default_content, dtype=default_dtype
        ).view(np.uint8)
        address_mask = np.ones_like(data, dtype=bool)
        self._array: ma.MaskedArray = ma.MaskedArray(
            data=data, mask=address_mask, dtype=np.uint8
        )
        self._written = np.zeros_like(data, dtype=np.uint8)

        if from_block is not None:
            # Copy the source block arrays into this block.
            dst_start = from_block._offset - offset if offset is not None else 0
            dst_end = dst_start + from_block._length
            self._array.mask[dst_start:dst_end] &= from_block._array.mask
            np.copyto(dst=self._array[dst_start:dst_end], src=from_block._array)
            np.copyto(dst=self._written[dst_start:dst_end], src=from_block._written)

    def is_written(self, idx: IdxT) -> bool:
        """:return: True if the given address has been explicitly written to, False otherwise."""
        translated_idx, dtype = self._translate_access(idx, item_size=1)
        return bool(self._written.view(dtype=dtype)[translated_idx].any())

    @overload
    def at(self, offset: int, item_size: int = 4) -> int:
        ...

    @overload
    def at(self, offset: slice, item_size: int = 4) -> ArrayLike:
        ...

    def at(
        self, offset: IdxT, item_size: int = 4
    ) -> Union[int, ArrayLike]:
        """
        Get the memory value at a given address offset.

        :param offset: Address offset or a slice of address offsets.
        :param item_size: Size of the value(s) to get.
        :return: The value(s) stored at the given offset(s).
        """
        translated_idx, dtype = self._translate_access(offset, item_size)
        return self._array.data.view(dtype=dtype)[translated_idx]

    def set_at(
        self,
        offset: IdxT,
        value: Union[int, ArrayLike],
        item_size: int = 4,
    ) -> None:
        """
        Set the memory value at a given index.

        :param offset: Address offset or a slice of address offsets.
        :param value: Value to write to the offset(s).
        :param item_size: Size of the value(s) to get.
        """
        translated_idx, dtype = self._translate_access(offset, item_size)
        self._array.data.view(dtype=dtype)[translated_idx] = value
        self._written.view(dtype=dtype)[translated_idx] = _one_bits(item_size)

    @overload
    def __getitem__(self, offset: int, /) -> int:
        ...

    @overload
    def __getitem__(self, offset: slice, /) -> ArrayLike:
        ...

    def __getitem__(self, offset: Union[int, slice], /) -> Union[int, ArrayLike]:
        """
        Get the memory value at a given index.
        Assumes a 4-byte item size; see MemoryBlock.set_at() to use a different item size.
        """
        return self.at(offset)

    def __setitem__(self, offset: IdxT, value: Union[int, ArrayLike], /) -> None:
        """
        Set the memory value at a given index.
        Assumes a 4-byte item size; see MemoryBlock.set_at() to use a different item size.
        """
        self.set_at(offset, value)

    def memory_iter(
        self,
        item_size: int = 4,
        with_offset: int = 0,
        written_only: bool = False,
    ) -> Iterator[Tuple[int, int]]:
        """
        Iterator over memory addresses and values contained in the block,
        in ascending address order.

        :param item_size: Size in bytes of each memory element.
        :param with_offset: Base address offset to use when iterating.
        :param native_byteorder: Cast from device byte order to native byte order.
        :param written_only: Yield only the adresses/values that have been explictly written to.
        :return: Iterator over (address, value) pairs describing the memory in the block.
        """
        if self._length % item_size != 0:
            raise ValueError(
                f"Memory block length {self._length} is not aligned to the item size "
                f"of {item_size}"
            )

        dtype = SIZE_TO_DTYPE[item_size]

        # Mask of locations that should be included in the result
        address_filter = ~self._array.mask
        if written_only:
            address_filter = np.logical_and(
                address_filter, self._written.view(dtype=bool)
            )

        # Array of addresses to return.
        # The internal array represents the memory at [offset...offset + length],
        # so we start the address range at offset plus the additional offset argument.
        # Note that we first apply the filter (which is at 1B granularity) and then cast to
        # the desired item size.
        address_start = self._offset + with_offset
        addresses = np.linspace(
            address_start,
            address_start + self._length,
            num=self._length,
            endpoint=False,
            dtype=int,
        )[address_filter][::item_size]

        # Array of values to return. These are also first filtered at a 1B granularity and the
        # cast to the desired item size
        values = self._array.data[address_filter].view(dtype)

        for address, value in zip(addresses, values):
            yield int(address), int(value)

    def __len__(self) -> int:
        """
        Length of the memory block.
        This may be different from the length provided in the constructor if a larger from_block
        was also given.
        """
        return len(self._array)

    def _translate_access(self, offset: IdxT, item_size: int) -> Tuple[IdxT, np.dtype]:
        """
        Translate index and size arguments given in the public API to indices in the internal
        numpy arrays.
        """
        translated_idx: IdxT
        dtype: np.dtype = SIZE_TO_DTYPE[item_size]

        if isinstance(offset, int):
            if offset % item_size != 0:
                raise ValueError(
                    f"Offset {offset} is not aligned to the item size of {item_size}"
                )

            translated_idx = (offset - self._offset) // item_size

        elif isinstance(offset, slice):
            if any(
                v is not None and v & item_size != 0
                for v in (offset.start, offset.stop, offset.step)
            ):
                raise ValueError(
                    f"Offsets {offset} are not aligned to the item size of {item_size}"
                )

            translated_idx = slice(
                (offset.start - self._offset) // item_size
                if offset.start is not None
                else None,
                (offset.stop - self._offset) // item_size
                if offset.stop is not None
                else None,
                (offset.step // item_size) if offset.step else None,
            )

        else:
            raise ValueError(f"Unsupported index: {offset}")

        return translated_idx, dtype

    def _tile(self, /, *, start: int, end: int, times: int) -> None:
        """Inline tile operation. See MemoryBlock.Builder.tile() for details"""
        length = end - start
        src_offset_start = start - self._offset
        src_offset_end = end - self._offset
        dst_offset_start = src_offset_start + length
        dst_offset_end = src_offset_start + length * times

        for array in (self._array.data, self._array.mask):
            array_src = array[src_offset_start:src_offset_end]
            array_dst = array[dst_offset_start:dst_offset_end].reshape(
                (times - 1, length)
            )
            array_dst[:] = array_src

    def _fill(self, /, *, start: int, end: int, value: int, item_size: int) -> None:
        """Inline fill operation. See MemoryBlock.Builder.fill() for details"""
        dtype = SIZE_TO_DTYPE[item_size]
        offset_start = start - self._offset
        offset_end = end - self._offset
        self._array.mask[offset_start:offset_end] = False
        data_dst = self._array.data[offset_start:offset_end].view(dtype)
        data_dst[:] = value


def _numpy_full(length: int, value: int, dtype: np.dtype) -> np.ndarray:
    """
    Replacement for np.full() with special handling for the zero case.
    numpy doesn't seem to handle numpy.full(..., fill_value=0) in any special way.
    Using np.zeros() for that case here provides a significant speedup for large arrays.
    """
    if value == 0:
        return np.zeros(length, dtype=dtype)
    else:
        return np.full(
            length,
            value,
            dtype=dtype,
        )


def _one_bits(item_size: int) -> int:
    """:return: bitfield of all ones with the given item size."""
    return 2 ** (8 * item_size) - 1
