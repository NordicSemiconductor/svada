# Copyright (c) 2023 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

"""
High level representation of a SVD device.
This representation does not aim to expose all the information contained in the SVD file,
but instead focuses on certain key features of the device description.

Since processing all the information contained in the SVD file can be computationally expensive,
many of the operations in this module lazily compute the data needed on first access.
"""

from __future__ import annotations

import dataclasses as dc
import typing
from abc import ABC, abstractmethod
from functools import cached_property
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Reversible,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import TypeGuard

from . import bindings
from ._device import (
    ChildIter,
    LazyFixedMapping,
    iter_merged,
    remove_registers,
    strip_suffix,
    svd_element_repr,
    topo_sort_derived_peripherals,
)
from .bindings import (
    Access,
    Cpu,
    Dimensions,
    FullRegisterProperties,
    ReadAction,
    RegisterProperties,
    WriteAction,
)
from .errors import (
    SvdDefinitionError,
    SvdIndexError,
    SvdKeyError,
    SvdMemoryError,
    SvdPathError,
)
from .memory_block import MemoryBlock
from .path import EPath, EPathType, FEPath
from .profiling import TimingReport, create_object_report, timed_method

if TYPE_CHECKING:
    from .parsing import Options


# Union of regular register types
RegisterUnion = Union["Array", "Register", "Struct"]

# Generic type variable contrained to being a regular register type
RegisterType = TypeVar("RegisterType", "Array", "Register", "Struct")

# Generic type variable of an array element type
ArrayMemberT = TypeVar("ArrayMemberT", "Register", "Struct")

# Union of flat register types
FlatRegisterUnion = Union["FlatRegister", "FlatStruct"]

# Generic type variable constrained to being a flat register type
FlatRegisterType = TypeVar("FlatRegisterType", "FlatRegister", "FlatStruct")

# Type variable constrained to either a regular or a flat register (but not a mix of regular/flat)
RegisterKindType = TypeVar("RegisterKindType", RegisterUnion, FlatRegisterUnion)

# Type variable constrained to either a regular or flat field
FieldKindType = TypeVar("FieldKindType", "Field", "FlatField")


class Device(Mapping[str, "Peripheral"]):
    """Representation of a SVD device."""

    @timed_method(max_times=1)
    def __init__(
        self,
        device: bindings.DeviceElement,
        options: Options,
        **kwargs: Any,
    ):
        self._device: bindings.DeviceElement = device
        self._options: Options = options
        self._reg_props: RegisterProperties = self._device.register_properties

        if self._device.address_unit_bits != 8:
            raise NotImplementedError(
                "the implementation assumes a byte-addressable device"
            )

        peripherals_unsorted: Dict[str, Peripheral] = {}

        # Initialize peripherals in topological order to ensure that base peripherals are
        # initialized before derived peripherals.
        for peripheral_element in topo_sort_derived_peripherals(device.peripherals):
            if peripheral_element.derived_from is not None:
                base_peripheral = peripherals_unsorted[peripheral_element.derived_from]
            else:
                base_peripheral = None

            if options is not None and options.skip_registers:
                remove_registers(peripheral_element, options.skip_registers)

            peripheral = Peripheral(
                peripheral_element,
                device=self,
                options=self._options,
                parent_reg_props=self._reg_props,
                base_peripheral=base_peripheral,
            )

            peripherals_unsorted[peripheral.name] = peripheral

        self._peripherals: Dict[str, Peripheral] = dict(
            sorted(peripherals_unsorted.items(), key=lambda kv: kv[1].base_address)
        )

        # Store the time spent parsing prior to calling this constructor
        self._time_parse = kwargs.get("time_parse", 0)

    @property
    def name(self) -> str:
        """Name of the device."""
        return self._device.name

    @property
    def series(self) -> Optional[str]:
        """Device series name."""
        return self._device.series

    @property
    def vendor_id(self) -> Optional[str]:
        """Device vendor ID."""
        return self._device.vendor_id

    @property
    def qualified_name(self) -> str:
        """Name of the device, including vendor and series."""
        return f"{self.vendor_id or ''}::{self.series or ''}::{self.name}"

    @property
    def cpu(self) -> Optional[Cpu]:
        """Device CPU information"""
        return self._device.cpu

    @property
    def address_unit_bits(self) -> int:
        """Number of data bits corresponding to an address."""
        return self._device.address_unit_bits

    @property
    def bus_bit_width(self) -> int:
        """Maximum data bits supported by the data bus in a single transfer."""
        return self._device.width

    @property
    def peripherals(self) -> Mapping[str, Peripheral]:
        """
        Map of peripherals in the device, indexed by name.
        The peripherals are sorted by ascending base address.
        """
        return MappingProxyType(self._peripherals)

    @property
    def _profiling_info(self) -> ProfilingInfo:
        """Profiling info for the device and peripherals."""
        return {
            "parse": float(self._time_parse),
            "device": create_object_report(self),
            "peripherals": {
                name: create_object_report(peripheral)
                for name, peripheral in self.items()
            },
        }

    def __getitem__(self, name: str) -> Peripheral:
        """
        :param name: Peripheral name.
        :raises SvdPathError: if the peripheral was not found.
        :return: Peripheral with the given name.
        """
        try:
            return self.peripherals[name]
        except LookupError as e:
            raise SvdKeyError(name, self, "peripheral not found") from e

    def __iter__(self) -> Iterator[str]:
        """:return: Iterator over the names of peripherals in the device."""
        return iter(self.peripherals)

    def __len__(self) -> int:
        """:return: Number of peripherals in the device."""
        return len(self.peripherals)

    def __repr__(self) -> str:
        """Short description of the device."""
        return svd_element_repr(self.__class__, self.qualified_name, length=len(self))


class ProfilingInfo(TypedDict):
    """Type of profiling information reported by the device."""

    parse: float
    device: TimingReport
    peripherals: Mapping[str, TimingReport]


class Peripheral(Mapping[str, RegisterUnion]):
    """
    Representation of a specific device peripheral.

    Internally, this class maintains a representation of a peripheral that is always guaranteed to
    be correct when compared to the allowable values prescribed by the SVD file the class was
    instantiated from. This representation starts off by having the default values defined within
    the SVD.
    """

    def __init__(
        self,
        element: bindings.PeripheralElement,
        device: Optional[Device],
        options: Options,
        parent_reg_props: bindings.RegisterProperties,
        base_peripheral: Optional[Peripheral] = None,
        new_base_address: Optional[int] = None,
    ):
        """
        :param element: SVD peripheral element.
        :param device: Parent Device object (may be None if this is a copy).
        :param base_reg_props: Register properties of the parent device.
        :param base_peripheral: Base peripheral that this peripheral is derived from, if any.
        :param new_base_address: Overridden base address of this peripheral, if any.
        """

        self._peripheral: bindings.PeripheralElement = element
        self._device: Optional[Device] = device
        self._options: Options = options
        self._base_peripheral: Optional[Peripheral] = base_peripheral
        self._base_address: int = (
            element.base_address if new_base_address is None else new_base_address
        )
        self._reg_props: bindings.RegisterProperties = (
            self._peripheral.get_register_properties(base_props=parent_reg_props)
        )

        # These dicts store every register associated with the peripheral.
        self._flat_registers: Dict[FEPath, FlatRegisterUnion] = {}
        self._dim_registers: Dict[EPath, RegisterUnion] = {}

    def copy_to(self, new_base_address: int) -> Peripheral:
        """
        Copy the peripheral to a new base address.

        :param new_base_address: Base address of the new peripheral.
        :returns: A copy of this peripheral at the new base address.
        """
        return Peripheral(
            element=self._peripheral,
            device=None,
            options=self._options,
            parent_reg_props=self._reg_props,
            base_peripheral=self,
            new_base_address=new_base_address,
        )

    @property
    def name(self) -> str:
        """Name of the peripheral."""
        return self._peripheral.name

    @property
    def base_address(self) -> int:
        """Base address of the peripheral."""
        return self._base_address

    @property
    def address_bounds(self) -> Tuple[int, int]:
        """Minimum and maximum address occupied by the registers in the peripheral."""
        bounds = self._register_info.address_bounds
        if bounds is None:
            raise SvdMemoryError(f"{self!s} has no registers")
        min_offset, max_offset = bounds
        return (self.base_address + min_offset, self.base_address + max_offset)

    @property
    def interrupts(self) -> Mapping[str, int]:
        """Interrupts associated with the peripheral. Mapping from interrupt name to value."""
        return {
            interrupt.name: interrupt.value for interrupt in self._peripheral.interrupts
        }

    @cached_property
    def registers(self) -> Mapping[str, RegisterUnion]:
        """Mapping of top-level registers in the peripheral, indexed by name."""
        return LazyFixedMapping(keys=self._specs.keys(), factory=self.__getitem__)

    @overload
    def register_iter(self, leaf_only: Literal[True]) -> Iterator[Register]:
        ...

    @overload
    def register_iter(self, leaf_only: bool = False) -> Iterator[RegisterUnion]:
        ...

    @timed_method(max_times=10)
    def register_iter(self, leaf_only: Any = False) -> Any:
        """
        Iterator over the registers in the peripheral in pre-order.
        Registers are ordered by increasing offset/address.

        :param flat: Do not yield individual registers in register arrays.
        :param leaf_only: Only yield registers at the bottom of the register tree.
        :return: Iterator over the registers in the peripheral.
        """
        return self._register_iter(EPath, leaf_only)

    @cached_property
    def flat_registers(self) -> Mapping[str, FlatRegisterUnion]:
        """
        Mapping of top-level flat registers in the peripheral, indexed by name.
        The
        """
        return LazyFixedMapping(
            keys=self._specs.keys(),
            factory=lambda n: self._get_or_create_register(FEPath(n)),
        )

    @overload
    def flat_register_iter(self, leaf_only: Literal[True]) -> Iterator[FlatRegister]:
        ...

    @overload
    def flat_register_iter(
        self, leaf_only: Literal[False]
    ) -> Iterator[FlatRegisterUnion]:
        ...

    @timed_method(max_times=10)
    def flat_register_iter(
        self, leaf_only: bool = False
    ) -> Iterator[FlatRegisterUnion]:
        """
        Iterator over the registers in the peripheral in pre-order.
        Registers are ordered by increasing offset/address.

        :param flat: Do not yield individual registers in register arrays.
        :param leaf_only: Only yield registers at the bottom of the register tree.
        :return: Iterator over the registers in the peripheral.
        """
        return self._register_iter(FEPath, leaf_only)

    def _register_iter(self, path_cls: Type, leaf_only: bool = False) -> Iterator:
        """Commmon register iteration implementation."""
        stack = [
            self._get_or_create_register(path_cls(name))
            for name in reversed(self._specs.keys())
        ]

        while stack:
            register = stack.pop()

            if register.leaf or not leaf_only:
                yield register

            if not register.leaf:
                stack.extend(reversed(register.child_iter()))

    @timed_method(max_times=10)
    def memory_iter(
        self,
        item_size: int = 1,
        absolute_addresses: bool = False,
        written_only: bool = False,
    ) -> Iterator[Tuple[int, int]]:
        """
        Get an iterator over the peripheral register contents.

        :param item_size: Byte granularity of the iterator.
        :param absolute_addresses: If True, use absolute instead of peripheral relative addresses.
        :param written_only: If true, only include those addresses that have been explicitly
        written to.
        :return: Iterator over the peripheral register contents.
        """
        address_offset = self.base_address if absolute_addresses else 0
        yield from self._memory_block.memory_iter(
            item_size, with_offset=address_offset, written_only=written_only
        )

    def __getitem__(self, path: Union[str, Sequence[Union[str, int]]]) -> RegisterUnion:
        """
        :param path: Name or path of the register.
        :return: Register instance.
        """
        return self._get_or_create_register(EPath(path))

    def __setitem__(self, name: str, content: int) -> None:
        """
        :param name: Name of the register to update.
        :param content: The raw register value to write to the specified register.
        """
        _register_set_content(self, name, content)

    def __iter__(self) -> Iterator[str]:
        """:return: Iterator over the names of top-level registers in the peripheral."""
        return iter(self._specs)

    def __len__(self) -> int:
        """:return: Number of top-level registers in the peripheral."""
        return len(self._specs)

    @overload
    def _get_or_create_register(self, path: EPath) -> RegisterUnion:
        ...

    @overload
    def _get_or_create_register(self, path: FEPath) -> FlatRegisterUnion:
        ...

    def _get_or_create_register(self, path: Any) -> Any:
        """
        Common method for accessing a register contained in the peripheral.
        If the register is accessed for the first time, it is first initialized.
        Otherwise, a cached register is returned.
        Note that if the requested register is below the top level, all the registers that are
        ancestors of the register are also initialized if needed.

        :param path: Path to the register.
        :return: The register instance at the given path.
        """
        if isinstance(path, EPath):
            return self._do_get_or_create_register(self._dim_registers, path)
        elif isinstance(path, FEPath):
            return self._do_get_or_create_register(self._flat_registers, path)
        else:
            raise ValueError(f"unrecognized path: {path}")

    @overload
    def _do_get_or_create_register(
        self, storage: Dict[EPath, RegisterUnion], path: EPath
    ) -> RegisterUnion:
        ...

    @overload
    def _do_get_or_create_register(
        self, storage: Dict[FEPath, FlatRegisterUnion], path: FEPath
    ) -> FlatRegisterUnion:
        ...

    def _do_get_or_create_register(self, storage: Dict, path: Any) -> Any:
        try:
            register = storage[path]
        except KeyError:
            ancestor_path = path.parent
            if ancestor_path is not None:
                # Build all the registers from the first available ancestor to the requested
                # register.
                while (
                    cached_register := storage.get(ancestor_path, None)
                ) is None and (parent := ancestor_path.parent) is not None:
                    ancestor_path = parent

                if cached_register is None:
                    register = self._create_register(ancestor_path)
                else:
                    register = cached_register

                for i in range(len(ancestor_path), len(path)):
                    register = self._create_register(path[: i + 1], register)
            else:
                register = self._create_register(path)

            storage[path] = register

        return register

    @overload
    def _create_register(
        self, path: EPath, parent: Optional[RegisterUnion] = None
    ) -> RegisterUnion:
        ...

    @overload
    def _create_register(
        self, path: FEPath, parent: Optional[FlatRegisterUnion] = None
    ) -> FlatRegisterUnion:
        ...

    def _create_register(
        self,
        path: Any,
        parent: Any = None,
    ) -> Any:
        if isinstance(path, EPath):
            return self._create_regular_register(path, parent)
        elif isinstance(path, FEPath):
            return self._create_flat_register(path, parent)
        else:
            raise ValueError(f"unrecognized path: {path}")

    def _create_regular_register(
        self, path: EPath, parent: Optional[RegisterUnion] = None
    ) -> RegisterUnion:
        if parent is None:
            try:
                spec = self._specs[path.stem]
            except KeyError:
                raise SvdPathError(path, self)

            instance_offset = 0

        elif isinstance(parent, Array):
            array_spec = parent._spec
            index = path.element_index
            if index is None:
                raise SvdIndexError(path, parent, "expected an array index path")

            try:
                array_offset = array_spec.dimensions.to_range()[index]
            except IndexError:
                raise SvdIndexError(
                    path,
                    parent,
                    f"index {index} is out of range for array with length "
                    f"{array_spec.dimensions.length}",
                )

            instance_offset = parent._instance_offset + array_offset
            spec = typing.cast(_RegisterSpec, parent._spec)

        elif isinstance(parent, Struct):
            try:
                spec = parent._spec.registers[path.stem]
            except KeyError:
                raise SvdKeyError(path.stem, parent)

            instance_offset = parent._instance_offset

        else:
            raise ValueError(f"Invalid parent register: {parent}")

        if is_array(spec) and path.element_index is None:
            return Array(
                spec=spec,
                peripheral=self,
                path=path,
                instance_offset=instance_offset,
            )
        elif is_struct(spec):
            return Struct(
                spec=spec,
                peripheral=self,
                path=path,
                instance_offset=instance_offset,
            )
        elif is_register(spec):
            return Register(
                spec=spec,
                peripheral=self,
                path=path,
                instance_offset=instance_offset,
            )
        else:
            raise ValueError(f"Element contains neither registers nor fields: {spec}")

    def _create_flat_register(
        self, path: FEPath, parent: Optional[FlatRegisterUnion] = None
    ) -> FlatRegisterUnion:
        """Create a flat register instance."""
        if parent is None:
            try:
                spec = self._specs[path.stem]
            except KeyError as e:
                raise SvdKeyError(path.stem, self) from e

            instance_offset = 0

        elif isinstance(parent, FlatStruct):
            try:
                spec = parent._spec.registers[path.stem]
            except KeyError as e:
                raise SvdKeyError(path.stem, parent) from e

            instance_offset = parent._instance_offset
        else:
            raise ValueError(f"Invalid parent register: {parent}")

        if is_struct(spec):
            return FlatStruct(
                spec=spec,
                peripheral=self,
                path=path,
                instance_offset=instance_offset,
            )
        elif is_register(spec):
            return FlatRegister(
                spec=spec,
                peripheral=self,
                path=path,
                instance_offset=instance_offset,
            )
        else:
            raise ValueError(f"Element contains neither registers nor fields: {spec}")

    @cached_property
    @timed_method(max_times=1)
    def _memory_block(self) -> MemoryBlock:
        """
        The memory block describing the values in the peripheral.
        This is computed the first time the property is accessed and cached for later.
        Initially this contains the reset values described in the SVD file.

        Note that accessing this property in a derived peripheral may also cause the memory blocks
        of base peripherals to be computed.
        """
        return self._register_info.memory_builder.build()

    @property
    def _specs(self) -> Mapping[str, _RegisterSpec]:
        """Mapping of register descriptions in the peripheral, indexed by name."""
        return self._register_info.register_specs

    @cached_property
    @timed_method(max_times=1)
    def _register_info(self) -> _ExtractedRegisterInfo:
        """
        Compute the descriptions of the registers contained in the peripheral, taking into
        account registers derived from the base peripheral, if any.
        """
        base_descriptions: Optional[Mapping[str, _RegisterSpec]] = None
        base_memory: Optional[Callable[[], MemoryBlock]] = None
        base_address_bounds: Optional[Tuple[int, int]] = None

        if self._base_peripheral is not None:
            base_peripheral: Peripheral = self._base_peripheral

            # If the register properties are equal, then it is possible to reuse all the immutable
            # properties from the base peripheral.
            if base_peripheral._reg_props == self._reg_props:
                base_descriptions = base_peripheral._specs
                base_memory = lambda: base_peripheral._memory_block
            # Otherwise, traverse the base registers again, because the difference in
            # register properties propagates down to the register elements.
            else:
                base_info = _extract_register_info(
                    base_peripheral._peripheral.registers,
                    options=self._options,
                    base_reg_props=self._reg_props,
                )
                base_descriptions = base_info.register_specs
                base_memory = lambda: base_info.memory_builder.build()

            base_address_bounds = base_peripheral._register_info.address_bounds

        try:
            info = _extract_register_info(
                self._peripheral.registers,
                options=self._options,
                base_reg_props=self._reg_props,
                base_specs=base_descriptions,
                base_memory=base_memory,
                base_address_bounds=base_address_bounds,
            )
        except _MissingDefaultResetValueError:
            raise SvdDefinitionError(
                [self._peripheral], "Missing default reset value for the peripheral."
            )

        return info

    def __repr__(self) -> str:
        """Short peripheral description."""
        return svd_element_repr(self.__class__, self.name, address=self.base_address)


def _register_set_content(container: Any, path: Union[int, str], content: int) -> None:
    """Common function for setting the content of a register."""
    try:
        container[path].content = content
    except AttributeError as e:
        raise SvdMemoryError(f"{container[path]!s} does not have content") from e


class _RegisterSpec(NamedTuple):
    """
    Immutable description of a SVD register/cluster element.
    This is separated from the register classes to optimize construction of derived peripherals.
    Since the description is not tied to a specific Peripheral instance, it can be
    instantiated once per register/cluster in the SVD document and reused for derived peripherals,
    as long as inherited properties are the same for the base and derived peripheral.
    """

    # Register name
    name: str
    # Lowest address offset contained within this element and any descendants
    offset_start: int
    # Highest address offset contained within this element and any descendants
    offset_end: int
    # Effective register properties, either inherited or specified on the element itself
    reg_props: bindings.RegisterProperties
    # Register dimensions
    dimensions: Optional[bindings.Dimensions]
    # Direct child registers of this element
    registers: Optional[Mapping[str, _RegisterSpec]]
    # Child fields of this element
    fields: Optional[Mapping[str, _FieldSpec]]
    # The SVD element itself
    element: Union[bindings.RegisterElement, bindings.ClusterElement]


class _CommonP(Protocol):
    """Protocol describing fields common to all register specs."""

    name: str
    offset_start: int
    offset_end: int
    reg_props: Union[RegisterProperties, FullRegisterProperties]
    dimensions: Optional[bindings.Dimensions]
    element: Union[bindings.RegisterElement, bindings.ClusterElement]


class _StructP(_CommonP):
    """
    Protocol describing fields specific to specs describing a struct.
    Used for type checking to avoid None and instance checks.
    """

    registers: Mapping[str, _RegisterSpec]
    reg_props: RegisterProperties
    element: bindings.ClusterElement


def is_struct(spec: _RegisterSpec) -> TypeGuard[_StructP]:
    """True if the SVD element describes a structure, i.e. contains other registers"""
    return spec.registers is not None


class _RegisterP(_CommonP):
    """
    Protocol describing fields specific to specs describing a register.
    Used for type checking to avoid None and instance checks.
    """

    reg_props: FullRegisterProperties
    fields: Mapping[str, _FieldSpec]
    element: bindings.RegisterElement


def is_register(spec: _RegisterSpec) -> TypeGuard[_RegisterP]:
    """True if the SVD element describes a register"""
    return spec.fields is not None


class _ArrayP(_CommonP):
    """
    Protocol describing fields specific to specs describing an array.
    Used for type checking to avoid None and instance checks.
    """

    dimensions: bindings.Dimensions


def is_array(spec: _RegisterSpec) -> TypeGuard[_ArrayP]:
    """True if the SVD element describes an array, i.e. has dimensions."""
    return spec.dimensions is not None


# Generic type variable describing a register spec variant.
# Register subclasses use one of the specific protocols this is constrained to for type checking.
_SpecT = TypeVar("_SpecT", _StructP, _RegisterP, _ArrayP)


class _RegisterNode(ABC, Generic[_SpecT, EPathType]):
    """Base class for all register level types"""

    __slots__ = [
        "_spec",
        "_peripheral",
        "_path",
        "_instance_offset",
    ]

    def __init__(
        self,
        spec: _SpecT,
        peripheral: Peripheral,
        path: EPathType,
        instance_offset: int = 0,
    ):
        """
        :param spec: Register description.
        :param peripheral: Parent peripheral.
        :param path: Path of the register.
        :param instance_offset: Address offset inherited from the parent register.
        """
        self._spec: _SpecT = spec
        self._peripheral: Peripheral = peripheral
        self._path: EPathType = path
        self._instance_offset: int = instance_offset

    @property
    def name(self) -> str:
        """Name of the register."""
        return self.path.name

    @property
    def path(self) -> EPathType:
        """Full path to the register."""
        return self._path

    @property
    def address(self) -> int:
        """Absolute address of the peripheral in memory"""
        return self._peripheral.base_address + self.offset

    @property
    def offset(self) -> int:
        """Address offset of the register, relative to the peripheral it is contained in"""
        return self._spec.offset_start + self._instance_offset

    @property
    @abstractmethod
    def leaf(self) -> bool:
        ...


class Array(_RegisterNode[_ArrayP, EPath], Sequence[ArrayMemberT]):
    """
    Container of regular Structs or Registers.

    In the hierarchy of regular register level types, an element with dimensions is
    expanded to an Array containing one object for each dimension index.
    The following SVD elements map to an Array object:
    * cluster element with dimensions -> Array[Struct]
    * register element with dimensions -> Array[Register]

    Arrays support the Sequence protocol, and can be used similarly to lists.
    """

    @overload
    def __getitem__(self, path: int, /) -> ArrayMemberT:
        ...

    @overload
    def __getitem__(self, path: slice, /) -> Sequence[ArrayMemberT]:
        ...

    @overload
    def __getitem__(self, path: Sequence[Union[int, str]], /) -> ArrayMemberT:
        ...

    def __getitem__(self, path: Any, /) -> Any:
        """Get the element at the given path."""
        if isinstance(path, slice):
            return [self[i] for i in range(*path.indices(len(self)))]

        return self._peripheral._get_or_create_register(self.path.join(path))

    def __setitem__(self, path: int, content: int, /) -> None:
        _register_set_content(self, path, content)

    def __iter__(self) -> Iterator[ArrayMemberT]:
        for i in range(len(self)):
            yield self[i]

    def __reversed__(self) -> Iterator[ArrayMemberT]:
        for i in reversed(range(len(self))):
            yield self[i]

    def __len__(self) -> int:
        """:return: Number of registers in the register array."""
        return self._spec.dimensions.length

    @property
    def leaf(self) -> bool:
        return False

    def child_iter(self) -> Reversible[ArrayMemberT]:
        return ChildIter(range(len(self)), self.__getitem__)

    def __repr__(self) -> str:
        """Short register description."""
        return svd_element_repr(
            self.__class__, str(self.path), address=self.offset, length=len(self)
        )


class _Struct(_RegisterNode[_StructP, EPathType], Mapping[str, RegisterKindType]):
    """Class implementing common struct functionality."""

    __slots__ = ["_registers"]

    def __init__(self, **kwargs: Any) -> None:
        """See parent class for a description of parameters."""
        super().__init__(**kwargs)

        self._registers: Optional[Mapping[str, RegisterKindType]] = None

    @property
    def leaf(self) -> bool:
        return False

    @property
    def registers(self) -> Mapping[str, RegisterKindType]:
        """:return A mapping of registers in the structure, ordered by ascending address."""
        if self._registers is None:
            self._registers = LazyFixedMapping(
                keys=iter(self), factory=self.__getitem__
            )

        return self._registers

    def __getitem__(
        self, path: Union[str, Sequence[Union[str, int]]]
    ) -> RegisterKindType:
        """
        :param index: Index of the register in the register array.
        :return: The instance of the specified register.
        """
        return self._peripheral._get_or_create_register(self.path.join(path))  # type: ignore

    def __iter__(self) -> Iterator[str]:
        """:return: Iterator over the names of registers in the register structure."""
        return iter(self._spec.registers)

    def __len__(self) -> int:
        """:return: Number of registers in the register structure"""
        return len(self._spec.registers)

    def child_iter(self) -> Reversible[RegisterKindType]:
        return ChildIter(self._spec.registers.keys(), self.__getitem__)

    def __repr__(self) -> str:
        """Short register description."""
        return svd_element_repr(self.__class__, str(self.path), address=self.offset)


class FlatStruct(_Struct[FEPath, FlatRegisterUnion]):
    """
    Flat structure representing a group of registers.
    These registers can be used to to read the properties defined on the SVD
    elements and inspect the structure of the SVD device.
    Flat structures are located at the offset specified on the corresponding element
    in the SVD file.

    A FlatStruct instance corresponds directly to a SVD cluster element,
    with or without dimensions.
    """

    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Dimensions of the register, if any."""
        return self._spec.dimensions

    def __repr__(self) -> str:
        """Short struct description."""
        return svd_element_repr(
            self.__class__,
            str(self.path),
            address=self.offset,
            length=self.dimensions.length if self.dimensions is not None else None,
        )


class Struct(_Struct[EPath, RegisterUnion]):
    """
    Regular structure representing a group of registers.
    The address offset of a regular struct accounts for array element offsets of any
    parent elements.

    A Struct instance corresponds to either a SVD cluster element without dimensions,
    or a specific index of a cluster array element.
    """

    def __setitem__(self, path: str, content: int) -> None:
        _register_set_content(self, path, content)


class _Register(_RegisterNode[_RegisterP, EPathType], Mapping[str, FieldKindType]):
    """Commmon regular/flat register functionality."""

    __slots__ = ["_fields"]

    def __init__(self, **kwargs: Any) -> None:
        """See parent class for a description of parameters."""
        super().__init__(**kwargs)

        self._fields: Optional[Mapping[str, FieldKindType]] = None

    @property
    def leaf(self) -> bool:
        return True

    @property
    def bit_width(self) -> int:
        """Bit width of the register."""
        return self._spec.reg_props.size

    @property
    def access(self) -> Access:
        """Register access."""
        return self._spec.reg_props.access

    @property
    def reset_content(self) -> int:
        """Register reset value."""
        return self._spec.reg_props.reset_value

    @property
    def reset_mask(self) -> int:
        """Mask of bits in the register that are affected by a reset."""
        return self._spec.reg_props.reset_mask

    @property
    def write_action(self) -> WriteAction:
        """Side effect of writing the register"""
        return self._spec.element.modified_write_values

    @property
    def read_action(self) -> Optional[ReadAction]:
        """Side effect of reading from the register"""
        return self._spec.element.read_action

    @property
    def fields(self) -> Mapping[str, FieldKindType]:
        """Map of fields in the register, indexed by name"""
        if self._fields is None:
            self._fields = LazyFixedMapping(
                keys=self._spec.fields.keys(),
                factory=self._create_field,
            )

        return MappingProxyType(self._fields)

    def __getitem__(self, name: str) -> FieldKindType:
        """
        :param name: Field name.
        :return: The instance of the specified field.
        """
        try:
            return self.fields[name]
        except LookupError as e:
            raise SvdKeyError(
                name, self, explanation="no field matching the given path was found"
            ) from e

    def __iter__(self) -> Iterator[str]:
        """:return: Iterator over the field names in the register."""
        return iter(self._spec.fields)

    def __len__(self) -> int:
        """:return: Number of fields in the register."""
        return len(self._spec.fields)

    @abstractmethod
    def _create_field(self, name: str) -> FieldKindType:
        """Initialize the field with the given name."""
        ...


class FlatRegister(_Register[FEPath, "FlatField"]):
    """
    Flat register instance.
    These registers can be used to to read the properties defined on the SVD
    elements and inspect the structure of the SVD device.
    Flat registers ar located at the offset specified on the element in the SVD file.
    Unlike regular Register instances, flat registers do not contain a memory value.

    A FlatRegister instance corresponds directly to a SVD register element,
    with or without dimensions.
    """

    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Dimensions of the register, if any."""
        return self._spec.dimensions

    def _create_field(self, name: str) -> FlatField:
        return FlatField(spec=self._spec.fields[name], register=self)

    def __repr__(self) -> str:
        return svd_element_repr(
            self.__class__,
            str(self.path),
            address=self.offset,
            length=self.dimensions.length if self.dimensions is not None else None,
        )


class Register(_Register[EPath, "Field"]):
    """
    Regular register instance.
    The address offset of a regular register accounts for array element offsets of any
    parent elements.
    Regular registers can be used to query or update the memory content of the peripheral.

    A Register instance corresponds to either a SVD register element without dimensions,
    or a specific index of a register array element.
    """

    @property
    def leaf(self) -> bool:
        return True

    @property
    def address_range(self) -> range:
        return range(self.address, self.address + self.bit_width // 8)

    @property
    def offset_range(self) -> range:
        """Range of addresses covered by the register."""
        return range(self.offset, self.offset + self.bit_width // 8)

    def __setitem__(self, key: str, content: Union[str, int]) -> None:
        """
        :param key: Either the bit offset of a field, or the field's name.
        :param content: A raw numeric value, or a field enumeration, to write
            to the selected register field.
        """
        self[key].content = content  # type: ignore

    @property
    def modified(self) -> bool:
        """True if the register contains a different value now than at reset."""
        return self.content != self.reset_content

    @property
    def written(self) -> bool:
        """True if the register content has been written."""
        return self._peripheral._memory_block.is_written(self.offset)

    @property
    def content(self) -> int:
        """Current value of the register."""
        content = self._peripheral._memory_block.at(self.offset, self.bit_width // 8)
        return int(content)

    @content.setter
    def content(self, new_content: int) -> None:
        """
        Set the value of the register.

        :param new_content: New value for the register.
        """
        self.set_content(new_content)

    def set_content(self, new_content: int, mask: Optional[int] = None) -> None:
        """
        Set the value of the register.

        :param new_content: New value for the register.
        :param mask: Mask of the bits to copy from the given value. If None, all bits are copied.
        """
        reg_width = self.bit_width

        if new_content.bit_length() > reg_width:
            raise SvdMemoryError(
                f"Value {hex(new_content)} is too large for {reg_width}-bit register {self.path}."
            )

        for field in self.values():
            # Only check fields that are affected by the mask
            if mask is None or mask & field.mask:
                field_content = field._extract_content_from_register(new_content)
                if field_content not in field.allowed_values:
                    raise SvdMemoryError(
                        f"Value {hex(new_content)} is invalid for register {self.path}, as field "
                        f"{field.name} does not accept the value {hex(field_content)}."
                    )

        if mask is not None:
            # Update only the bits indicated by the mask
            new_content = (self.content & ~mask) | (new_content & mask)
        else:
            new_content = new_content

        self._peripheral._memory_block.set_at(
            self.offset, new_content, item_size=reg_width // 8
        )

    def unconstrain(self) -> None:
        """Remove all value constraints imposed on the register."""
        for field in self.values():
            field.unconstrain()

    def _create_field(self, name: str) -> Field:
        return Field(spec=self._spec.fields[name], register=self)

    def __repr__(self) -> str:
        """Short register description."""
        bool_props = ("modified",) if self.modified else ()

        return svd_element_repr(
            self.__class__,
            str(self.path),
            address=self.offset,
            content=self.content,
            bool_props=bool_props,
        )


class _FieldSpec(NamedTuple):
    """
    Class containing immutable data describing a SVD field element.
    This is separated from the Field class to optimize construction of derived peripherals.
    Since the description is not tied to a specific Peripheral instance, it can be
    instantiated once per field in the SVD document and reused for derived peripherals.
    """

    name: str
    bit_range: bindings.BitRange
    enums: Dict[str, int]
    allowed_values: Collection[int]
    element: bindings.FieldElement

    @classmethod
    def from_element(cls, element: bindings.FieldElement) -> _FieldSpec:
        """
        :param element: SVD field element binding object.
        :return: Description of the field.
        """

        name = element.name
        bit_range = element.bit_range

        # We do not support "do not care" bits, as by marking bits "x", see
        # SVD docs "/device/peripherals/peripheral/registers/.../enumeratedValue"
        if element.enumerated_values is not None:
            enums = {e.name: e.value for e in element.enumerated_values.enums}
        else:
            enums = {}

        allowed_values = set(enums.values()) if enums else range(2**bit_range.width)

        return cls(
            name=name,
            bit_range=bit_range,
            enums=enums,
            allowed_values=allowed_values,
            element=element,
        )


FieldParent = TypeVar("FieldParent", Register, FlatRegister)


class _Field(Generic[FieldParent]):
    """Common regular/flat field functionality."""

    __slots__ = ["_spec", "_register", "_allowed_values"]

    def __init__(
        self,
        spec: _FieldSpec,
        register: FieldParent,
    ):
        """
        Initialize the class attribute(s).

        :param description: Field description.
        :param register: Register to which the field belongs.
        """
        self._spec: _FieldSpec = spec
        self._register: FieldParent = register
        self._allowed_values: Collection[int] = spec.allowed_values

    @property
    def name(self) -> str:
        """Name of the field."""
        return self._spec.name

    @property
    def reset_content(self) -> int:
        """Default field value."""
        return self._extract_content_from_register(self._register.reset_content)

    @property
    def bit_offset(self) -> int:
        """Bit offset of the field."""
        return self._spec.bit_range.offset

    @property
    def bit_width(self) -> int:
        """Bit width of the field."""
        return self._spec.bit_range.width

    @property
    def mask(self) -> int:
        """Bitmask of the field."""
        return ((1 << self.bit_width) - 1) << self.bit_offset

    @property
    def access(self) -> Access:
        """Access property of the field."""
        if (field_access := self._spec.element.access) is not None:
            return field_access
        return self._register.access

    @property
    def write_action(self) -> WriteAction:
        """Side effect of writing to the field."""
        if (field_write_action := self._spec.element.modified_write_values) is not None:
            return field_write_action
        return self._register.write_action

    @property
    def read_action(self) -> Optional[ReadAction]:
        """Side effect of reading from the field."""
        if (field_read_action := self._spec.element.read_action) is not None:
            return field_read_action
        return self._register.read_action

    @property
    def allowed_values(self) -> Collection[int]:
        """
        Possible valid values for the bitfield.
        By default, the values allowed for the field are defined by the field enumeration
        values. If the field does not have enumerations, all values that fit within the
        field bit width are allowed.
        """
        return self._allowed_values

    @property
    def enums(self) -> Mapping[str, int]:
        """
        A mapping between field enumerations and their corresponding values.
        Field enumerations are values such as "Allowed" = 1, "NotAllowed" = 0
        and are defined by the device's SVD file. This may be an empty map,
        if enumerations are not applicable to the field.
        """
        return self._spec.enums

    def _extract_content_from_register(self, register_content: int) -> int:
        """
        Internal method for extracting the field value from the parent register value.

        :param register_value: Value of the parent register
        :return: Field value extracted based on the field bit range
        """
        return (register_content & self.mask) >> self.bit_offset


class FlatField(_Field):
    """
    Flat field instance.
    These fields can be used to to read the properties defined on the SVD
    elements and inspect the structure of the SVD device.
    Unlike regular Field instances, FlatFields do not contain a memory value.

    A FlatField instance corresponds directly to a SVD field element.
    """

    def __repr__(self) -> str:
        """Short field description."""
        return svd_element_repr(
            self.__class__,
            self.name,
            content_max_width=self.bit_width,
        )


class Field(_Field):
    """
    Regular field instance.
    Regular fields can be used to query or update the memory content of the peripheral
    in addition to reading the properties defined on the SVD elements.

    A Field instance corresponds directly to a SVD field element.
    """

    @property
    def content(self) -> int:
        """The value of the field."""
        return self._extract_content_from_register(self._register.content)

    @content.setter
    def content(self, new_content: Union[int, str]) -> None:
        """
        Set the value of the field.

        :param value: A numeric value, or the name of a field enumeration (if applicable), to
        write to the field.
        """
        if isinstance(new_content, int):
            if new_content not in self.allowed_values:
                raise ValueError(
                    f"{self!r} does not accept"
                    f" the bit value '{new_content}' ({hex(new_content)})."
                )
            resolved_value = new_content
        elif isinstance(new_content, str):
            if new_content not in self.enums:
                raise ValueError(
                    f"{self!r} does not accept"
                    f" the enum '{new_content}'."
                )
            resolved_value = self.enums[new_content]
        else:
            raise TypeError(
                f"Field does not accept write of '{new_content}' of type '{type(new_content)}'"
                " Permitted values types are 'str' (field enum) and 'int' (bit value)."
            )

        self._register.set_content(resolved_value << self.bit_offset, self.mask)

    @property
    def content_enum(self) -> str:
        """The name of the enum corresponding to the field value."""
        content = self.content
        for enum_str, value in self.enums.items():
            if content == value:
                return enum_str

        raise LookupError(
            f"{self!r} content '{content}' ({hex(content)}) does not correspond to an enum."
        )

    @property
    def modified(self) -> bool:
        """True if the field contains a different value now than at reset."""
        return self.content != self.reset_content

    def unconstrain(self) -> None:
        """
        Remove restrictions on values that may be entered into this field. After this,
        the field will accept any value that can fit inside its bit width.
        """
        self._allowed_values = range(2**self.bit_width)

    def __repr__(self) -> str:
        """Short field description."""
        bool_props = ("modified",) if self.modified else ()

        return svd_element_repr(
            self.__class__,
            self.name,
            content=self.content,
            content_max_width=self.bit_width,
            bool_props=bool_props,
        )


class _ExtractedRegisterInfo(NamedTuple):
    """Container for register descriptions and reset values."""

    register_specs: Mapping[str, _RegisterSpec]
    memory_builder: MemoryBlock.Builder
    address_bounds: Optional[Tuple[int, int]]


def _extract_register_info(
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    options: Options,
    base_reg_props: bindings.RegisterProperties,
    base_specs: Optional[Mapping[str, _RegisterSpec]] = None,
    base_memory: Optional[Callable[[], MemoryBlock]] = None,
    base_address_bounds: Optional[Tuple[int, int]] = None,
) -> _ExtractedRegisterInfo:
    """
    Extract register descriptions for the given SVD register level elements.
    The returned structure mirrors the structure of the SVD elements.
    Each level of the structure is internally sorted by ascending address.

    :param elements: Register level elements to process.
    :param options: Parsing options.
    :param base_reg_props: Register properties of the peripheral.
    :param base_descriptions: Register descriptions inherited from the base peripheral, if any.
    :param base_memory: Memory inherited from the base peripheral, if any.
    :return: Map of register descriptions, indexed by name.
    """
    memory_builder = MemoryBlock.Builder()

    if base_memory is not None:
        memory_builder.lazy_copy_from(base_memory)

    spec_list, min_address, max_address = _extract_register_descriptions_helper(
        memory=memory_builder,
        elements=elements,
        options=options,
        base_reg_props=base_reg_props,
    )

    specs = {d.name: d for d in spec_list}

    if base_specs is not None:
        # The register maps are each sorted internally, but need to be merged by address
        # to ensure sorted order in the combined map
        specs = dict(
            iter_merged(
                specs.items(),
                base_specs.items(),
                key=lambda kv: kv[1].offset_start,
            )
        )

    address_bounds: Optional[Tuple[int, int]]

    if spec_list:
        # Use the child address range if there is at least one child
        memory_builder.set_extent(offset=min_address, length=max_address - min_address)
        if base_address_bounds is not None:
            base_min, base_max = base_address_bounds
            address_bounds = (min(min_address, base_min), max(max_address, base_max))
        else:
            address_bounds = (min_address, max_address)
    else:
        address_bounds = base_address_bounds

    if base_reg_props.reset_value is None:
        # We don't have sufficient context to report this error in a nice way here,
        # so instead we raise this so that the caller can handle it.
        raise _MissingDefaultResetValueError()

    memory_builder.set_default_content(base_reg_props.reset_value)

    return _ExtractedRegisterInfo(specs, memory_builder, address_bounds)


class _MissingDefaultResetValueError(RuntimeError):
    """Signal that indicates a missing default reset value"""

    ...


class _ExtractHelperResult(NamedTuple):
    register_specs: List[_RegisterSpec]
    min_address: int
    max_address: int


def _extract_register_descriptions_helper(
    memory: MemoryBlock.Builder,
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    options: Options,
    base_reg_props: bindings.RegisterProperties,
    base_address: int = 0,
) -> _ExtractHelperResult:
    """
    Helper that recursively extracts the names, addresses, register properties, dimensions,
    fields etc. of a collection of SVD register level elements.

    :param memory: Memory builder.
    :param elements: SVD register level elements.
    :param options: Parsing options.
    :param base_reg_props: Base address of the parent SVD element.
    :param base_address: Base address of the parent SVD element.

    :return: Extraction result.
    """

    specs: List[_RegisterSpec] = []
    min_address: int = 2**32
    max_address: int = 0

    for element in elements:
        # Remove suffixes used for elements with dimensions
        name = strip_suffix(element.name, "[%s]")
        reg_props = element.get_register_properties(base_props=base_reg_props)
        dim_props = element.dimensions
        address_offset = element.offset

        registers: Optional[Mapping[str, _RegisterSpec]] = None
        fields: Optional[Mapping[str, _FieldSpec]] = None

        if isinstance(element, bindings.RegisterElement):
            if not FullRegisterProperties.is_full(reg_props):
                raise SvdDefinitionError(
                    [element],
                    "Missing required register properties. "
                    f"Register has the following properties: {reg_props}",
                )

            # Register addresses are defined relative to the enclosing element
            if address_offset is not None:
                address_start = base_address + address_offset
            else:
                address_start = base_address

            size_bytes = reg_props.size // 8

            # Contiguous fill
            if dim_props is None or dim_props.step == size_bytes:
                length = dim_props.length if dim_props is not None else 1
                address_end = address_start + size_bytes * length
                memory.fill(
                    start=address_start,
                    end=address_end,
                    content=reg_props.reset_value,
                    item_size=size_bytes,
                )

            # Fill with gaps
            elif dim_props is not None and dim_props.step > size_bytes:
                memory.fill(
                    start=address_start,
                    end=address_start + size_bytes,
                    content=reg_props.reset_value,
                    item_size=size_bytes,
                )
                memory.tile(
                    start=address_start,
                    end=address_start + dim_props.step,
                    times=dim_props.length,
                )

            else:
                raise SvdDefinitionError(
                    element,
                    f"Step of 0x{dim_props.step:x} is less than the size of the "
                    f"array (0x{size_bytes})",
                )

            fields = _extract_field_descriptions(element.fields)

        else:  # ClusterElement
            if address_offset is not None:
                # By the SVD specification, cluster addresses are defined relative to the peripheral
                # base address, but some SVDs don't follow this rule.
                address_start = (
                    base_address
                    if not options.parent_relative_cluster_address
                    else base_address + address_offset
                )
            else:
                address_start = base_address

            (
                sub_specs,
                sub_min_address,
                sub_max_address,
            ) = _extract_register_descriptions_helper(
                memory=memory,
                elements=element.registers,
                options=options,
                base_reg_props=reg_props,
                base_address=address_start,
            )

            if sub_specs:
                registers = {d.name: d for d in sub_specs}

                if address_offset is None:
                    address_start = sub_min_address

                if dim_props is not None and dim_props.length > 1:
                    if dim_props.step < sub_max_address - address_start:
                        raise SvdDefinitionError(
                            element,
                            f"Step of 0x{dim_props.step:x} is less than the size required to "
                            f"cover all the child elements (0x{sub_max_address - address_start:x})",
                        )

                    address_end = address_start + dim_props.step * dim_props.length

                    # Copy memory from sub elements along the struct array dimension
                    memory.tile(
                        start=address_start,
                        end=address_start + dim_props.step,
                        times=dim_props.length,
                    )

                else:  # Not an array
                    address_end = sub_max_address

            else:  # Empty struct
                registers = {}
                address_end = address_start

        spec = _RegisterSpec(
            element=element,
            name=name,
            offset_start=address_start,
            offset_end=address_end,
            reg_props=reg_props,
            dimensions=dim_props,
            registers=registers,
            fields=fields,
        )

        specs.append(spec)
        min_address = min(min_address, address_start)
        max_address = max(max_address, address_end)

    sorted_result = sorted(specs, key=lambda r: r.offset_start)

    # Check that our structural assumptions hold.
    if not options.ignore_overlapping_structures:
        if len(sorted_result) > 1:
            for i in range(1, len(sorted_result)):
                r1 = sorted_result[i - 1]
                r2 = sorted_result[i]
                if r1.offset_end > r2.offset_start:
                    r1_str = f'"{r1.name}" ({r1.offset_start:#x}-{r1.offset_end:#x})'
                    r2_str = f'"{r2.name}" ({r2.offset_start:#x}-{r2.offset_end:#x})'
                    error_msg = f"Element addresses for {r1_str} and {r2_str} overlap"
                    raise SvdDefinitionError([r1.element, r2.element], error_msg)

    return _ExtractHelperResult(sorted_result, min_address, max_address)


def _extract_field_descriptions(
    elements: Iterable[bindings.FieldElement],
) -> Optional[Mapping[str, _FieldSpec]]:
    """
    Extract field descriptions for the given SVD field elements.
    The resulting mapping is internally sorted by ascending field bit offset.

    :param elements: Field elements to process.
    :return: Mapping of field descriptions, indexed by name.
    """

    field_specs = sorted(
        [_FieldSpec.from_element(field) for field in elements],
        key=lambda field: field.bit_range.offset,
    )

    fields = {spec.name: spec for spec in field_specs}

    return fields
