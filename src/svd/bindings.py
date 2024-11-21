# Copyright (c) 2023 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

"""
"Low-level" read-only Python representation of the SVD format that aims to represent the full SVD
document. Each type of element in the SVD XML tree is represented by a class in this module.
The class properties correspond more or less directly to the XML elements/attributes,
with some abstractions and simplifications added for convenience.

Based on CMSIS-SVD schema v1.3.9.
"""

from __future__ import annotations

import enum
import typing
from dataclasses import dataclass
from typing import Iterator, NamedTuple, Optional, Protocol, Sequence, Union

from lxml import objectify
from lxml.objectify import BoolElement, StringElement
from typing_extensions import TypeGuard

from ._bindings import (
    SELF_CLASS,
    Attr,
    BindingRegistry,
    CaseInsensitiveStrEnum,
    Elem,
    SvdElement,
    SvdIntElement,
    get_binding_elem_props,
    iter_element_children,
    make_enum_wrapper,
    to_bool,
    to_int,
)

# Container for classes that represent non-leaf elements in the SVD XML tree.
BINDING_REGISTRY = BindingRegistry()

# Alias for the BINDING_REGISTRY.add for convenience.
binding = BINDING_REGISTRY.add

# Alias for the BINDING_REGISTRY.bindings for convenience.
BINDINGS = BINDING_REGISTRY.bindings

# Utility function for the parse module
get_binding_elem_props = get_binding_elem_props


@enum.unique
class Access(CaseInsensitiveStrEnum):
    """
    Access rights for a given register or field.
    See "accessType" in the SVD schema.
    """

    # Read access is permitted. Write operations have an undefined result.
    READ_ONLY = "read-only"
    # Write access is permitted. Read operations have an undefined result.
    WRITE_ONLY = "write-only"
    # Read and write accesses are permitted.
    READ_WRITE = "read-write"
    # Only the first write after reset has an effect. Read operations have an undefined results.
    WRITE_ONCE = "writeOnce"
    # Only the first write after reset has an effect. Read access is permitted.
    READ_WRITE_ONCE = "read-writeOnce"


AccessElement = make_enum_wrapper(Access)


@enum.unique
class ReadAction(CaseInsensitiveStrEnum):
    """
    Side effect following a read operation of a given register or field.
    See "readActionType" in the SVD schema.
    """

    # The register/field is set to zero following a read operation.
    CLEAR = "clear"
    # The register/field is set to ones following a read operation.
    SET = "set"
    # The register/field is modified by a read operation.
    MODIFY = "modify"
    # A dependent resource is modified by a read operation.
    MODIFY_EXTERNAL = "modifyExternal"


ReadActionElement = make_enum_wrapper(ReadAction)


@enum.unique
class Endian(CaseInsensitiveStrEnum):
    """
    Processor endianness.
    See "endianType" in the SVD schema.
    """

    # Little endian
    LITTLE = "little"
    # Big endian
    BIG = "big"
    # Endianness is configurable for the device, taking effect on the next reset.
    SELECTABLE = "selectable"
    # Neither big nor little endian
    OTHER = "other"


EndianElement = make_enum_wrapper(Endian)


@enum.unique
class SauAccess(CaseInsensitiveStrEnum):
    """
    SAU region access type.
    See "sauAccessType" in the SVD schema.
    """

    # Non-secure accessible
    NON_SECURE = "n"
    # Secure callable
    SECURE_CALLABLE = "c"


SauAccessElement = make_enum_wrapper(SauAccess)


@enum.unique
class AddressBlockUsage(CaseInsensitiveStrEnum):
    """
    Defined usage type of a peripheral address block.
    See "addressBlockType" in the SVD schema.
    """

    REGISTER = "registers"
    BUFFER = "buffer"
    RESERVED = "reserved"


AddressBlockUsageElement = make_enum_wrapper(AddressBlockUsage)


@enum.unique
class Protection(CaseInsensitiveStrEnum):
    """
    Security privilege required to access an address region.
    See "protectionStringType" in the SVD schema.
    """

    # Secure permission required for access
    SECURE = "s"
    # Non-secure or secure permission required for access
    NON_SECURE = "n"
    # Privileged permission required for access
    PRIVILEGED = "p"


ProtectionElement = make_enum_wrapper(Protection)


@enum.unique
class EnumUsage(CaseInsensitiveStrEnum):
    """
    Usage of an enumerated value.
    See "enumUsageType" in the SVD schema.
    """

    # The value is relevant for read operations.
    READ = "read"
    # The value is relevant for write operations.
    WRITE = "write"
    # The value is relevant for read and write operations.
    READ_WRITE = "read-write"


EnumUsageElement = make_enum_wrapper(EnumUsage)


@enum.unique
class WriteAction(CaseInsensitiveStrEnum):
    """
    Side effect following a write operation of a given register or field.
    See "modifiedWriteValuesType" in the SVD schema.
    """

    # Bits written to one are set to zero in the register/field.
    ONE_TO_CLEAR = "oneToClear"
    # Bits written to one are set to one in the register/field.
    ONE_TO_SET = "oneToSet"
    # Bits written to one are inverted in the register/field.
    ONE_TO_TOGGLE = "oneToToggle"
    # Bits written to zero are set to zero in the register/field.
    ZERO_TO_CLEAR = "zeroToClear"
    # Bits written to zero are set to one in the register/field.
    ZERO_TO_SET = "zeroToSet"
    # Bits written to zero are inverted in the register/field.
    ZERO_TO_TOGGLE = "zeroToToggle"
    # All bits are set to zero on writing to the register/field.
    CLEAR = "clear"
    # All bits are set to one on writing to the register/field.
    SET = "set"
    # All bits are modified on writing to the register/field.
    MODIFY = "modify"


ModifiedWriteValuesElement = make_enum_wrapper(WriteAction)


@enum.unique
class DataType(CaseInsensitiveStrEnum):
    """
    Data types defined in the SVD specification.
    See "dataTypeType" in the SVD schema.
    """

    UINT8_T = "uint8_t"
    UINT16_T = "uint16_t"
    UINT32_T = "uint32_t"
    UINT64_T = "uint64_t"
    INT8_T = "int8_t"
    INT16_T = "int16_t"
    INT32_T = "int32_t"
    INT64_T = "int64_t"
    UINT8_PTR_T = "uint8_t *"
    UINT16_PTR_T = "uint16_t *"
    UINT32_PTR_T = "uint32_t *"
    UINT64_PTR_T = "uint64_t *"
    INT8_PTR_T = "int8_t *"
    INT16_PTR_T = "int16_t *"
    INT32_PTR_T = "int32_t *"
    INT64_PTR_T = "int64_t *"


DataTypeElement = make_enum_wrapper(DataType)


@enum.unique
class CpuName(CaseInsensitiveStrEnum):
    """
    CPU names defined in the SVD specification.
    See "cpuNameType" in the SVD schema.
    """

    CM0 = "CM0"
    CM0_PLUS_ = "CM0PLUS"
    CM0_PLUS = "CM0+"
    CM1 = "CM1"
    CM3 = "CM3"
    CM4 = "CM4"
    CM7 = "CM7"
    CM23 = "CM23"
    CM33 = "CM33"
    CM35P = "CM35P"
    CM55 = "CM55"
    CM85 = "CM85"
    SC000 = "SC000"
    SC300 = "SC300"
    ARMV8MML = "ARMV8MML"
    ARMV8MBL = "ARMV8MBL"
    ARMV81MML = "ARMV81MML"
    CA5 = "CA5"
    CA7 = "CA7"
    CA8 = "CA8"
    CA9 = "CA9"
    CA15 = "CA15"
    CA17 = "CA17"
    CA53 = "CA53"
    CA57 = "CA57"
    CA72 = "CA72"
    SMC1 = "SMC1"
    OTHER = "other"


CpuNameElement = make_enum_wrapper(CpuName)


@binding
class RangeWriteConstraint(SvdElement):
    """Value range constraint for a register or field."""

    TAG: str = "range"

    # Minimum permitted value
    minimum: Elem[int] = Elem("minimum", SvdIntElement)

    # Maximum permitted value
    maximum: Elem[int] = Elem("maximum", SvdIntElement)


@enum.unique
class WriteConstraint(enum.Enum):
    """Type of write constraint for a register or field."""

    # Only the last read value can be written.
    WRITE_AS_READ = enum.auto()
    # Only enumerated values can be written.
    USE_ENUMERATED_VALUES = enum.auto()
    # Only values within a given range can be written.
    RANGE = enum.auto()


@binding
class WriteConstraintElement(SvdElement):
    """Constraint on permitted values in a register or field."""

    TAG: str = "writeConstraint"

    # Value range constraint
    value_range: Elem[Optional[RangeWriteConstraint]] = Elem(
        "range", RangeWriteConstraint, default=None
    )

    def as_enum(self) -> Optional[WriteConstraint]:
        """Return the write constraint as an enum value."""
        if self._write_as_read:
            return WriteConstraint.WRITE_AS_READ
        if self._use_enumerated_values:
            return WriteConstraint.USE_ENUMERATED_VALUES
        if self.value_range is not None:
            return WriteConstraint.RANGE
        return None

    # (internal) If true, only the last read value can be written.
    _write_as_read: Elem[bool] = Elem("writeAsRead", BoolElement, default=False)

    # (internal) If true, only enumerated values can be written.
    _use_enumerated_values: Elem[bool] = Elem(
        "useEnumeratedValues", BoolElement, default=False
    )


@binding
class SauRegion(SvdElement):
    """Predefined Secure Attribution Unit (SAU) region."""

    TAG: str = "region"

    # If true, the SAU region is enabled.
    enabled: Attr[bool] = Attr("enabled", converter=to_bool, default=True)

    # Name of the SAU region.
    name: Attr[Optional[str]] = Attr("name", default=None)

    # Base address of the SAU region.
    base_address: Elem[int] = Elem("base", SvdIntElement)

    # Limit address of the SAU region.
    limit: Elem[int] = Elem("limit", SvdIntElement)

    # Access permissions of the SAU region.
    access: Elem[SauAccess] = Elem("access", SauAccessElement)


@binding
class SauRegionsConfig(SvdElement):
    """Container for predefined Secure Attribution Unit (SAU) regions."""

    TAG: str = "sauRegions"

    # If true, the SAU is enabled.
    enabled: Attr[bool] = Attr("enabled", converter=to_bool, default=True)

    # Default protection for disabled SAU regions.
    protection_when_disabled: Attr[Optional[Protection]] = Attr(
        "name", converter=Protection, default=None
    )

    @property
    def regions(self) -> Iterator[SauRegion]:
        """Iterate over all predefined SAU regions."""
        it = iter_element_children(self, SauRegion.TAG)
        return typing.cast(Iterator[SauRegion], it)

    # (internal) SAU regions.
    _region: Elem[SauRegion] = Elem("region", SauRegion)


@binding
class Cpu(SvdElement):
    """Description of the device processor."""

    TAG: str = "cpu"

    # CPU name. See CpuName for possible values.
    name: Elem[CpuName] = Elem("name", CpuNameElement)

    # CPU hardware revision with the format "rNpM".
    revision: Elem[str] = Elem("revision", StringElement)

    # Default endianness of the CPU.
    endian: Elem[Endian] = Elem("endian", EndianElement)

    # True if the CPU has a memory protection unit (MPU).
    has_mpu: Elem[Optional[bool]] = Elem("mpuPresent", BoolElement, default=None)

    # True if the CPU has a floating point unit (FPU).
    has_fpu: Elem[Optional[bool]] = Elem("fpuPresent", BoolElement, default=None)

    # True if the CPU has a double precision floating point unit (FPU).
    fpu_is_double_precision: Elem[Optional[bool]] = Elem(
        "fpuDP", BoolElement, default=None
    )

    # True if the CPU implements the SIMD DSP extensions.
    has_dsp: Elem[Optional[bool]] = Elem("dspPresent", BoolElement, default=None)

    # True if the CPU has an instruction cache.
    has_icache: Elem[Optional[bool]] = Elem("icachePresent", BoolElement, default=None)

    # True if the CPU has a data cache.
    has_dcache: Elem[Optional[bool]] = Elem("dcachePresent", BoolElement, default=None)

    # True if the CPU has an instruction tightly coupled memory (ITCM).
    has_ictm: Elem[Optional[bool]] = Elem("itcmPresent", BoolElement, default=None)

    # True if the CPU has a data tightly coupled memory (DTCM).
    has_dctm: Elem[Optional[bool]] = Elem("dtcmPresent", BoolElement, default=None)

    # True if the CPU has a Vector Table Offset Register (VTOR).
    has_vtor: Elem[bool] = Elem("vtorPresent", BoolElement, default=True)

    # Bit width of interrupt priority levels in the Nested Vectored Interrupt Controller (NVIC).
    num_nvic_priority_bits: Elem[int] = Elem("nvicPrioBits", SvdIntElement)

    # True if the CPU has a vendor-specific SysTick Timer.
    # If False, the Arm-defined System Tick Timer is used.
    has_vendor_systick: Elem[bool] = Elem("vendorSystickConfig", BoolElement)

    # Maximum interrupt number in the CPU plus one.
    num_interrupts: Elem[Optional[int]] = Elem(
        "deviceNumInterrupts", SvdIntElement, default=None
    )

    # Number of supported Secure Attribution Unit (SAU) regions.
    num_sau_regions: Elem[int] = Elem("sauNumRegions", SvdIntElement, default=0)

    # Predefined Secure Attribution Unit (SAU) regions, if any.
    preset_sau_regions: Elem[Optional[SauRegionsConfig]] = Elem(
        "sauRegionsConfig", SauRegionsConfig, default=None
    )


@binding
class AddressBlock(SvdElement):
    """Address range mapped to a peripheral."""

    TAG: str = "addressBlock"

    # Start address of the address block, relative to the peripheral base address.
    offset: Elem[int] = Elem("offset", SvdIntElement)

    # Number of address unit bits covered by the address block.
    size: Elem[int] = Elem("size", SvdIntElement)

    # Address block usage. See AddressBlockUsage for possible values.
    usage: Elem[AddressBlockUsage] = Elem("usage", AddressBlockUsageElement)

    # Protection level for the address block.
    protection: Elem[Optional[Protection]] = Elem("protection", ProtectionElement)


class DerivedMixin(objectify.ObjectifiedElement):
    """Common functionality for elements that contain a SVD 'derivedFrom' attribute."""

    # Name of the element that this element is derived from.
    derived_from: Attr[Optional[str]] = Attr("derivedFrom", default=None)

    @property
    def is_derived(self) -> bool:
        """Return True if the element is derived from another element."""
        return self.derived_from is not None


@binding
class EnumeratedValue(SvdElement):
    """Value definition for a field."""

    TAG: str = "enumeratedValue"

    # Name of the enumerated value.
    name: Elem[str] = Elem("name", StringElement)

    # Description of the enumerated value.
    description: Elem[Optional[str]] = Elem("description", StringElement, default=None)

    # Value of the enumerated value.
    value: Elem[int] = Elem("value", SvdIntElement)

    # True if the enumerated value is the default value of the field.
    is_default: Elem[bool] = Elem("isDefault", BoolElement, default=False)


@binding
class Enumeration(SvdElement, DerivedMixin):
    """Container for enumerated values."""

    TAG: str = "enumeratedValues"

    # Name of the enumeration.
    name: Elem[Optional[str]] = Elem("name", StringElement, default=None)

    # Identifier of the enumeration in the device header file.
    header_enum_name: Elem[Optional[str]] = Elem(
        "headerEnumName", StringElement, default=None
    )

    # Description of which types of operations the enumeration is used for.
    usage: Elem[EnumUsage] = Elem(
        "usage", EnumUsageElement, default=EnumUsage.READ_WRITE
    )

    @property
    def enums(self) -> Iterator[EnumeratedValue]:
        """Iterate over all enumerated values."""
        it = iter_element_children(self, EnumeratedValue.TAG)
        return typing.cast(Iterator[EnumeratedValue], it)

    # (internal) Enumerated values
    _enumerated_values: Elem[EnumeratedValue] = Elem("enumeratedValue", EnumeratedValue)


@binding
class DimArrayIndex(SvdElement):
    """Description of the index used for an array of registers."""

    TAG: str = "dimArrayIndex"

    # The base name of enumerations.
    header_enum_name: Elem[Optional[str]] = Elem(
        "headerEnumName", StringElement, default=None
    )

    # Values contained in the enumeration.
    enumerated_values: Elem[Iterator[EnumeratedValue]] = Elem(
        "enumeratedValue", EnumeratedValue
    )


@binding
class Interrupt(SvdElement):
    """Peripheral interrupt description."""

    TAG: str = "interrupt"

    # Name of the interrupt.
    name: Elem[str] = Elem("name", StringElement)

    # Description of the interrupt.
    description: Elem[Optional[str]] = Elem("description", StringElement, default=None)

    # Interrupt number.
    value: Elem[int] = Elem("value", SvdIntElement)


class BitRange(NamedTuple):
    """Bit range of a field."""

    # Bit offset of the field.
    offset: int

    # Bit width of the field.
    width: int


@binding
class FieldElement(SvdElement, DerivedMixin):
    """SVD field element."""

    TAG: str = "field"

    # Name of the field.
    name: Elem[str] = Elem("name", StringElement)

    # Description of the field.
    description: Elem[Optional[str]] = Elem("description", StringElement, default=None)

    # Access rights of the field.
    access: Elem[Optional[Access]] = Elem("access", AccessElement, default=None)

    # Side effect when writing to the field.
    modified_write_values: Elem[Optional[WriteAction]] = Elem(
        "modifiedWriteValue", ModifiedWriteValuesElement, default=None
    )

    # Constraints on writing to the field.
    write_constraint: Elem[Optional[WriteConstraintElement]] = Elem(
        "writeConstraint", WriteConstraintElement, default=None
    )

    # Side effect when reading from the field.
    read_action: Elem[Optional[ReadAction]] = Elem(
        "readAction", ReadActionElement, default=None
    )

    # Permitted values of the field.
    enumerated_values: Elem[Optional[Enumeration]] = Elem(
        "enumeratedValues", Enumeration, default=None
    )

    @property
    def bit_range(self) -> BitRange:
        """
        Bit range of the field.
        :return: Tuple of the field's bit offset and bit width.
        """

        if self._lsb is not None and self._msb is not None:
            return BitRange(offset=self._lsb, width=self._msb - self._lsb + 1)

        if self._bit_offset is not None:
            width = self._bit_width if self._bit_width is not None else 32
            return BitRange(offset=self._bit_offset, width=width)

        if self._bit_range is not None:
            msb_string, lsb_string = self._bit_range[1:-1].split(":")
            msb, lsb = to_int(msb_string), to_int(lsb_string)
            return BitRange(offset=lsb, width=msb - lsb + 1)

        return BitRange(offset=0, width=32)

    # (internal) Least significant bit of the field, if specified in the bitRangeLsbMsbStyle style.
    _lsb: Elem[Optional[int]] = Elem("lsb", SvdIntElement, default=None)

    # (internal) Most significant bit of the field, if specified in the bitRangeLsbMsbStyle style.
    _msb: Elem[Optional[int]] = Elem("msb", SvdIntElement, default=None)

    # (internal) Bit offset of the field, if specified in the bitRangeOffsetWidthStyle style.
    _bit_offset: Elem[Optional[int]] = Elem("bitOffset", SvdIntElement, default=None)

    # (internal) Bit width of the field, if specified in the bitRangeOffsetWidthStyle style.
    _bit_width: Elem[Optional[int]] = Elem("bitWidth", SvdIntElement, default=None)

    # (internal) Bit range of the field, given in the form "[msb:lsb]", if specified in the
    # bitRangePattern style.
    _bit_range: Elem[Optional[str]] = Elem("bitRange", StringElement, default=None)

    def __repr__(self) -> str:
        try:
            name = self.name
        except Exception:
            name = None

        return super()._repr(props={"name": name})


@binding
class FieldsElement(SvdElement):
    """Container for SVD field elements."""

    TAG: str = "fields"

    # Field elements.
    field: Elem[FieldElement] = Elem("field", FieldElement)


@dataclass
class RegisterProperties:
    """Common SVD device/peripheral/register level properties."""

    # Size of the register in bits.
    size: Optional[int]

    # Access rights of the register.
    access: Optional[Access]

    # Protection level of the register.
    protection: Optional[Protection]

    # Reset value of the register.
    reset_value: Optional[int]

    # Reset mask of the register.
    reset_mask: Optional[int]


class FullRegisterProperties(Protocol):
    """Protocol that describes a fully defined register properties object."""

    size: int
    access: Access
    protection: Optional[Protection]
    reset_value: int
    reset_mask: int

    @staticmethod
    def is_full(props: RegisterProperties) -> TypeGuard[FullRegisterProperties]:
        """Check if the given register properties has all the required fields set."""
        return all(
            f is not None
            for f in (
                props.size,
                props.access,
                props.reset_value,
                props.reset_mask,
            )
        )


class RegisterPropertiesGroupMixin(objectify.ObjectifiedElement):
    """Common functionality for elements that contain a SVD 'registerPropertiesGroup'."""

    @property
    def register_properties(self) -> RegisterProperties:
        """Register properties specified in the element itself."""
        return self.get_register_properties()

    def get_register_properties(
        self, base_props: Optional[RegisterProperties] = None
    ) -> RegisterProperties:
        """
        Get the register properties of the element, optionally inheriting from a
        base set of properties.
        """
        if base_props is None:
            return RegisterProperties(
                size=self._size,
                access=self._access,
                protection=self._protection,
                reset_value=self._reset_value,
                reset_mask=self._reset_mask,
            )

        return RegisterProperties(
            size=self._size if self._size is not None else base_props.size,
            access=self._access if self._access is not None else base_props.access,
            protection=(
                self._protection
                if self._protection is not None
                else base_props.protection
            ),
            reset_value=(
                self._reset_value
                if self._reset_value is not None
                else base_props.reset_value
            ),
            reset_mask=(
                self._reset_mask
                if self._reset_mask is not None
                else base_props.reset_mask
            ),
        )

    _size: Elem[Optional[int]] = Elem("size", SvdIntElement, default=None)
    _access: Elem[Optional[Access]] = Elem("access", AccessElement, default=None)
    _protection: Elem[Optional[Protection]] = Elem(
        "protection", ProtectionElement, default=None
    )
    _reset_value: Elem[Optional[int]] = Elem("resetValue", SvdIntElement, default=None)
    _reset_mask: Elem[Optional[int]] = Elem("resetMask", SvdIntElement, default=None)


@dataclass
class Dimensions:
    """Dimensions of a repeated SVD element."""

    # Number of times the element is repeated.
    length: int

    # Increment between each element.
    step: int

    def to_range(self) -> Sequence[int]:
        """Convert to a range of offsets"""
        return range(0, (self.length - 1) * self.step + 1, self.step)


class DimElementGroupMixin(objectify.ObjectifiedElement):
    """Common functionality for elements that contain a SVD 'dimElementGroup'."""

    # Index of the element, if it is repeated.
    dim_index: Elem[Optional[int]] = Elem("dimIndex", SvdIntElement, default=None)

    # Name of the dimension, if it is repeated.
    dim_name: Elem[Optional[str]] = Elem("dimName", StringElement, default=None)

    # Array index of the element, if it is repeated.
    dim_array_index: Elem[Optional[DimArrayIndex]] = Elem(
        "dimArrayIndex", DimArrayIndex, default=None
    )

    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Get the dimensions of the element, if it is repeated."""
        if self._dim is None or self._dim_increment is None:
            return None

        return Dimensions(
            length=self._dim,
            step=self._dim_increment,
        )

    _dim: Elem[Optional[int]] = Elem("dim", SvdIntElement, default=None)
    _dim_increment: Elem[Optional[int]] = Elem(
        "dimIncrement", SvdIntElement, default=None
    )


@binding
class RegisterElement(
    SvdElement,
    DimElementGroupMixin,
    RegisterPropertiesGroupMixin,
    DerivedMixin,
):
    """SVD register element."""

    TAG: str = "register"

    # Name of the register.
    name: Elem[str] = Elem("name", StringElement)

    # Display name of the register.
    display_name: Elem[Optional[str]] = Elem("displayName", StringElement, default=None)

    # Description of the register.
    description: Elem[Optional[str]] = Elem("description", StringElement, default=None)

    # Alternate group of the register.
    alternate_group: Elem[Optional[str]] = Elem(
        "alternateGroup", StringElement, default=None
    )

    # Name of a different register that corresponds to this register.
    alternate_register: Elem[Optional[str]] = Elem(
        "alternateRegister", StringElement, default=None
    )

    # Address offset of the register, relative to the parent element.
    offset: Elem[Optional[int]] = Elem("addressOffset", SvdIntElement, default=None)

    # C data type to use when accessing the register.
    data_type: Elem[Optional[DataType]] = Elem(
        "dataType", DataTypeElement, default=None
    )

    # Side effect of writing the register.
    modified_write_values: Elem[WriteAction] = Elem(
        "modifiedWriteValues", ModifiedWriteValuesElement, default=WriteAction.MODIFY
    )

    # Write constraint of the register.
    write_constraint: Elem[Optional[WriteConstraintElement]] = Elem(
        "writeConstraint", WriteConstraintElement, default=None
    )

    # Side effect of reading the register.
    read_action: Elem[Optional[ReadAction]] = Elem(
        "readAction", ReadActionElement, default=None
    )

    @property
    def fields(self) -> Iterator[FieldElement]:
        """Iterator over the fields of the register."""
        it = iter_element_children(self._fields, FieldElement.TAG)
        return typing.cast(Iterator[FieldElement], it)

    # (internal) Fields of the register.
    _fields: Elem[Optional[FieldElement]] = Elem("fields", FieldsElement, default=None)

    def __repr__(self) -> str:
        try:
            name = self.name
        except Exception:
            name = None

        return super()._repr(props={"name": name})


@binding
class ClusterElement(
    SvdElement,
    DimElementGroupMixin,
    RegisterPropertiesGroupMixin,
    DerivedMixin,
):
    """SVD cluster element."""

    TAG: str = "cluster"

    # Name of the cluster.
    name: Elem[str] = Elem("name", StringElement)

    # Description of the cluster.
    description: Elem[Optional[str]] = Elem("description", StringElement, default=None)

    # Name of a different cluster that corresponds to this cluster.
    alternate_cluster: Elem[Optional[str]] = Elem(
        "alternateCluster", StringElement, default=None
    )

    # Name of the C struct used to represent the cluster.
    header_struct_name: Elem[Optional[str]] = Elem(
        "headerStructName", StringElement, default=None
    )

    # Address offset of the cluster, relative to the parent peripheral element.
    offset: Elem[Optional[int]] = Elem("addressOffset", SvdIntElement, default=None)

    @property
    def registers(self) -> Iterator[Union[RegisterElement, ClusterElement]]:
        """Iterator over the registers and clusters that are direct children of this cluster."""
        it = iter_element_children(self, RegisterElement.TAG, ClusterElement.TAG)
        return typing.cast(Iterator[Union[RegisterElement, ClusterElement]], it)

    # (internal) Register elements in the cluster.
    _register: Elem[Optional[RegisterElement]] = Elem(
        "register", RegisterElement, default=None
    )

    # (internal) Cluster elements in the cluster.
    _cluster: Elem[Optional[ClusterElement]] = Elem("cluster", SELF_CLASS, default=None)

    def __repr__(self) -> str:
        try:
            name = self.name
        except Exception:
            name = None

        return super()._repr(props={"name": name})


@binding
class RegistersElement(SvdElement):
    """Container for SVD register/cluster elements."""

    TAG: str = "registers"

    # Cluster elements in the container.
    cluster: Elem[Optional[ClusterElement]] = Elem(
        "cluster", ClusterElement, default=None
    )

    # Register elements in the container.
    register: Elem[Optional[RegisterElement]] = Elem(
        "register", RegisterElement, default=None
    )


@binding
class PeripheralElement(
    SvdElement,
    DimElementGroupMixin,
    RegisterPropertiesGroupMixin,
    DerivedMixin,
):
    """SVD peripheral element."""

    TAG: str = "peripheral"

    # Name of the peripheral.
    name: Elem[str] = Elem("name", StringElement)

    # Version of the peripheral.
    version: Elem[Optional[str]] = Elem("version", StringElement, default=None)

    # Description of the peripheral.
    description: Elem[Optional[str]] = Elem("description", StringElement, default=None)

    # Base address of the peripheral.
    base_address: Elem[int] = Elem("baseAddress", SvdIntElement)

    @property
    def interrupts(self) -> Iterator[Interrupt]:
        """Iterator over the interrupts of the peripheral."""
        it = iter_element_children(self, Interrupt.TAG)
        return typing.cast(Iterator[Interrupt], it)

    @property
    def address_blocks(self) -> Iterator[AddressBlock]:
        """Iterator over the address blocks of the peripheral."""
        it = iter_element_children(self, AddressBlock.TAG)
        return typing.cast(Iterator[AddressBlock], it)

    @property
    def registers(self) -> Iterator[Union[RegisterElement, ClusterElement]]:
        """Iterator over the registers and clusters that are direct children of this peripheral."""
        it =iter_element_children(
            self._registers, RegisterElement.TAG, ClusterElement.TAG
        )
        return typing.cast(Iterator[Union[RegisterElement, ClusterElement]], it)

    # Name of a different peripheral that corresponds to this peripheral.
    alternate_peripheral: Elem[Optional[str]] = Elem(
        "alternatePeripheral", StringElement, default=None
    )

    # Name of the group that the peripheral belongs to.
    group_name: Elem[Optional[str]] = Elem("groupName", StringElement, default=None)

    # String to prepend to the names of registers contained in the peripheral.
    prepend_to_name: Elem[Optional[str]] = Elem(
        "prependToName", StringElement, default=None
    )

    # String to append to the names of registers contained in the peripheral.
    append_to_name: Elem[Optional[str]] = Elem(
        "appendToName", StringElement, default=None
    )

    # Name of the C struct that represents the peripheral.
    header_struct_name: Elem[Optional[str]] = Elem(
        "headerStructName", StringElement, default=None
    )

    disable_condition: Elem[Optional[str]] = Elem(
        "disableCondition", StringElement, default=None
    )

    # (internal) Interrupt elements in the peripheral.
    _interrupts: Elem[Optional[Interrupt]] = Elem("interrupt", Interrupt, default=None)

    # (internal) Address block elements in the peripheral.
    _address_blocks: Elem[Optional[AddressBlock]] = Elem(
        "addressBlock", AddressBlock, default=None
    )

    # (internal) Register/cluster container.
    _registers: Elem[Optional[RegistersElement]] = Elem(
        "registers", RegistersElement, default=None
    )

    def __repr__(self) -> str:
        try:
            name = self.name
        except Exception:
            name = None

        props = {"name": name}

        if (derived_from := self.derived_from) is not None:
            props["derived_from"] = derived_from

        return super()._repr(props=props)


@binding
class PeripheralsElement(SvdElement):
    """Container for SVD peripheral elements."""

    TAG: str = "peripherals"

    # Peripheral elements in the container.
    peripheral: Elem[Optional[PeripheralElement]] = Elem(
        "peripheral", PeripheralElement, default=None
    )


@binding
class DeviceElement(SvdElement, RegisterPropertiesGroupMixin):
    """SVD device element."""

    TAG: str = "device"

    # Version of the CMSIS schema that the SVD file conforms to.
    schema_version: Attr[float] = Attr("schemaVersion", converter=float)

    # Name of the device.
    name: Elem[str] = Elem("name", StringElement)

    # Device series name.
    series: Elem[Optional[str]] = Elem("series", StringElement, default=None)

    # Version of the device.
    version: Elem[str] = Elem("version", StringElement)

    # Full device vendor name.
    vendor: Elem[Optional[str]] = Elem("vendor", StringElement, default=None)

    # Abbreviated device vendor name.
    vendor_id: Elem[Optional[str]] = Elem("vendorID", StringElement, default=None)

    # Description of the device.
    description: Elem[str] = Elem("description", StringElement)

    # The license to use for the device header file.
    license_text: Elem[Optional[str]] = Elem("licenseText", StringElement)

    # Description of the device processor.
    cpu: Elem[Optional[Cpu]] = Elem("cpu", Cpu, default=None)

    # Device header filename without extension.
    header_system_filename: Elem[Optional[str]] = Elem(
        "headerSystemFilename", StringElement, default=None
    )

    # String to prepend to all type definitions in the device header file.
    header_definitions_prefix: Elem[Optional[str]] = Elem(
        "headerDefinitionsPrefix", StringElement, default=None
    )

    # Number of data bits selected by each address.
    address_unit_bits: Elem[int] = Elem("addressUnitBits", SvdIntElement)

    # Width of the maximum data transfer supported by the device.
    width: Elem[int] = Elem("width", SvdIntElement)

    @property
    def peripherals(self) -> Iterator[PeripheralElement]:
        """Iterate over all peripherals in the device"""
        it = iter_element_children(self._peripherals, PeripheralElement.TAG)
        return typing.cast(Iterator[PeripheralElement], it)

    # (internal) Peripheral elements in the device.
    _peripherals: Elem[PeripheralElement] = Elem(
        "peripherals",
        PeripheralsElement,
    )
