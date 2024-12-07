# Copyright (c) 2022 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

from .bindings import (
    Access,
    ReadAction,
    Endian,
    SauAccess,
    AddressBlockUsage,
    Protection,
    EnumUsage,
    WriteAction,
    DataType,
    CpuName,
    Cpu,
    AddressBlock,
    SauRegion,
)
from .errors import (
    SvdError,
    SvdParseError,
    SvdDefinitionError,
    SvdMemoryError,
    SvdPathError,
    SvdIndexError,
    SvdKeyError,
)
from .parsing import (
    parse,
    Options,
)
from .path import FEPath, EPath
from .device import (
    Array,
    Field,
    FlatRegister,
    FlatRegisterUnion,
    FlatRegisterType,
    FlatStruct,
    FlatField,
    Device,
    Peripheral,
    Register,
    RegisterUnion,
    RegisterType,
    Struct,
)

import importlib.metadata

__version__ = importlib.metadata.version("svada")

__all__ = [
    # from bindings
    "Access",
    "ReadAction",
    "Endian",
    "SauAccess",
    "AddressBlockUsage",
    "Protection",
    "EnumUsage",
    "WriteAction",
    "DataType",
    "CpuName",
    "Cpu",
    "AddressBlock",
    "SauRegion",
    # from errors
    "SvdError",
    "SvdParseError",
    "SvdDefinitionError",
    "SvdMemoryError",
    "SvdPathError",
    "SvdIndexError",
    "SvdKeyError",
    # from parsing
    "parse",
    "Options",
    # from path
    "FEPath",
    "EPath",
    # from device
    "Array",
    "Field",
    "FlatRegister",
    "FlatRegisterUnion",
    "FlatRegisterType",
    "FlatStruct",
    "FlatField",
    "Device",
    "Peripheral",
    "Register",
    "RegisterUnion",
    "RegisterType",
    "Struct",
    # other
    "__version__",
]

del importlib.metadata
