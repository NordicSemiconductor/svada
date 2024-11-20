#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

import importlib.metadata

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

__version__ = importlib.metadata.version("svada")

__all__ = [
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
    "SvdError",
    "SvdParseError",
    "SvdDefinitionError",
    "SvdMemoryError",
    "SvdPathError",
    "SvdIndexError",
    "SvdKeyError",
    "parse",
    "Options",
    "FEPath",
    "EPath",
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
]

del importlib.metadata
