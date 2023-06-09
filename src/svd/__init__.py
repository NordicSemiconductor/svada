#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from .parsing import (
    parse_peripheral,
)
from .peripheral import (
    Peripheral,
    RegisterElement,
    Register,
    Field,
    get_memory_map,
    get_bit_range,
    get_register_elements,
)
from .util import (
    strip_prefixes_suffixes,
    to_int,
)

import importlib.metadata

try:
    __version__ = importlib.metadata.version("svada")
except importlib.metadata.PackageNotFoundError:
    # Package is not installed
    import setuptools_scm
    __version__ = setuptools_scm.get_version(root="../..", relative_to=__file__)
