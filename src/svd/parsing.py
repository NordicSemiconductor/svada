#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from typing import Union, Dict
import xml.etree.ElementTree as ET

from .peripheral import Peripheral, Register


def parse_peripheral(
    svd_path: Union[str, Path], peripheral_name: str
) -> Dict[int, Register]:
    """
    Parse an SVD for a specific peripheral and return it as a map of a memory offset and the
    register at that offset.

    :param svd_path: SVD file to use
    :param peripheral_name: Peripheral to parse

    :raise FileNotFoundError: If the SVD file does not exist.

    :return: Mapping of offset:registers for the peripheral.
    """

    svd_file = Path(svd_path)

    if not svd_file.is_file():
        raise FileNotFoundError(f"No such file: {svd_file.absolute()}")

    device = ET.parse(svd_file).getroot()
    peripheral = Peripheral(device, peripheral_name)

    return peripheral
