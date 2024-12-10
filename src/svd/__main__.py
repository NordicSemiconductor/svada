# Copyright (c) 2024 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib.util
import json
import logging
import sys
from pathlib import Path
from textwrap import dedent

import svd
from svd.util import BuildSelector, ContentBuilder

HAS_INTELHEX = importlib.util.find_spec("intelhex") is not None
HAS_TOMLKIT = importlib.util.find_spec("tomlkit") is not None


class Format(enum.Enum):
    JSON = enum.auto()
    BIN = enum.auto()
    IHEX = enum.auto()
    TOML = enum.auto()


INPUT_FORMATS = [Format.JSON]
OUTPUT_FORMATS = [Format.JSON, Format.BIN]

if HAS_INTELHEX:
    INPUT_FORMATS.append(Format.IHEX)
    OUTPUT_FORMATS.append(Format.IHEX)

if HAS_TOMLKIT:
    INPUT_FORMATS.append(Format.TOML)
    OUTPUT_FORMATS.append(Format.TOML)


def cli() -> None:
    top = argparse.ArgumentParser(
        description=dedent(
            """\
            Collection of utility scripts for working with System View Description (SVD) files.
            """
        ),
        allow_abbrev=False,
    )
    top.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Output verbose logs. Can be given multiple times to increase the verbosity. "
            "By default only critical messages are output."
        ),
    )

    sub = top.add_subparsers(title="subcommands")

    gen = sub.add_parser(
        "content-gen",
        help="Encode and decode device content to and from various formats.",
        description=dedent(
            """\
            Encode device content from one of the supported formats and output it to another
            supported format.
            """
        ),
        allow_abbrev=False,
    )
    gen.set_defaults(_command="content-gen")

    gen_in = gen.add_argument_group("input options")
    gen_in.add_argument(
        "-I",
        "--input-format",
        choices=[f.name.lower() for f in INPUT_FORMATS],
        required=True,
        help="Input format.",
    )
    gen_in.add_argument(
        "-i",
        "--input-file",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="File to read the input from. If not given, stdin is used.",
    )

    gen_svd = gen.add_argument_group("SVD options")
    gen_svd.add_argument(
        "-s",
        "--svd-file",
        required=True,
        type=Path,
        help="Path to the device SVD file.",
    )
    gen_svd.add_argument(
        "-n",
        "--no-strict",
        action="store_true",
        help="Don't enforce constraints on register and field values based on the SVD file.",
    )
    gen_svd.add_argument(
        "--svd-parse-options",
        type=json.loads,
        help=(
            "JSON object used to override fields in the Options object to customize svada parsing "
            "behavior. Mainly intended for advanced use cases such as working around "
            "difficult SVD files. "
        ),
    )

    gen_sel = gen.add_argument_group("selection options")
    gen_sel.add_argument(
        "-p",
        "--peripheral",
        metavar="NAME",
        dest="peripherals",
        action="append",
        help="Limit output content to the given peripheral. May be given multiple times.",
    )
    gen_sel.add_argument(
        "-a",
        "--address-range",
        metavar=("START", "END"),
        nargs=2,
        type=integer,
        help="Limit output to a specific address range. Addresses can be given as hex or decimal.",
    )
    gen_sel.add_argument(
        "-c",
        "--content-status",
        choices=[c.value for c in BuildSelector.ContentStatus.__members__.values()],
        help="Limit output based on the status of the register content.",
    )

    gen_out = gen.add_argument_group("output options")
    gen_out.add_argument(
        "-O",
        "--output-format",
        choices=[f.name.lower() for f in OUTPUT_FORMATS],
        required=True,
        help="Output format.",
    )
    gen_out.add_argument(
        "-o",
        "--output-file",
        type=argparse.FileType("w", encoding="utf-8"),
        default=sys.stdout,
        help="File to write the output to. If not given, output is written to stdout.",
    )

    args = top.parse_args()

    log_level = {
        0: logging.CRITICAL,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }.get(args.verbose, logging.DEBUG)
    svd.log.setLevel(log_level)

    if not hasattr(args, "_command"):
        top.print_usage()
        sys.exit(2)

    if args._command == "content-gen":
        cmd_content_gen(args)
    else:
        top.print_usage()
        sys.exit(2)

    sys.exit(0)


def integer(val: str) -> int:
    return int(val, 0)


def cmd_content_gen(args: argparse.Namespace) -> None:
    options = svd.Options()
    if args.svd_parse_options:
        options = dataclasses.replace(options, **args.svd_parse_options)

    device = svd.parse(args.svd_file, options=options)
    builder = ContentBuilder(device, enforce_svd_constraints=not args.no_strict)

    input_format = Format[args.input_format.upper()]
    if input_format == Format.JSON:
        input_dict = json.load(args.input_file)
        builder.apply_dict(input_dict)
    elif input_format == Format.IHEX:
        from intelhex import IntelHex

        ihex = IntelHex(args.input_file)
        ihex_memory = {a: ihex[a] for a in ihex.addresses()}
        builder.apply_memory(ihex_memory)
    elif input_format == Format.TOML:
        import tomlkit

        input_dict = tomlkit.load(args.input_file).unwrap()
        builder.apply_dict(input_dict)

    selector = BuildSelector(
        peripherals=args.peripherals if args.peripherals else None,
        address_range=args.address_range if args.address_range else None,
        content_status=(
            BuildSelector.ContentStatus(args.content_status)
            if args.content_status
            else None
        ),
    )

    output_format = Format[args.output_format.upper()]
    if output_format == Format.JSON:
        output_dict = builder.build_dict(selector)
        json.dump(output_dict, args.output_file)
    elif output_format == Format.BIN:
        output_bin = builder.build_bytes(selector)
        args.output_file.buffer.write(output_bin)
    elif output_format == Format.IHEX:
        from intelhex import IntelHex

        output_ihex = IntelHex(builder.build_memory(selector))
        output_ihex.write_hex_file(args.output_file)
    elif output_format == Format.TOML:
        import tomlkit

        output_dict = builder.build_dict(selector)
        tomlkit.dump(output_dict, args.output_file)


# Entry point when running with python -m svd
if __name__ == "__main__":
    cli()
