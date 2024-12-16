# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Latest](https://github.com/NordicSemiconductor/svada)

## [v2.2.0](https://github.com/NordicSemiconductor/svada/tree/v2.2.0)

### Added
* Added a CLI script called ``svada`` that can be used to generate register content in various formats.
  See ``svada --help`` for more information.
* Added a helper for populating and outputting peripheral memory in ``svd.util``.
* Added public APIs to ``__all__`` to silence linter warnings about private API usage.

### Updated
* Switched to the hatchling build system.
* Updated the default options so that ``parent_relative_cluster_offset=True``.
  While not compliant with the literal reading of the specification, this seems to be the intended way to parse cluster offsets.
* Removed automatic "trailing zero adjustment" for field content.

### Fixed
* Fixed a off-by-one error in ``svd.Register.set_content``, which caused incorrect bounds checks for values that were a power of 2.

## [v2.1.0](https://github.com/NordicSemiconductor/svada/tree/v2.1.0)

### Changed

* NCSDK-29781: Updated package dependencies to enable support for Python 3.13.
  Consequently, removed support for Python 3.9 and below.

## [v2.0.2](https://github.com/NordicSemiconductor/svada/tree/v2.0.2)

### Fixed
* NCSDK-24532: SVD files are now opened in binary mode instead of defaulting to the locale preferred encoding.
  This avoids issues that occur when the locale encoding doesn't match the file encoding.

## [v2.0.1](https://github.com/NordicSemiconductor/svada/tree/v2.0.1)

### Fixed
* NCSDK-24490: Specified minimum version requirements for all dependencies.

## [v2.0.0](https://github.com/NordicSemiconductor/svada/tree/v2.0.0)

This version is a large expansion of svada to support a lot of new features.
`svada` now supports parsing the whole SVD device in a way mostly following the SVD specification, and more of the content in the SVD file is exposed.

### Added
* A `Device` class that represents the complete SVD device.
* The Path-like classes `EPath`, `FEPath` for referencing registers by their path relative to the parent peripheral.
  The structure of the paths are similar to how registers would be referenced in a C representation of the peripheral.
* Custom error classes with SVD-specific semantics.
* Type hints.

### Changed
* The top level parsing function is now called `parse()` instead of `parse_peripheral()`, and returns a `Device` instance instead of a `Peripheral`.
  The `Device` object can be used to access the individual peripherals in the SVD file.
* Reworked the `Register` and `Field` classes to better reflect the SVD structure.
  Register-level elements now mirror the nested structure in the SVD description.
  The peripheral can be accessed in either a regular or "flat" mode, and there are separate register-level classes for each of these modes.
  In the regular mode, registers use the dimensions defined on SVD elements to expand registers into an array representation.
  Regular registers also have memory content associated with them.
  Flat mode registers on the other hand correspond directly to the element layout in the SVD file, and have no content associated with them.
* Updated the peripheral, register and field APIs to fit the reworked class hierarchy and expose more SVD information.
* Registers are now referenced using the new register path classes instead of a flattened lowercase path.
* Peripheral memory is now stored using `numpy` arrays for a correct and reasonably efficient binary representation.
  Memory in a peripheral can be accessed through individual (regular) registers and fields, and also iterated over in full from the peripheral level.
* Refactored the Peripheral implementation to access the SVD properties through an intermediary "binding" layer based on `lxml`.

## [v1.0.1](https://github.com/nordicsemiconductor/svada/tree/v1.0.1)

### Fixed

* Fixed an issue with raw register values not accounting for reset values of bits without a field (#3)
* Fixed an issue with nested register numbering (#4)

## [v1.0.0](https://github.com/nordicsemiconductor/svada/tree/v1.0.0)

* Initial release.
