# Svada - SVD Parsing for Python

`svada` is a general-purpose parser for quickly and efficiently parsing CMSIS SVD files into concise objects for use in various applications.

Contents:

* [Building and installing from source]
* [Contributing]
	- [Code style]
	- [Virtual environments]

## Building and installing from source
[Building and installing from source]: #building-and-installing-from-source

Make sure that your environment is using the latest `build` package from PyPA,
then invoke it to build the package from the `setup.cfg` file:

```
python3 -m pip install --user -U build
python3 -m build
```

This will generate both `tar.gz` and `.whl` artifacts for the current version, either of which can be used for installing from the command line:

```
python3 -m pip install dist/svada-X.Y.Z-py3-none-any.whl
```

## Contributing
[Contributing]: #contributing

### Code style
[Code style]: #code-style

This project uses [black] and [pylint] for aligning source code with PEP standards and keeping to a single, common format across the codebase.
These can be installed via:

```
python3 -m pip install -r scripts/requirements-test.txt
```

To more easily facilitate this, the repository contains the `scripts/pre-commit` hook script that can be placed in your local `.git/hooks` folder.
This script will block committed code that fails formatting and linting checks by these programs, ensuring that new contributions align to the policies before being pushed and caught by CI.

The hook script has options that can be added to `.gitconfig` to further customize the behavior.
For example, a specific `.pylintrc` file can be used for linting by setting this option locally:

```
# ~/.gitconfig
[hooks]
	pylintrc=/home/username/.pylintrc
```

It is required to do this for all developers and contributors of this project to keep source code aligned across files.

[black]: https://pypi.org/project/black/
[pylint]: https://pypi.org/project/pylint/

### Virtual environments
[Virtual environments]: #virtual-environments

During development, it is strongly recommended to have this package installed in a separate virtual environment for easily testing local changes without continuous installation and uninstallation.
You can find the official documentation about virtual environments in Python and how to use them on their [official venv tutorial page][venv-tutorial].

To use the tool in a virtual environment, do the following:

1. Create and activate a virtual environment:

```
python3 -m venv my_venv
```

Alternatively, Python3's [virtualenv] package can also create the same virtual environments, and offers some extra features from the built-in [venv].

2. Create a dynamic installation of the package, running the setup script with the `develop` argument from the repository root:

```
python3 setup.py develop
```

This installs the package using always the latest changes of its source files, including when invoked through its command-line executables.
`setup.py` automatically grabs build and metadata information from `setup.cfg` and `pyproject.toml`.

To uninstall the dynamic package version installed with `develop`, re-run the command with the `--uninstall` or `-u` option.

[venv-tutorial]: https://docs.python.org/3/tutorial/venv.html
[virtualenv]: https://virtualenv.pypa.io/en/latest/
[venv]: https://docs.python.org/3/library/venv.html
