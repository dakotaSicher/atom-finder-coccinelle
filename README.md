# Atom Finder Coccinelle

This tool incorporates a series of Coccinelle semantic patches designed to detect 14 out of the 15 [Atoms of Confusion](https://atomsofconfusion.com/data.html)
as identified and empirically validated in our prior work. These atoms are common syntactic features in C code that 
can lead to misunderstanding and errors. It's important to note that the "Preprocessor in Statement" atom cannot be 
detected using Coccinelle.

## Features
- **Coccinelle Patches**: Targets 14 different atoms of confusion.
- **CLI Wrapper**: Simplifies the process of applying these patches to a C codebase.

## Prerequisites

Before installing this tool, you must install Coccinelle. Please be aware that Coccinelle is designed to run on Linux 
and cannot be installed natively on Windows (but can be installed on MacOS).

### Installing Coccinelle

The supported version of Coccinelle is specified in `config.json` (currently version 1.3). To install Coccinelle, 
follow the instructions from the [official installation guide](https://github.com/coccinelle/coccinelle/blob/master/install.txt).

## Installation

Once Coccinelle is installed, you can install this tool. To install it in an editable mode, which is useful for 
development, run the following command from the root of the project:
```bash
pip install -e .
```

## Usage

To run the patches, use the command `aoc-cocci`. By default, all patches will be applied, but you can also specify a 
particular patch to run:

```bash
aoc-cocci dir-with-c-files --patch reversed_subscript --output-dir output
```

This command specifies that only the reversed subscript patch should be applied and the results should be saved to 
the output directory.

To run the linux fixes tool use the command `aoc-linux-fixes`. This tool is used to examine bug fix commits to the linux kernel.

```bash
aoc-linux-fixes dir-with-linux-repo --output-dir output_dir --history_length "1 year" --cpus 4
```

This command specifies to run atoms patches on linux kernel commits going back 1 year from now using 4 worker processes.
If history length is not specified, all commits going back to the introduction of specific commit messages for bug fixes.
If cpus is set to 0, it will use the system's cpu count, default is 4.