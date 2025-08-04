# Source Code for Multimodal-Cellular-Processing Plugin (Incomplete Docs)

This repository contains source code for analyzing cell cultures using microscopy and impedance data, tailored to the experimental setup at imec.

## Prerequisites

1. Python Installation

To use this application, you need to have Python installed.

- **Windows**: Open PowerShell  
- **Mac**: Open Terminal  

Check your installation by running:

```bash
python --version
```

If the command is not recognized, download and install Python from https://www.python.org

2. Install Napari and Dependencies
Install the Napari plugin along with its dependencies using the following command:

```bash
python -m pip install "napari[all]"
```
For more details, refer to the Napari installation guide

3. Install Required Packages
Run the following commands to install the specific packages used in this project:

```bash
pip install roifile
pip install cellpose==3.1.1.2
pip install matplotlib
```

4. Install the Plugin
Download or clone this repository using the buttons on the GitHub page or with Git (if you've a GitHub account and key set up):

```bash
git clone <repository-url>
```

Move into downloaded directory

```bash
cd multimodal-cellular-processing
```

Then install the plugin in editable mode:

```bash
pip install -e .
```

5. Launch the Application
Start Napari by running:

```bash
napari
```
Note: The first load may take some time. Please be patient.

<!--
# multimodal-cellular-processing

[![License MIT](https://img.shields.io/pypi/l/multimodal-cellular-processing.svg?color=green)](https://github.com/prateekgrover-in/multimodal-cellular-processing/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/multimodal-cellular-processing.svg?color=green)](https://pypi.org/project/multimodal-cellular-processing)
[![Python Version](https://img.shields.io/pypi/pyversions/multimodal-cellular-processing.svg?color=green)](https://python.org)
[![tests](https://github.com/prateekgrover-in/multimodal-cellular-processing/workflows/tests/badge.svg)](https://github.com/prateekgrover-in/multimodal-cellular-processing/actions)
[![codecov](https://codecov.io/gh/prateekgrover-in/multimodal-cellular-processing/branch/main/graph/badge.svg)](https://codecov.io/gh/prateekgrover-in/multimodal-cellular-processing)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/multimodal-cellular-processing)](https://napari-hub.org/plugins/multimodal-cellular-processing)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

Napari Plugin for Analysing Cell Cultures with Microscopy and Impedance Data tailored to experimental apparatus at IMEC.

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].


Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html


## Installation

You can install `multimodal-cellular-processing` via [pip]:

    pip install multimodal-cellular-processing



To install latest development version :

    pip install git+https://github.com/prateekgrover-in/multimodal-cellular-processing.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"multimodal-cellular-processing" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
-->

[file an issue]: https://github.com/prateekgrover-in/multimodal-cellular-processing/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
