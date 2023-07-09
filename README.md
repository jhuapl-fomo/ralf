
<div align="center">
<img alt="ralf_logo" src="https://github.com/jhuapl-fomo/ralf/raw/main/assets/ralf_logo_v1.png" width="200">

# ralf

[![Documentation](https://readthedocs.org/projects/ralf-jhuapl/badge/?version=latest)](https://ralf-jhuapl.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/ralf-jhuapl.svg)](https://badge.fury.io/py/ralf-jhuapl)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jhuapl-fomo/ralf/blob/main/LICENSE)
</div>

**ralf** is a Python library intended to assist developers in creating applications
that involve calls to Large Language Models (LLMs). A core concept in **ralf** is the idea of *composability*,
which allows chaining together LLM calls such that the output of one call can be
used to form the prompt of another. **ralf** makes it easy to chain together both
LLM-based and Python-based actions&mdash; enabling developers to construct complex 
information processing pipelines composed of simpler building blocks. Using LLMs
in this way can lead to more capable, robust, steerable and inspectable applications.

Currently, the **ralf** base library offers generic functionality for action chaining
(through the ``Dispatcher`` and ``Action`` classes) as well as text classificaiton
(through the ``ZeroShotClassifier`` class). Check out the other projects within
the RALF ecosystem for more specialized functionality, like dialogue management 
and information extraction.


## Quickstart Guide
This quickstart guide is intended to get you up and running with **ralf** within
a few minutes.
### Installation

We recommend creating a Conda environment before installing the package:

    conda create -n ralf python=3.10
    conda activate ralf

#### Install from PyPI

You may install **ralf** from PyPI using ``pip``:

    pip install ralf-jhuapl

#### Install from Source

Alternatively, you can build the package from source. First, clone the Github repository:

    git clone https://gitlab.jhuapl.edu/ralf/ralf

Next, install the requirements using ``pip``:
    
    cd ralf
    pip install -r requirements.txt

Then, build the package using ``flit`` and install it using ``pip``:

    flit build
    pip install .

Or if you would like an editable installation, you can instead use:

    pip install -e .

### OpenAI Configuration
**ralf** currently relies on language models provided by OpenAI. 
In order to access the models, you must store your OpenAI API key as an 
environment variable by executing the following in bash:

    echo "export OPENAI_API_KEY='yourkey'" >> ~/.bashrc
    source ~/.bashrc

### Running the Demos
To test if installation was successful, try running the demo scripts:

    cd demos
    python dispatcher_demo.py
    python classifier_demo.py

If the scripts execute successfully, you are good to go! You may want to look 
through the demo scripts to learn about some of the things **ralf** can do, or 
follow the more detailed tutorials.
## Documentation & Tutorials
The best way to get started with **ralf** is to follow the tutorials, which can be found in the [full documentation](https://ralf-jhuapl.readthedocs.io/en/latest/).

## License

This project is released under the [MIT License](https://github.com/jhuapl-fomo/ralf/blob/main/LICENSE).

Copyright &copy; 2023 The Johns Hopkins University Applied Physics Laboratory

Contact: ralf@jhuapl.edu
