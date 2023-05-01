
<div align="center">
<img alt="ralf_logo" src="assets/ralf_logo_v1.png" width="200">

# ralf

[![Documentation](https://readthedocs.org/projects/ralf-jhuapl/badge/?version=latest)](https://ralf-jhuapl.readthedocs.io/en/latest/)
</div>

**ralf** is a Python library intended to assist developers in creating applications
that involve calls to Large Language Models (LLMs). A core concept in **ralf** is the idea of *composability*,
which allows chaining together LLM calls such that the output of one call can be
used to form the prompt of another. **ralf** makes it easy to chain together both
LLM-based and Python-based actions-- enabling developers to construct complex 
information processing pipelines composed of simpler building blocks. Using LLMs
in this way can lead to more capable, robust, steerable and inspectable applications.

Currently, the **ralf** base library offers generic functionality for action chaining
(through the ``Dispatcher`` and ``Action`` classes) as well as text classificaiton
(through the ``ZeroShotClassifier`` class). Check out the other projects within
the RALF ecosystem for more specialized functionality, like dialogue management 
and information extraction.



## Getting Started

First, clone the Github repository:

    git clone https://gitlab.jhuapl.edu/ralf/ralf

Next, install the requirements using ``pip``:
   
    pip install -r requirements.txt

Then, build the package using ``flit`` and install it using ``pip``:

    flit build
    pip install .

Or if you would like an editable installation, you can instead use:

    pip install -e .

## Documentation & Tutorials
The best way to get started with **ralf** is to follow the tutorials in the [TODO] [Documentation site](https://google.com). If you're eager to get started and want to skip the tutorials, you might instead consider checking out the `dispatcher_demo.py` and `classifier_demo.py` files in the `demo/` directory.

## License
[**TODO**]
