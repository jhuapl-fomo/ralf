Quickstart Guide
================
This quickstart guide is intended to get you up and running with **ralf** within
a few minutes.


Installation
------------
First, clone the Gitlab project::

    git clone https://gitlab.jhuapl.edu/ralf/ralf

Next, install the requirements using ``pip``::
   
    pip install -r requirements.txt

Then, build the package using ``flit`` and install it using ``pip``::

    flit build
    pip install .

Or if you would like an editable installation, you can instead use::

    pip install -e .

OpenAI Configuration
--------------------

**ralf** currently relies on language models provided by OpenAI. 
In order to access the models, you must store your OpenAI API key as an 
environment variable by executing the following in bash::

    echo "export OPENAI_API_KEY='yourkey'" >> ~/.bashrc
    source ~/.bashrc

Running the Demos
-----------------

To test if installation was successful, try running the demo scripts::

    python demos/dispatcher_demo.py
    python demos/classifier_demo.py

If the scripts execute successfully, you are good to go! You may want to look 
through the demo scripts to learn about some of the things **ralf** can do, or 
follow the more detailed tutorials.

