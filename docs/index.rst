.. ralf documentation master file, created by
   sphinx-quickstart on Tue Mar 21 09:42:43 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome!
================================
Welcome to the documentation for **ralf**, the base library within the larger
`RALF Ecosystem <https://gitlab.jhuapl.edu/ralf>`_. Here you will find documentation
and tutorials that will help you get started with building applications using Large
Language Models (LLMs).

.. note::
   This project is still under active development. If you find a bug or want to 
   suggest a change, please create a Github issue or email ralf@jhuapl.edu. 
   Also, if you’re interested in contributing to the project, let us know! We are 
   eager to see the project grow if the community finds it useful.

About
=====


**ralf** is a Python library intended to assist developers in creating applications
that involve calls to LLMs. A core concept in **ralf** is the idea of *composability*,
which allows chaining together LLM calls such that the output of one call can be
used to form the prompt of another. **ralf** makes it easy to chain together both
LLM-based and Python-based actions-- enabling developers to construct complex 
information processing pipelines composed of simpler building blocks. Using LLMs
in this way can lead to more capable, robust, steerable and inspectable applications.

Currently, the **ralf** base library offers generic functionality for action chaining
(through the ``Dispatcher`` and ``Action`` classes`) as well as text classificaiton
(through the ``ZeroShotClassifier`` class). Check out the other projects within
the RALF ecosystem for more specialized functionality, like dialogue management 
and information extraction.


.. .. toctree::
..    :maxdepth: 2
..    :caption: Contents:

Contents
========

Introduction
------------

   | :doc:`Quickstart Guide <introduction/quickstart>`
   
Tutorials
---------

   | :doc:`Executing Actions <tutorials/executing_actions>`
   | :doc:`Classification <tutorials/classification>`

API
---

   :doc:`API <api/ralf>`

.. Hidden TOCs

.. toctree::
  :caption: Introduction
  :maxdepth: 2
  :hidden:

  introduction/quickstart

.. toctree::
  :caption: Tutorials
  :maxdepth: 2
  :hidden:

  tutorials/executing_actions
  tutorials/classification

.. toctree::
  :caption: API
  :maxdepth: 2
  :hidden:

  api/ralf

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
