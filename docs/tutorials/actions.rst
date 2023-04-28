Executing Actions
=================

At the heart of **ralf** is the ``Action`` class. An ``Action`` object represents
a basic data processing block and can take one of two forms: (1) a call to a 
Large Language Model (LLM) or (2) a Python function.



In this set of tutorials, we will see how we can use

.. todo::
    Add a diagram depicting dispatcher and actions

Tutorial 1: Simple Actions
----------------------------

We will now look at how to use ``Action`` to perform a simple sentence completion
using a call to an LLM. 

First, we import the ``Action`` class from **ralf**. Then we create an ``Action`` 
object, at which point we can specify the prompt we want the LLM to complete.
Internally, the class will understand that this action involves an LLM call 
(as opposed to a Python function).

We can then execute this individual action as if it were a function, as shown:

.. code-block::

    from ralf.dispatcher import Action

    aussie_capital = Action(prompt="The capital of Australia is")
    aussie_capital()

Output:

.. code-block::

    {'output': ' Canberra.'}

Note that in the above example, we did not specify


Tutorial 2: Action Sequences
----------------------------


Tutorial 3: Using YAML Files
----------------------------


Tutorial 4:
-----------