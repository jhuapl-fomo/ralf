#!/usr/bin/env python

"""
A demonstration of using ActionDispatcher to execute a sequence of actions.
"""
import re

from ralf.dispatcher import Action, ActionDispatcher

########################
# DEFINING THE ACTIONS #
########################

print(
    "Define an implicit action for fruit enumeration, "
    "using YAML-based specification"
)
fruit_counter = Action(
    prompt_name='enumerate_fruits',
    output_name='fruit_counts'
)

print("Define an explicit action for fruit count summing ...")
def sum_counts(text):
    text_counts = re.findall('[0-9]+', text)
    fruit_total = sum([int(x) for x in text_counts])

    return fruit_total

print("Construct the action ...")
count_adder = Action(func=sum_counts,
                     input_name='fruit_counts',
                     output_name='fruit_total')

print("Directly define an implicit action for commenting on the total fruit count ...")
count_commentary = Action(
    prompt_template='Q:Is {fruit_total} fruits a lot?\nA:',
    output_name='fruit_commentary',
    model_config={'max_tokens':64}
)

#########################################
# CREATING THE DISPATCHER AND EXECUTING #
#########################################

print("Define the dispatcher by pointing it to our yaml files ...")
ad = ActionDispatcher(dir='ralf_data')

print("Create the script which is just a list of Actions ...")
fruit_script = [fruit_counter, count_adder, count_commentary]

print("Run an input through a sequence of actions ...")
input_text = "sally has 8 apples, six pomegranates, 4 bananas and 2 carrots"
result = ad.execute(fruit_script, utterance=input_text)

print(result)
