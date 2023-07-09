# Copyright (c) 2023 The Johns Hopkins University Applied Physics Laboratory

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import yaml
import time
import inspect
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, Tuple

import openai

import ralf.utils as ru

############################
##  OpenAI Configuration  ##
############################

# Load the OpenAI API key
try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    print(
        "You must save your OpenAI API key as an environment "
        "variable. Please see README for details.\n"
    )
    raise SystemExit(1)


###############
##  Globals  ##
###############

openai_model_types = {
    "gpt-4"             : "chatcompletion",
    "gpt-4-0314"        : "chatcompletion",
    "gpt-4-32k"         : "chatcompletion",
    "gpt-4-32k-0314"    : "chatcompletion",
    "gpt-3.5-turbo"     : "chatcompletion",
    "gpt-3.5-turbo-0301": "chatcompletion",
    "text-davinci-003"  : "completion",
    "text-davinci-002"  : "completion",
    "text-curie-001"    : "completion",
    "text-babbage-001"  : "completion",
    "text-ada-001"      : "completion",
    "davinci"           : "completion",
    "curie"             : "completion",
    "babbage"           : "completion",
    "ada"               : "completion",
    
}

# default system message for ChatCompletion
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant." #TODO: make configurable

###############
##  Helpers  ##
###############

def clean_response(response: str) -> str:
    """
    Given a response (prompt completion) from a
    foundation model (as, e.g., OpenAI's GPT-3), clean
    the response -- for now, by stripping leading and
    trailing space.

    :param response: foundation model completion
    :type response: str

    :return: the cleaned foundation model completion
    :rtype: str
    """
    return response.strip()

#########################
##  The Action Report  ##
##       Class         ##
#########################

@dataclass
class ActionReport:
    implicit: bool
    completed: bool = False
    parent_script: str = None
    source_code: str = None
    model_config: dict = None
    prompt: str = None
    start_time: float = None
    start_time_str: str = None
    time_taken: float = None
    successful: bool = None
    result: dict = None

    def start(self) -> None:
        """Begins timing of action execution"""
        self.start_time = time.time()
        # Also keep a string version of start time
        self.start_time_str = time.strftime(
            "%H:%M:%S %Y-%m-%S", time.localtime(self.start_time)
        )

    def finish(self, successful: bool, result: dict) -> None:
        """Completes timing of action execution and saves completion info

        :param successful: whether or not action completed successfully
        :type successful: bool
        :param result: output dictionary of the action
        :type result: dict
        """
        self.completed = True
        self.successful = successful
        self.result = result
        self.time_taken = time.time() - self.start_time


###################
##  The Action   ##
##     Class     ##
###################

BAD_ACTION_MSG = "When creating an action, you must specify"
"exactly one of the following: 'prompt', 'prompt_name', 'prompt_template', or 'func'"
class Action:
    def __init__(
            self,
            prompt=None,
            prompt_name=None,
            prompt_template=None,
            model_config=None,
            func=None,
            input_name=None,
            context=None,
            output_name="output"
    ) -> None:
        # Check that exactly one of the action specifiers was supplied
        assert sum([bool(prompt),
                    bool(prompt_name), 
                    bool(prompt_template), 
                    bool(func)]) == 1, BAD_ACTION_MSG

        self.prompt = prompt
        self.prompt_name = prompt_name
        self.prompt_template = prompt_template
        self.func = func
        self.input_name = input_name
        self.model_config = model_config

        self.implicit = False if self.func else True
        self.output_name = output_name
        
    def __call__(
            self,
            context: Optional[dict] = None,
            prompt: Optional[str] = None,
            model_config: Optional[dict] = None,
            report: ActionReport = None
    ) -> dict:
        """Executes an action

        :param context: context dictionary for filling prompt template, defaults to None
        :type context: Optional[dict], optional
        :param prompt: a fully formed prompt for LLM-based actions, defaults to None
        :type prompt: Optional[str], optional
        :param model_config: LLM model configuration dictionary, defaults to None
        :type model_config: Optional[dict], optional
        :return: dictionary containing named outputs of the action
        :rtype: dict
        """
        if self.implicit:
            # Calling an LLM-based action

            if context and self.prompt_template:
                # User provided context when calling action, and must have
                # provided a prompt template when defining the action

                prompt = ru.format_safe(self.prompt_template, context)
            else:
                # Use the prompt given when calling, or if one isn't given,
                # use the one specified when defining the action
                assert prompt or self.prompt
                prompt = prompt if prompt else self.prompt

            model_config = (ru.DEFAULT_ACTION_MODEL | 
                            (self.model_config if self.model_config else {}) |
                            (model_config if model_config else {})
            )

            if report:
                report.prompt = prompt
                report.model_config = model_config

            if openai_model_types[model_config['model']] == 'completion':
                response = openai.Completion.create(
                    prompt=prompt,
                    **model_config
                )
                completion = response.choices[0].text

            elif openai_model_types[model_config['model']] == 'chatcompletion':
                response = openai.ChatCompletion.create(
                    messages=[
                        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt}
                    ],
                    **model_config
                )
                completion = response['choices'][0]['message']['content']

            else:
                raise ValueError("No such OpenAI model type.")

            return {self.output_name : completion}
        else:
            # Calling a Python action
            return {self.output_name : self.func(context[self.input_name])}


#######################
##  The Dispatcher   ##
##       Class       ##
#######################

class ActionDispatcher:
    def __init__(self, dir: Optional[str] = None) -> None:
        self.dir = dir

        if self.dir:
            # Load prompts from YAML file
            with open(str(Path(dir)/'prompts.yml'), 'r') as f:
                self.prompts = yaml.load(f, Loader=yaml.FullLoader)

            # Load model configurations from YAML file
            with open(str(Path(dir)/'models.yml'), 'r') as f:
                self.models = yaml.load(f, Loader=yaml.FullLoader)

    def __call__(self, input: Union[Action, list[Action]], **kwargs) -> dict:
        """Convenience for invoking `execute`.

        :param input: an action or a list of actions to be executed in sequence
        :type input: Union[Action, list[Action]]
        :return: a dictionary containing named outputs of the last action
        :rtype: dict
        """

        return self.execute(input, **kwargs)
        
    def execute(self, input: Union[Action, list[Action]], **kwargs) -> dict:
        """Public interface for executing either a single action or a sequence 
        of actions provided as a list

        :param input: an action or a list of actions to be executed in sequence
        :type input: Union[Action, list[Action]]
        :return: a dictionary containing named outputs of the last action
        :rtype: dict
        """
        if isinstance(input, list):
            return self._run_script(input, **kwargs)
        elif isinstance(input, Action):
            return self._run_action(input, **kwargs)
        else:
            raise TypeError(
                "Must pass in an Action or List[Action]; "
                f"instead, got {type(input)}."
            )

    def _prepare_prompt(self, action, context):
        """
        Given an action object, find its prompt template, then use a context 
        dictionary to fill in the missing values in the prompt template, yielding
        a ready prompt.

        :param action: action object (implicit) whose prompt needs to be prepared
        :type action: str
        :param context: the current context dictionary, which should contain
        a subset of keys matching the template values in the prompt template
        :type context: dict[str, str]

        :raises Exception: "No such prompt" in the case of a prompt_name that does
        not exist in the prompt config file

        :return: the fully filled-in prompt, ready for completion by LLM
        :rtype: str
        """

        assert action.prompt or action.prompt_template or action.prompt_name

        if action.prompt:
            # Caller directly specified full prompt
            return action.prompt
        elif action.prompt_template:
            # Caller directly specified prompt template
            prompt_template = action.prompt_template
        else:
            # Prompt template specified in YAML
            if action.prompt_name in self.prompts:
                prompt_template = self.prompts[action.prompt_name]['prompt_template']
            else:
                raise KeyError(f"No such prompt: '{action.prompt_name}'")
        
        return ru.format_safe(prompt_template, context)
    
    def _get_model_config(self, action: Action) -> dict:
        """Returns the model configuration corresponding to a given LLM-based action.

        :param action: the action for which we want to obtain a model config
        :type action: Action
        :return: the model configuration dictionary for executing this action
        :rtype: dict
        """
        
        assert action.prompt or action.prompt_template or action.prompt_name

        if action.prompt_template:
            # Caller directly supplied prompt template and possibly model config
            model_config = action.model_config if action.model_config else {}
        else:
            # Get the model name that was specified in prompts YAML
            if action.prompt_name in self.prompts:
                prompt_dict = self.prompts[action.prompt_name]
                model_name = prompt_dict['model'] if 'model' in prompt_dict else None
            else:
                raise KeyError(f"No such prompt: {action.prompt_name}")

            # TODO: maybe allow using a model name from YAML even if caller
            # directly specified a prompt template via function argument

            # Find the configuration for the model based on its name
            if model_name is None: # no model supplied in prompts YAML
                model_config = {}
            elif model_name in self.models:
                model_config = self.models[model_name]
            else:
                raise KeyError(
                    f"Model named '{model_name}' not in the model dictionary.")
        
        return ru.DEFAULT_ACTION_MODEL | model_config

    def _run_script(self,
                    action_list: list[Action],
                    return_reports: bool = False, 
                    **kwargs
    ) -> Tuple:

        """Takes in a list of Action objects and runs them in sequence

        :param action_list: a list of actions to execute in sequence
        :type action_list: list[Action]
        :param return_reports: whether or not to return a list of ActionReports 
        :type return_reports: bool, defaults to False
        :return: a tuple containing:
                    - output dictionary of the last action in the sequence
                    - dictionary with accumulated outputs of intermediate actions
                    - a list of ActionReport objects if return_reports is True
        :rtype: Tuple[dict, dict] or Tuple[dict, dict, list] if return_reports
        """
        context = kwargs
        output = {}
        reports = []

        for action in action_list:
            report = ActionReport(implicit=action.implicit)

            report.start()
            try:
                output = self._run_action(action, context, report)
                success= True
            except Exception as e:
                output = {'error_msg' : str(e)}
                success = False
                raise e
            finally:
                report.finish(successful=success, result=output)
                reports.append(report)

            context = context | output

        if return_reports:
            # TODO: decide whether to just replace context with reports in return_reports
            return output, context, reports
        else:
            return output, context

    def _run_action(self, 
                    action: Action,
                    context: Optional[dict] = None,
                    report: ActionReport = None,
                    **kwargs
    ) -> dict:
        """Takes in a single Action object and runs it.

        :param action: the action to be executed
        :type action: Action
        :param context: a dictionary of context items, defaults to None
        :type context: Optional[dict], optional
        :return: the output dictionary returned from executing the action
        :rtype: dict
        """
        # TODO: modify this to take context as kwargs
        context = context if context else {}

        if action.implicit:
            prompt = self._prepare_prompt(action, context)
            model_config = self._get_model_config(action)
        else:
            prompt = None
            model_config = None

        output = action(
            context=context,
            prompt=prompt,
            model_config=model_config,
            report=report
        )

        return output



