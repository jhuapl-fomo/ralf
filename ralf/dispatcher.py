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
if len(os.environ["OPENAI_API_KEY"]) < 35: 
    openai.api_type = "azure"
    openai.api_base = ru.cfg['openai_api_base']
    openai.api_version = "2023-05-15"  # subject to change


###############
##  Helpers  ##
###############

def determine_model_type(model_name: str) -> str:
    """Determines the endpoint type given an OpenAI model name

    :param model_name: name of the OpenAI model
    :type model_name: str
    :return: endpoint type (currently either completion or chatcompletion)
    :rtype: str
    """

    if model_name.startswith(("gpt-",)):
        return "chatcompletion"
    else:
        return "completion"

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

    def start(self):
        self.start_time = time.time()
        # Also keep a string version of start time
        self.start_time_str = time.strftime(
            "%H:%M:%S %Y-%m-%S", time.localtime(self.start_time)
        )

    def finish(self, successful, result):
        self.completed = True
        self.successful = successful
        self.result = result
        self.time_taken = time.time() - self.start_time

    def to_json(self):
        return asdict(self)


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
            messages=None,
            prompt_name=None,
            prompt_template=None,
            model_config=None,
            func=None,
            input_name=None,
            context=None,
            output_name="output",
            request_timeout=ru.cfg["timeout_limit"],
    ) -> None:
        # Check that exactly one of the action specifiers was supplied
        assert sum([bool(prompt),
                    bool(messages),
                    bool(prompt_name), 
                    bool(prompt_template), 
                    bool(func)]) == 1, BAD_ACTION_MSG

        self.prompt = prompt
        self.prompt_name = prompt_name
        self.prompt_template = prompt_template
        self.func = func
        self.input_name = input_name
        self.model_config = model_config
        self.messages = messages
        self.request_timeout = request_timeout

        self.implicit = False if self.func else True
        self.output_name = output_name
        
    def __call__(
            self,
            context: Optional[dict] = None,
            prompt: Optional[str] = None,
            messages: Optional[list[dict]] = None,
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
                assert prompt or self.prompt or messages or self.messages
                prompt = prompt if prompt else self.prompt
                messages = messages if messages else self.messages

            model_config = (ru.DEFAULT_ACTION_MODEL | 
                            (self.model_config if self.model_config else {}) |
                            (model_config if model_config else {})
            )

            if report:
                report.prompt = prompt
                report.model_config = model_config

            task = determine_model_type(model_config['model'])
            completion = None

            if task == 'completion':
                if prompt:
                    response = self.submit_openai_request(
                        task=task,
                        prompt=prompt,
                        **model_config
                    )
                    if response:
                        completion = response.choices[0].text
                else:
                    raise Exception("Message prompt format cannot be used for completion")

            elif task == 'chatcompletion':
                if messages:
                    response = self.submit_openai_request(
                        task=task,
                        messages=messages,
                        **model_config
                    )
                    if response:
                        completion = response['choices'][0]['message']['content']

                else:
                    response = self.submit_openai_request(
                        task=task,
                        messages=[
                            {"role": "system", "content": ru.cfg['default_system_message']},
                            {"role": "user", "content": prompt}
                        ],
                        **model_config
                    )
                    if response:
                        completion = response['choices'][0]['message']['content']

            else:
                raise ValueError("No such OpenAI model type.")

            return {self.output_name : completion if completion else f"No response from OpenAI to {task}"}
        else:
            # Calling a Python action
            return {self.output_name : self.func(context[self.input_name])}

    def submit_openai_request(self, task, retries=10, **kwargs):

        errors = 0

        if task == "completion":
            task_func = openai.Completion.create
        elif task == "chatcompletion":
            task_func = openai.ChatCompletion.create
        else:
            raise ValueError(f"{task} is not a recognized OpenAI task")

        while retries + 1 > 0:
            try:
                response = task_func(**kwargs, request_timeout=self.request_timeout)
                return response
            except openai.error.Timeout:
                print(f"OpenAI timeout error: ", end="")
            except openai.error.APIError:
                print(f"OpenAI API error: ", end="")
            except openai.error.APIConnectionError:
                print(f"OpenAI API connection error: ", end="")
            except openai.error.InvalidRequestError:
                print(f"OpenAI invalid request error: ", end="")
            except openai.error.AuthenticationError:
                print(f"OpenAI authentication error: ", end="")
            except openai.error.PermissionError:
                print(f"OpenAI permission error: ", end="")
            except openai.error.RateLimitError:
                print(f"OpenAI rate limit error: ", end="")
            except openai.error.ServiceUnavailableError:
                print(f"OpenAI service unavailable error: ", end="")

            # exponential backoff
            delay = 2**(errors/2)
            print(f"Retrying in {delay:.2f} {'second' if int(delay) == 1 else 'seconds'} ({retries} {'retry' if retries == 1 else 'retries'} remaining)")
            time.sleep(delay)
            retries -= 1
            errors += 1


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

        # model config specified when defining the action
        inline_model_config = action.model_config if action.model_config else {}

        # look for YAML model specification 
        if action.prompt_name: # Caller is using YAML to define prompt template
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
                yaml_model_config = {}
            elif model_name in self.models:
                yaml_model_config = self.models[model_name]
            else:
                raise KeyError(
                    f"Model named '{model_name}' not in the model dictionary.")
        else:
            yaml_model_config = {}
        
        return ru.DEFAULT_ACTION_MODEL | yaml_model_config | inline_model_config

    def _run_script(self,
                    action_list: list[Action],
                    return_reports: bool = False, 
                    **kwargs
    ) -> Tuple[dict, dict]:

        """Takes in a list of Action objects and runs them in sequence

        :param action_list: a list of actions to execute in sequence
        :type action_list: list[Action]
        :param return_reports: whether or not to return a list of ActionReports 
        :type return_reports: bool, defaults to False
        :return: a tuple containing:
                    - output dictionary of the last action in the sequence
                    - dictionary with accumulated outputs of intermediate actions
        :rtype: Tuple[dict, dict]
        """
        context = kwargs
        output = {}
        reports = []

        for action in action_list:
            report = ActionReport(implicit=action.implicit)

            report.start()
            try:
                output = self._run_action(action, context, report)
                success = True
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



