#############
## Imports ##
#############

# Standard lib
import re
import yaml
from typing import Optional

# ML/etc.
import numpy as np
from sentence_transformers import SentenceTransformer


################
##  Defaults  ##
################

DEFAULT_ENCODER = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

DEFAULT_ACTION_MODEL = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 256,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

DEFAULT_CLASSIFIER_MODEL = {
    "model": "text-davinci-003",
    "temperature": 0.0,
    "max_tokens": 100,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": ['.'],
}


#########################
##      General        ##
##  Utility Functions  ##
#########################

def load_yaml(yaml_path: str) -> dict:
    """Loads the YAML file into a dictionary

    :param yaml_path: path to YAML file
    :type yaml_path: str
    :return: dictionary with data from YAML file
    :rtype: dict
    """
    with open(yaml_path, 'r') as f:
        try:
            yaml_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Unable to load YAML file: {e}")

    return yaml_dict


####################
## String Methods ##
####################

def format_safe(template: str, context: dict) -> str:
    """
    A modified version of string formatting that allows extra entries to exist
    in the provided dict for which no corresponding placeholders exist in the
    template string.

    :param template: the text that contains placeholders to be filled in
    :type template: str
    :param context: a dictionary containing the context items
    :type context: dict
    :return: the full text with filled in context items
    :rtype: str
    """
    required_items = re.findall(r'\{(.+?)\}', template)
    restricted_context = {
        key: context[key]
        for key
        in context.keys()
        if key in required_items
    }

    return template.format(**restricted_context)


def _strip_quotes(string):
    if string.startswith('"') or string.startswith("'"):
        string = string[1:]

    if string.endswith('"') or string.endswith("'"):
        string = string[:-1]

    return string


def _normalize_space(string):
    return ' '.join(string.strip().split())


def _normalize(string):
    return _strip_quotes(
        _normalize_space(string)
    )


def str_to_dict_safe(world_state_str):
    world_state_dict = {}

    lines = [
        elt
        for line
        in world_state_str.split(',')
        for elt in line.split('\n')
    ]

    for line in lines:
        if ':' in line:
            key, value = line.split(':')
            world_state_dict[_normalize(key)] = _normalize(value)

    return world_state_dict


###################
##  Projections  ##
###################

def similarity(target: str,
               references: list[str],
               model: Optional[str] = None,
               normalize: Optional[bool] = True) -> np.ndarray:
    """Computes similarity scores between target string and references using
    SentenceTransformer embeddings.

    :param target: The target string
    :type target: str
    :param references: A list of reference strings
    :type references: list[str]
    :param model: Name of the SentenceTransformer model to use, defaults to
        None, which uses `all-MiniLM-L6-v2`
    :type model: Optional[dict], optional
    :param normalize: Option to perform softmax normalization of logits,
        defaults to True
    :type normalize: Optional[bool], optional
    :return: A numpy array containing similarity scores to each reference string
    :rtype: np.ndarray
    """
    encoder = SentenceTransformer(model) if model else DEFAULT_ENCODER

    # Encode the target and reference items
    target_emb = encoder.encode(target)
    ref_embs = encoder.encode(references)

    # Compute similarities
    logits = ref_embs @ target_emb

    if normalize:
        return np.exp(logits)/sum(np.exp(logits))
    else:
        return logits


################
## Exceptions ##
################

class MissingContextException(Exception):
    def __init__(self, template: str, context: dict):

        # Find context items needed in prompt template
        required_items = re.findall(r'\{(.+?)\}', template)

        provided_items = list(context.keys())

        error_msg = \
            f"Prompt expected context items {required_items} " \
            f"and {provided_items} were provided."

        super().__init__(error_msg)
