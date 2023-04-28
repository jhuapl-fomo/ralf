import random
import numpy as np
from typing import Optional
from collections import namedtuple

from ralf.dispatcher import Action
import ralf.utils as ru

DEFAULT_CLASSIFICATION_RESPONSE_PREFIX = 'Answer: '

DEFAULT_CLASSIFICATION_TEMPLATE = (
    "Classify the following sentences. The answer choices are: {class_list}.\n"
    "{classification_examples}\n"
    "{classification_input}\n" +
    DEFAULT_CLASSIFICATION_RESPONSE_PREFIX
)

ZSClass = namedtuple("ZSClass", ['label', 'examples'])

class ZeroShotClassifier():
    def __init__(
            self,
            completion_model_config: Optional[dict] = None,
            encoder_model_name: Optional[str] = None,
            prompt_template: Optional[str] = DEFAULT_CLASSIFICATION_TEMPLATE,
            randomize_examples: Optional[bool] = True
    ) -> None:
        """A zero-shot text classifier that employs a two-step process for
        classification. First, a generative language model is used to peform
        prompt-based classification by producing completion text. The output is
        then compared to the finite label set based on similarity computed using 
        a masked langauge model.

        :param completion_model_config: Dictionary containing completion model
            configuration parameters, defaults to None which uses default
            classifier configuration defined in ralf.utils
        :type completion_model_config: Optional[dict], optional
        :param encoder_model_name: Name of the SentenceTransformer model to use
            for similarity measurement, defaults to `all-MiniLM-L6-v2`
        :type encoder_model_name: Optional[str], optional
        :param prompt_template: Prompt template used by the completion model,
            defaults to DEFAULT_CLASSIFICATION_TEMPLATE defined in classifier.py
        :type prompt_template: Optional[str], optional
        :param randomize_examples: Option to randomize the order of examples when
            creating the prompt for completion model, defaults to True
        :type randomize_examples: Optional[bool], optional
        """
        
        # Merge completion model configuration, if specified, with default
        if completion_model_config:
            self.completion_model = ru.DEFAULT_CLASSIFIER_MODEL | completion_model_config
        else:
            self.completion_model = ru.DEFAULT_CLASSIFIER_MODEL

        self.encoder_model = encoder_model_name
        self.prompt_template = prompt_template
        self.randomize_examples = randomize_examples

        self._classes = []

    def __call__(self, input: str) -> tuple[str, np.ndarray]:
        """A convenience function for prediction.

        :param input: The input string that needs to be classified
        :type input: str
        :return: The predicted class label and a numpy array of scores from 
            the similarity measurement via masked langauge model embeddings
        :rtype: tuple[str, np.ndarray]
        """

        pred, scores = self.predict(input)

        return (pred, scores)

    def predict(self, input: str) -> tuple[str, np.ndarray]:
        """Performs the two-step classification process by assmebling the full
        prompt for completion model, running the completion, then using embedding 
        similarity to choose the best guess among finite label set.

        :param input: The input string that needs to be classified
        :type input: str
        :return: The predicted class label and a numpy array of scores from 
            the similarity measurement via masked langauge model embeddings
        :rtype: tuple[str, np.ndarray]
        """
        
        # Verify that classes have been defined
        n_classes = len(self.classes())
        if n_classes < 2:
            raise ValueError(
                "Classifier must know at least 2 classes, "
                f"but {n_classes} {'was' if n_classes == 1 else 'were'} "
                "defined."
            )

        prompt = self._prepare_prompt(input)

        # TODO: remove need to specify prompt when creating an action? if you
        # just want to use it as a completion function with a full prompt. It 
        # would be more natural to do Action(), then gpt3_classify(prompt=prompt)

        # Perform LLM completion-based classification
        gpt3_classify = Action(prompt=prompt, model_config=self.completion_model)
        x = gpt3_classify()

        completion_pred = x['output']

        # Project the LLM completion onto the class label set
        scores = ru.similarity(
            completion_pred,
            self.classes(),
            model=self.encoder_model
        )
        pred = self.classes()[np.argmax(scores)]

        return (pred, scores)

    def classes(self) -> list[str]:
        """Returns the labels for all classes currently known to the classifier.

        :return: A list of class labels
        :rtype: list[str]
        """

        return [class_.label for class_ in self._classes]

    def add_class(
            self,
            class_label: str,
            examples: Optional[list[str]] = None
    ) -> None:
        """Adds a class to the set of classes known to the classifier.

        :param class_label: The string to associate with this class
        :type class_label: str
        :param examples: An optional list of examples that fall into this class,
            which will be used during prompt formation for the completion model, 
            defaults to None
        :type examples: Optional[list[str]], optional
        """

        examples = examples if examples else []
        self._classes.append(ZSClass(class_label, examples))

    def _prepare_prompt(self, input: str) -> str:
        """Given a new text input to be classified, prepares the full prompt 
        for the completion model to predict the class label. The prompt 
        incorporates all class labels and examples known to the classifier object.

        :param input: The new text input to be classified
        :type input: str
        :return: The full prompt for the completion model to emit a class label
            prediction
        :rtype: str
        """

        # Create the examples
        example_pairs = []
        for class_ in self._classes:
            for example in class_.examples:
                example_pairs.append(
                    f"{example}\n"
                    f"{DEFAULT_CLASSIFICATION_RESPONSE_PREFIX}{class_.label}")

        # Randomize the order if option selected
        if self.randomize_examples:
            random.shuffle(example_pairs)

        # Make the text that contains the examples
        example_text = '\n'.join(example_pairs)

        # A comma separated string of class names
        class_list = ', '.join(self.classes())

        prompt = self.prompt_template.format(
            class_list=class_list,
            classification_examples=example_text,
            classification_input=input
        )

        return prompt
