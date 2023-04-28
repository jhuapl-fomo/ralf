#!/usr/bin/env python

"""
A demonstration of using ZeroShotClassifier for sentiment classification.
"""

from ralf import classifier

# Define the zero shot classifer
sentiment_classifier = classifier.ZeroShotClassifier(
    completion_model_config={'model':'gpt-4'}
)

# Add classes with zero or more examples of each
sentiment_classifier.add_class("positive", examples=["this couldn't be better"])
sentiment_classifier.add_class("negative", examples=["I'm unimpressed.",
                                                     "not her best work..."])
sentiment_classifier.add_class("neutral")

# Test the classifier on some inputs
text_inputs = ["It's fine, just don't expect too much!",
               "A big miss.",
               "Marvelous performance"]

for x in text_inputs:
    pred, scores = sentiment_classifier(x)
    print(f"'{x}' -> {pred}")
