#!/usr/bin/env python

"""
A simple chat interface that uses OpenAI Chat Completion.
"""

import os
import argparse
from collections import deque

from termcolor import colored
from art import tprint
import openai

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    print(
        "You must save your OpenAI API key as an environment "
        "variable. Please see README for details.\n"
    )
    raise SystemExit(1)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--n_saved_turns', type=int, default=5)
    args = parser.parse_args()

    chat_messages = deque(
        [{"role": "system", "content": "You are a helpful assistant."}], 
        maxlen=args.n_saved_turns
    )
    
    tprint("SimpleChat")
    print("Welcome to SimpleChat!\n")
    print("You are chatting with: " + colored(args.model, 'blue'))
    print(f"Turns saved in conversation history: {args.n_saved_turns}")
    print('\n')

    while True:
        human_utterance = input(colored("Human: ", attrs=['bold']))

        if human_utterance in ["exit"]:
            print(colored("Goodbye!", 'green'))
            break

        # TODO: ralf-ify this?
        chat_messages.append({"role": "user", "content": human_utterance})
        response = openai.ChatCompletion.create(
            model=args.model,
            messages=list(chat_messages)
        )

        ai_utterance = response['choices'][0]['message']['content']
        print(colored("AI:", 'blue', attrs=["bold"]) + ' ' + colored(ai_utterance, 'blue'))
        chat_messages.append({"role": "assistant", "content": ai_utterance})

if __name__ == "__main__":
    main()