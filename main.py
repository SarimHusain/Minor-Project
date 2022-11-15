#!python
from sys import argv
from datasets import load_dataset

from config import defaults, makeconfig, print_help
from training import train
from testing import test
from validation import validate


if __name__ == "__main__":
    # Rudimentary argument parser for command line arguments.
    # Lets us have otherwise complicated behaviour, like chaining commands.
    actions = list()
    params = list()
    for arg in argv[1:]:
        if arg == "help":
            print_help()
        elif arg.startswith("--"):
            params.append(arg[2:])
        else:
            actions.append(arg)

    # Build default config params
    makeconfig(params)

    model = None
    trainer = None
    token = None
    
    for command in actions:
        if command == "train":
            dataset = load_dataset(defaults["dataset"])
            model, trainer, token = train(dataset)
        
        elif command == "validate":
            assert all([model, trainer, token]), \
                "Model, Trainer and Tokenizer need to be set by train before validation"
            validate(model, trainer, token)
        
        elif command == "test":
            test()
        
        else:
            print(f"Unknown command `{command}`")
            print_help()
    exit(0)