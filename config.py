defaults = {
    "model_name": "distilbert-finetuned-squad",
    "model_path": "./trained/",

    "dryrun": False,

    # Dataset
    "dataset": "squad_v2",
    "batch_size": 16,
    "max_length": 384,
    "doc_stride": 128,

    # Validation
    "n_best_size": 20, 
    "max_answer_length": 30,

    # Training parameters
    "lr": 2e-5,
    "epochs": 3,
    "device": "cpu",
}

def makeconfig(argv: list[str]):
    global defaults
    config_dict = defaults.copy()
    for arg in argv:
        if not arg: continue

        value_index = arg.find("=")
        name, value = arg, True
        if value_index != -1:
            name = arg[0:value_index]
            value = arg[value_index+1:]
        
        assert name in config_dict.keys(), "Cannot add configurations"
        config_dict[name] = type(config_dict[name])(value)

    if config_dict["dryrun"]:
        config_dict = { **config_dict,
            "batch_size": 16,
            "epochs": 1,
        }

    defaults = config_dict

def print_help():
    nspaces = lambda n: ''.join([' ' for _ in range(n)])
    M = max(map(len, list(defaults.keys()))) + 4
    options = '\n'.join([f"\t{k}{nspaces(M-len(k))}{v}" for k, v in defaults.items()][:-1])
    print(f"""
    Usage:
        ./main.py [COMMANDS] [OPTIONS]
        
    Option Format
        --<option_name>=<option_value>
        
    COMMANDS:
    
        train   Train a new model
        test    Use a trained model
        help    Print this help message
    
    OPTIONS:
    
    {options}
        
    Help String
    """)
    exit(0)