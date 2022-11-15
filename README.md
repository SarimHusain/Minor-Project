# Minor Project
### Answer Selection in a Question Answering System

Question answering is a computer science discipline within the fields of 
information retrieval and natural language processing, which is concerned 
with building systems that automatically answer questions posed by humans 
in a natural language.

## Usage

### Setup and Install Dependencies
We are using Pipenv to manage dependencies. 
You can find out how to install this for your system.

```bash
pipenv shell
```
```bash
pipenv install
```

### Training and Testing

To train a new model from scratch, you can use the train command.
Keep in mind that this takes a long time and needs a lot of computational resources.
```bash
python main.py train
```

To test a trained model, you can use the test command.
Put the trained model files in the `trained` directory. They are git ignored 
and so will not be committed.
```bash
python main.py test

> Context: This is the context you want to ask a question in...

> Question: Ask a question
> Answer: Get an answer
```
You can ask many questions in a loop on the same context.

To see other available options, use the help command
```bash
python main.py help
```
Here you can set training and other model parameters.

## Documentation

The report can be written in a `.tex` file provided in the `docs` directory.
