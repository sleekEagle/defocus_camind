from sacred import Experiment
ex = Experiment()

@ex.config
def my_config():
    foo = 42
    bar = 'baz'

@ex.main
def my_main():
    pass