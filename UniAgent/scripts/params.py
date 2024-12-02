# 
import torch

# class containing input params 
class args:
    
    def __init__(
        self,
    ):
        print("Initialize params for agent training")
        self.args = {}   
 
    def add_args(
        self,
        key, # argument key
        val, # argumnet value
    ):
        
        self.args[key] = val

    def add_multi_args(
        self,
        keys, # list of keys
        vals, # list of values 
    ):

        for key, val in zip(keys, vals):
            self.add_args(
                key,
                val,
            )

    def print_args(
        self,
    ):
        for key, val in self.args.items():
            print("{}:\t{}".format(key, val))
