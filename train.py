import yaml
import argparse
from envs import NaryTreeEnvironment

# Reading the config file
parser = argparse.ArgumentParser(description="Load a YAML config file.")
parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file")
args = parser.parse_args()
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

'''
1. Load the enviorenment 
2. Load the dataset 
3. Load it into the dataloader 
4. Initialize it to the 
'''
project_name = config['project']

if config['env'] == "tree":
    env = NaryTreeEnvironment(config['depth'], config['branching_factor'])

if config['hyperbolic']:
    
else:


