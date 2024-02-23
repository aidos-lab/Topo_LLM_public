# Topo_LLM

## Project Structure

### Config file management

- We want to use Hydra for the config managment:
  https://hydra.cc/docs/patterns/configuring_experiments/

- Overwrite config variable:
  `python run.py run.seed=42`

- Multirun example:
  `python run.py --multirun run.seed=1,2,3,4`

## Datasets

- Dialogue data
  - MultiWOZ
  - SGD:
    https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

