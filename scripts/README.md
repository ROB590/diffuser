# Scripts
## file usage
```
close.py  # Example code of do inference using close loop
combine.py # Example code of running A star in diffuser environment
create_data.py # Disgarded, test code for train different horizon model, but evaluate on different horizon (could acheieve through modify command line, no need to use this file)
evaluate_diffuser.py # Example code of evaluate a model trained on different map
main_rrt.py # Current updated version implementation of using rrt connect in recovery policy
rdm.py # Current reproduce RDM from (adpative online replanning)
plan_maze2d.py # Init plan maze2d from diffuser
switch_between.py #Switch between Astar and diffuser
test.py #forgot when to use, disgarded
train.py #model training scripts from diffuser
```

## If you want to create more data and do data collection
```
https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_maze2d_datasets.py # for maze2d environment
```