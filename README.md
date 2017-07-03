# Reinforcement Learning Summer School : Practical Tutorial on RL for Continuous Control

Here we go over:

+ How to setup <a href="http://www.mujoco.org/">MuJoCo</a> and <a href="https://github.com/openai/rllab">openai/rllab</a>
+ How to run basic TRPO and DDPG code
+ The core code snippets in TRPO and DDPG so you can build on top of these algorithms
+ How to create your own modified MuJoCo environment (Multi-task modifications can be pull-requested into <a href="https://github.com/Breakend/gym-extensions">gym-extensions</a>)


## How to run examples

### Run TRPO

```bash
cd code; source activate rllab3; python run_trpo.py Hopper-v1
```

### Run DDPG

```bash
cd code; source activate rllab3; python run_ddpg.py Hopper-v1
```

### Plotting Results

```bash
cd code; python plot_results.py data/progress.csv Hopper-v1 --labels "trpo"
```


### Manual testing of an env and custom env
```bash
cd code; python test_manual Hopper-v1
```

```bash
cd code; python test_modified_hopper_env_manually.py
```
