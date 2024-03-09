# Reinforcement Learning for Quadcopter Control

This repo is for RL-Quad training in a simulation environment. For hardware implementation, a separated repo will be online soon.

1. create a new python env using Anaconda
   `conda env create -n <env_name> -f environment.yml`

2. train model for taking off
   `python main.py --mode train --num_episodes 1000 --control_mode takeoff`
3. train model for tracking unicycle
   `python main.py --mode train --num_episodes 1000 --control_mode tracking` 

4. run pre-trained agent for taking off
   `python main.py --mode test --load_model_path checkpoints/takeoff_NED_25m_50hz_02`
