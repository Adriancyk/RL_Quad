# Reinforcement Learning for Quadcopter Control


1. create a new python env using Anaconda
   `conda env create -n <env_name> -f environment.yml`

2. train model 
   `python main.py --num_episodes 1000`

3. run pre-trained agent for takeoff
   `python main.py --mode test --ckpt_path checkpoints/sac_takeoff_Quadrotor_1m`
