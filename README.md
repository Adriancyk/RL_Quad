# Reinforcement Learning for Quadcopter Control


1. create a new python env using Anaconda
   `conda env create -n <env_name> -f environment.yml`

2. train model for taking off
   `python main.py --num_episodes 1000 --control_mode takeoff`
3. train model for tracking unicycle
   `python main.py --num_episodes 1000 --control_mode tracking` 

4. run pre-trained agent for takeoff
   `python main.py --mode test --load_model_path checkpoints/sac_Quadrotor_takeoff_1m_01`
