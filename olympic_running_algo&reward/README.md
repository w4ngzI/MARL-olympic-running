## Olympics Running Competition

rl_trainer/main_reward.py,rl_trainer/main_ddpg.py,rl_trainer/main_sac.py are used to train a PPO/DDPG/SAC model. You should run like this:
```shell
python rl_trainer/main_reward.py --device cuda --map 1

```
In rl_trainer/main_reward.py , --use_***_reward can specify the reward types.
For example, 
--use_angle_reward, or
--use_wall_reward, or
--use_dash_reward, or
--use_dist_reward



Finally,

--use_dist_win can specify the victory judgement.

### Usage

```shell
git clone https://github.com/Leo-xh/Competition_Olympics-Running.git
cd Competition_Olympics-Running

# training ppo with random opponent
python rl_trainer/main.py --device cuda --map 1

# evaluating ppo with random opponent
python evaluation.py --my_ai ppo --my_ai_run_dir run1 --my_ai_run_episode 1500 --map 1
```



