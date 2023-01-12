## train with random on all maps in turn, each map training for 3000 episodes
python rl_trainer/all_map_main.py --device cuda --max_episodes 3000 --opponent random

## self-play based training on all maps in turn, each map training for 3000 episodes
# python rl_trainer/all_map_main.py --device cuda --max_episodes 3000 --opponent self

## evaluate the model
python evaluation.py --my_ai ppo --my_ai_run_dir run1 --my_ai_run_episode 33000 \
                        --opponent random --episode 100 --map all