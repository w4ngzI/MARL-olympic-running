## Olympics Running Competition - Strategy

**Please check `all_maps_train.sh`**


### Evaluation

To directly evaluate the *pre-trained agent by us*, please run
```bash
python evaluation.py --my_ai ppo --my_ai_run_dir run1 --my_ai_run_episode 33000 \
                        --opponent random --episode 100 --map all
```
We have already save the best model parameters under the folder `rl_trainer/models/olympics-running/ppo/run1/trained_model`

&nbsp;

### Train

To train an PPO agent vs. random on all maps in turn, please run
```bash
python rl_trainer/all_map_main.py --device cuda --max_episodes 3000 --opponent random
```
We recommend this training method to reproduce our performance.

&nbsp;

To train an self-play based PPO agent on all maps in turn, please run
```bash
python rl_trainer/all_map_main.py --device cuda --max_episodes 3000 --opponent self
```