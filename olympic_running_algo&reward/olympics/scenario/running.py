# from asyncio import FastChildWatcher
from olympics.core import OlympicsBase
import time
import numpy as np
import itertools

class Running(OlympicsBase):
    def __init__(self, map, seed=None):
        super(Running, self).__init__(map, seed)

        self.gamma = 1  # v衰减系数
        self.restitution = 0.5
        self.print_log = False
        self.print_log2 = False
        self.tau = 0.1

        self.speed_cap = 100
        self.use_map_dist = True
        self.draw_obs = True
        self.show_traj = True

        # self.is_render = True

    def check_overlap(self):
        # todo
        pass

    def get_reward(self):

        agent_reward = [0.0 for _ in range(self.agent_num)]

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                agent_reward[agent_idx] = 100.0

        return_dist_rewards = np.array([_ for _ in agent_reward])
        agent_reward = [0.0 for _ in range(self.agent_num)]
        if self.use_map_dist:
            if hasattr(self, "map_dist"):
                map_dist = self.map_dist
            else:
                # npy_path = f"../olympics/map_dist/map{self.map_num}.npy"
                npy_path = f"olympics/dist_matrix/matrix{self.map_num}.npy"

                map_dist = np.load(npy_path)
                print(f"Loaded from {npy_path}")
            for agent_i in range(self.agent_num):
                agent_pos = self.agent_pos[agent_i]
                dist_this_pos = map_dist[int(agent_pos[1]), int(agent_pos[0])]

                if dist_this_pos == 0:
                    all_coordinates = list(itertools.product(range(map_dist.shape[0]), range(map_dist.shape[1])))
                    dist_to_this_pos = [(agent_pos[1] - x) ** 2 + (agent_pos[0] - y) ** 2 for (x, y) in all_coordinates]
                    coordinate_idx_sorted_on_dist = np.argsort(dist_to_this_pos)  # from small to large
                    for coordinate_idx in coordinate_idx_sorted_on_dist:
                        coord = all_coordinates[coordinate_idx]
                        if map_dist[coord] != 0:
                            dist_this_pos = map_dist[coord]
                            break
                    map_dist[int(agent_pos[1]), int(agent_pos[0])] = dist_this_pos

                dist_norm_factor = 100  # NOTE: divide !
                reward_this_pos = -dist_this_pos / dist_norm_factor
                agent_reward[agent_i] += reward_this_pos
                # return_dist_rewards[agent_i] = reward_this_pos

            self.map_dist = map_dist

        return agent_reward,return_dist_rewards

    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                return True

        return False

    def step(self, actions_list):

        previous_pos = self.agent_pos

        time1 = time.time()
        self.stepPhysics(actions_list, self.step_cnt)
        time2 = time.time()
        # print('stepPhysics time = ', time2 - time1)
        self.speed_limit()

        self.cross_detect(previous_pos, self.agent_pos)

        self.step_cnt += 1
        step_reward,dist_reward = self.get_reward()
        done = self.is_terminal()

        time3 = time.time()
        obs_next = self.get_obs()
        time4 = time.time()
        # print('render time = ', time4-time3)
        # obs_next = 1
        # self.check_overlap()
        self.change_inner_state()

        return obs_next, dist_reward, done, "",step_reward
