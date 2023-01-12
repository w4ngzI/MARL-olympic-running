import numpy as np
from skimage import measure
import cv2
from scipy import ndimage

def cal_angle_of_arrow(obs):
    arrow_mask = (obs == 4)
    if (arrow_mask==0).all:
        return 0,False
    strong_component = measure.label(arrow_mask, connectivity = 1)
    one_arrow_mask = (strong_component == 1).astype(int)
    # print(one_arrow_mask)
    
    mass_center_tuple = ndimage.measurements.center_of_mass(one_arrow_mask)
    mass_center = np.array([mass_center_tuple[0], mass_center_tuple[1]])
    if (one_arrow_mask != 0).any():
        non_zeros_part = np.argwhere(one_arrow_mask == 1)
        y_min = np.min(non_zeros_part[:, 0])
        y_max = np.max(non_zeros_part[:, 0])
        x_min = np.min(non_zeros_part[:, 1])
        x_max = np.max(non_zeros_part[:, 1])
        
        top_min_x = np.argwhere(one_arrow_mask[y_min] == 1).min()
        top_max_x = np.argwhere(one_arrow_mask[y_min] == 1).max()
        bottom_min_x = np.argwhere(one_arrow_mask[y_max] == 1).min()
        bottom_max_x = np.argwhere(one_arrow_mask[y_max] == 1).max()
        left_min_y = np.argwhere(one_arrow_mask[:, x_min] == 1).min()
        left_max_y = np.argwhere(one_arrow_mask[:, x_min] == 1).max()
        right_min_y = np.argwhere(one_arrow_mask[:, x_max] == 1).min()
        right_max_y = np.argwhere(one_arrow_mask[:, x_max] == 1).max()
        
        tmp = np.zeros((25, 25))
        tmp[y_min, top_min_x] = 1
        tmp[y_min, top_max_x] = 1
        tmp[y_max, bottom_min_x] = 1
        tmp[y_max, bottom_max_x] = 1
        tmp[left_min_y, x_min] = 1
        tmp[left_max_y, x_min] = 1
        tmp[right_min_y, x_max] = 1
        tmp[right_max_y, x_max] = 1
        
        # print(tmp.astype(int))
        
        template = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        
        corners = np.argwhere(tmp == 1)
        for corner in corners:
            if tmp[corner[0], corner[1]] == 0:
                continue
            if corner[0] > 1 and corner[0] < 24 and corner[1] > 1 and corner[1] < 24:
                tmp[corner[0]-1:corner[0]+2, corner[1]-1:corner[1]+2] = tmp[corner[0]-1:corner[0]+2, corner[1]-1:corner[1]+2] * template
            
        # print(tmp.astype(int))
        
        corners_after_surpress = np.argwhere(tmp == 1)
        
        if corners_after_surpress.shape[0] == 3:
            # print("corners_after_surpress", corners_after_surpress)
            # print("mass_center", mass_center)
            diff = corners_after_surpress - mass_center
            # print("diff", diff)
            norm = np.linalg.norm(diff, ord = 2, axis = 1)
            # print(norm.shape)
            peak_index = np.argmin(norm)
            peak = corners_after_surpress[peak_index]
            # print("peak", peak)
            
            agent_point = np.array([24, 12])
            
            distance = np.linalg.norm(agent_point - peak, ord = 2, axis = 0)
            if distance < 12.0:
                angle_list = []
                for corner_index in range(corners_after_surpress.shape[0]):
                    if corner_index == peak_index:
                        continue
                    #######(25 - peak[0]) - (25 - corners_after_surpress[corner_index][0])
                    angle = np.arctan2(corners_after_surpress[corner_index][0] - peak[0], peak[1] - corners_after_surpress[corner_index][1]) * 180 / np.pi
                    
                    # print(corners_after_surpress[corner_index])
                    # print('angle', angle)
                    angle_list.append(angle)
                    # print(angle_list)
                angle = np.array(angle_list)
                
                return (90.0 - angle.mean()),True

            else:
                #############(25 - peak[0]) - (25 - agent_point[0])
                angle = np.arctan2(agent_point[0] - peak[0], peak[1] - agent_point[1])
                return (90.0 - angle),False
        
        return 0,False
        
def cal_wall_distance(obs):
    wall_mask = (obs == 6)
    # print('aaaaaaaaaaaaaa')
    # print(wall_mask.astype(int))
    wall_points = np.argwhere(wall_mask == 1)
    agent_points = np.array([24, 12])
    
    diff = wall_points - agent_points
    distance = np.linalg.norm(diff, ord = 2, axis = 1)
    
    point_with_min_dist = wall_points[np.argmin(distance)]
    tmp = np.zeros(shape = (25, 25))
    tmp[point_with_min_dist[0], point_with_min_dist[1]] = 1
    # print(tmp.astype(int))
    
    return distance.min()


def cal_ray(angle):
    theta = angle * np.pi / 180
    tmp = np.zeros(shape = (25, 25))
    if theta == 0:
        tmp[:, 12] = 1
    else:
        #############(y - 24) / (x - 12)=  tan(theta - pi/2)
        for raw in range(25):
            for col in range(25):
                ideal_pos = 24 + np.tan(theta - np.pi / 2) * (col - 12)
                if (raw >= ideal_pos - 1) and (raw <= ideal_pos + 1):
                    tmp[raw, col] = 1

    return tmp
    
def dash(obs, action_angle):
    wall_mask = (obs == 6)
    goal_mask = (obs == 7)

    ray_mask = cal_ray(action_angle)
    wall_points_on_map = wall_mask * ray_mask
    goal_points_on_map = goal_mask * ray_mask

    agent_point = np.array([24, 12])
    if (wall_points_on_map != 0).any() and (goal_points_on_map != 0).any():
        wall_points = np.argwhere(wall_points_on_map == 1)
        goal_points = np.argwhere(goal_points_on_map == 1)

        wall = np.mean(wall_points, axis = 0)
        goal = np.mean(goal_points, axis = 0)

        wall_distance = np.linalg.norm(wall - agent_point, ord = 2, axis = 0)
        goal_distance = np.linalg.norm(goal - agent_point, ord = 2, axis = 0)
        if goal_distance < wall_distance:
            return True

    return False

def distance_to_wall_in_action_angle(obs, action_angle):
    wall_mask = (obs == 6)
    ray_mask = cal_ray(action_angle)

    wall_points_on_map = wall_mask * ray_mask
    wall_points = np.argwhere(wall_points_on_map == 1)

    agent_points = np.array([24, 12])
    diff = wall_points - agent_points
    distance = np.linalg.norm(diff, ord = 2, axis = 1)
    if len(distance)==0:
        return -1
    return distance.min()