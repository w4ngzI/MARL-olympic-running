import json
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
WALL_VALUE = 1
ARC_VALUE = 1
CROSS_VALUE = 1

def fn(init_pos):
    """
    Ax + By = C
    """
    l1, l2 = init_pos
    A = l1[1] - l2[1]
    B = l2[0] - l1[0]
    C = (l1[1] - l2[1]) * l1[0] + (l2[0] - l1[0]) * l1[1]
    
    return A, B, C

def check_on_line(point, A, B, C, init_pos):
    l1, l2 = init_pos
    temp = A * point[0] + B * point[1]
    if abs(temp - C) <= 1e-1: #1e-6
        if not (
            (min(l1[0], l2[0]) <= point[0] <= max(l1[0], l2[0]))
            and (min(l1[1], l2[1]) <= point[1] <= max(l1[1], l2[1])
            )
        ):
            return False  
        else:
            return True
    return False

file_path = 'E:\multiagent\olympic\olympics\maps.json'

def bfs(map,start_pos):
    queue = []
    queue.append(([start_pos[0],start_pos[1]],0))
    result_map = np.zeros_like(map)
    occ_map = np.zeros_like(map)
    occ_map[start_pos[0],start_pos[1]] = 1
    while(len(queue)!=0):
        curr_pos,dist = queue[0]
        queue = queue[1:]
        if map[curr_pos[0],curr_pos[1]] != 1:
            neighbors = [[curr_pos[0],curr_pos[1]+1],[curr_pos[0],curr_pos[1]-1],[curr_pos[0]+1,curr_pos[1]],[curr_pos[0]-1,curr_pos[1]]]
            for n in neighbors:
                
                if 0 <= n[0]<map.shape[0] and 0 <= n[1] <= map.shape[1] and map[n[0],n[1]] != 1 and occ_map[n[0],n[1]] != 1:
                    # print(n)
                    queue.append((n,dist+1))
                    result_map[n[0],n[1]] = dist+1
                    occ_map[n[0],n[1]] = 1
    return result_map

map_id = 2
for map_id in tqdm(range(1)):
    with open(file_path) as f:
        map_config = json.load(f)
        
    map_ = map_config['map'+str(map_id+11)]
    # print(map_)

    rendered_map = np.zeros((700, 700))
    ###########     wall    ###############
    if 'wall' in map_:
        wall = map_['wall']
        wall_num = wall['num']
        if wall_num != 0:
            for key in wall['objects'].keys():
                component = wall['objects'][key]      
                pos = component['initial_position']
                A_, B_, C_ = fn(pos)
                for i in range(700):       #x
                    for j in range(700):    #y
                        if check_on_line([i, j], A_, B_, C_, pos):
                            rendered_map[j, i] = WALL_VALUE
                        
    ###########    arc    ##################
    if 'arc' in map_:
        arc = map_['arc']
        arc_num = arc['num']
        if arc_num != 0:
            for key in arc['objects'].keys():
                component = arc['objects'][key]        
                init_pos = component['initial_position']
                start_radian = component['start_radian']
                end_radian = component['end_radian']
                
                center = [init_pos[0] + 1 / 2 * init_pos[2], init_pos[1] + 1 / 2 * init_pos[3]]
                R = 1 / 2 * init_pos[2]
                
                if start_radian >= 0 and end_radian < 0:
                    end_radian += 360
                thetas = np.linspace(start_radian, end_radian, num=5000)
                    
                for theta in thetas:
                    y = center[1] - R * np.sin(theta * np.pi / 180.0)
                    x = center[0] + R * np.cos(theta * np.pi / 180.0)
                    rendered_map[int(y), int(x)] = ARC_VALUE

    if 'cross' in map_:
        cross = map_['cross']
        cross_num = cross['num']
        if cross_num != 0:
            for key in cross['objects'].keys():
                component = cross['objects'][key]      
                color = component['color']
                if color == 'red':
                    pos = component['initial_position']
                    
                    A_, B_, C_ = fn(pos)
                    for i in range(700):       #x
                        for j in range(700):    #y
                            if check_on_line([i, j], A_, B_, C_, pos):
                                rendered_map[j, i] = CROSS_VALUE
    # start_pos = [pos[0][1]+pos[1][1],pos[0][0]+pos[1][0]]
    # start_pos = [int(_/2) for _ in start_pos]
    start_x = pos[0][0]
    # start_y = pos[0][1]
    all_result_map = np.ones((700,700))*100000
    
    for start_y in range(int(pos[0][1])+1,int(pos[1][1])):
    # for start_x in range(int(pos[0][0])+1,int(pos[1][0])):
    # for start_x in range(291,360):
        rendered_map[start_y,start_x] = 0
        rendered_map[start_y,start_x-1] = 1
        result_map = bfs(rendered_map,[start_y,start_x])
        all_result_map = np.minimum(all_result_map,result_map)
    all_result_map[all_result_map==100000] = 0
    # result_map = result_map/max(result_map) * 255
    # cv2.imwrite('generated_map{}.png'.format(map_id+1), result_map)
    f, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    ax = sns.heatmap(all_result_map, xticklabels=100, yticklabels=100)
    plt.title('Map 11')
    plt.savefig('map11.png', dpi=300)
