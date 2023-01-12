import numpy as np

a = np.zeros((10,10))
a[0] = 1
a[-1] = 1
a[:,0] = 1
a[:,-1] = 1
a[5,:5] = 1
start_pos = [1,1]

def bfs(map,start_pos):
    queue = []
    queue.append(([start_pos[0],start_pos[1]],0))
    result_map = np.zeros_like(a)
    occ_map = np.zeros_like(a)
    occ_map[start_pos[0],start_pos[1]] = 1
    while(len(queue)!=0):
        curr_pos,dist = queue[0]
        queue = queue[1:]
        neighbors = [[curr_pos[0],curr_pos[1]+1],[curr_pos[0],curr_pos[1]-1],[curr_pos[0]+1,curr_pos[1]],[curr_pos[0]-1,curr_pos[1]]]
        for n in neighbors:
            if map[n[0],n[1]] != 1 and occ_map[n[0],n[1]] != 1:
                queue.append((n,dist+1))
                result_map[n[0],n[1]] = dist+1
                occ_map[n[0],n[1]] = 1
    return result_map
result_map = bfs(a,start_pos)
print(a)
print(result_map)

    