import numpy as np

def k_point_crossover(a: np.ndarray, b: np.ndarray, points: np.ndarray):
    temp_1 = a
    temp_2 = b
    
    if 1 == len(points) or 2 == len(points):
        endindx = len(temp_1) if 1 == len(points) else points[1]
        startind = points[0] + 1
        
        temp_1[startind:endindx], temp_2[startind:endindx] = temp_2[startind:endindx], temp_1[startind:endindx]
        
        return temp_1, temp_2
    
    for i in range(0, len(points), 1):
        curr_point = 0
        last_point = 0
            
        if i != 0:
            curr_point = points[i]
            last_point = points[i - 1] + 1
        
        if (i == len(points) - 1):
            if i % 2 != 0:
                temp_1[last_point:curr_point], temp_2[last_point:curr_point] = temp_2[last_point:curr_point], temp_1[last_point:curr_point]
            else:
                temp_1[curr_point:], temp_2[curr_point:] = temp_2[curr_point:], temp_1[curr_point:]
        if i % 2 != 0:
            temp_1[last_point:curr_point], temp_2[last_point:curr_point] = temp_2[last_point:curr_point], temp_1[last_point:curr_point]
    
    return temp_1, temp_2