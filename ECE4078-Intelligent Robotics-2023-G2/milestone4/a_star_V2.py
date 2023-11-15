import math, json, os.path
from generateGroundTruth import updateTruemap

class Node:
    def __init__(self, grid, x, y):
        self.x = x
        self.y = y
        self.grid = grid
        self.type = 'road'
        self.g_score = float('inf')
        self.f_score = float('inf')

    def get_neighbors(self):
        # Collection of arrays representing the x and y displacement
        rows = len(self.grid)
        cols = len(self.grid[0])
        directions = [[1, 0], [1, 1], [0, 1], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
        neighbors = []
        for direction in directions:
            neighbor_x = self.x  + direction[0]
            neighbor_y = self.y + direction[1]
            if neighbor_x >= 0 and neighbor_y >= 0 and neighbor_x < cols and neighbor_y < rows:
                neighbors.append(self.grid[neighbor_y][neighbor_x])
        return neighbors

# returns distance between two nodes
def distance(node1, node2):
    return math.sqrt(math.pow(node1.x - node2.x, 2) + math.pow(node1.y - node2.y, 2))

# Measures distance from node to endpoint with nodes only being able to travel vertically, horizontally, or diagonally
def h_score(start, end):
    x_dist = abs(end.x - start.x)
    y_dist = abs(end.y - start.y)
    diagonal_steps = min(x_dist, y_dist)
    straight_steps = y_dist + x_dist - 2 * diagonal_steps
    return diagonal_steps * math.sqrt(2) + straight_steps

def reconstruct_path(grid, came_from, current):
    path_temp = []
    current_key = str(current.x) + ' ' + str(current.y)
    path_temp.append((current.x,current.y))
    while current_key in came_from:
        current = came_from[current_key]
        current_key = str(current.x) + ' ' + str(current.y)
        path_temp.append((current.x,current.y))
    # Convert points to xy-coordinate tuple
    path = []  
    for i in range(len(path_temp)-2,0,-1):
        x_coor = round((path_temp[i][0] - 8)*0.2,2)
        y_coor = round(-(path_temp[i][1] - 8)*0.2,2)
        path.append((x_coor,y_coor))
    print(path)
    # Reduce the number of points by equating the slopes (where possible)
    skip_block = 1          # AMOUNT OF ALLOWABLE BLOCKS TO BE SKIPPED (CAN BE CHANGED)
    counter = 0
    points_to_remove = []
    for i in range(1,len(path)-1):
        if path[i][0] == path[i-1][0]:
            delta_x_1 = 0.01
        else:
            delta_x_1 = path[i][0]-path[i-1][0]
        if path[i+1][0] == path[i][0]:
            delta_x_2 = 0.01
        else:
            delta_x_2 = path[i+1][0]-path[i][0]
        if round((path[i][1]-path[i-1][1])/delta_x_1,1) == round(((path[i+1][1]-path[i][1])/delta_x_2),1):
            if counter != skip_block:
                points_to_remove.append(path[i])
                print("Remove: ",path[i])
                counter += 1
            else:
                counter = 0
    for i in range(len(points_to_remove)):
        path.remove(points_to_remove[i])
    return path

def lowest_f_score(node_list):
    final_node = None
    for node in node_list:
        if not final_node or node.f_score < final_node.f_score:
            final_node = node
    return final_node

# Performs the pathfinding algorithm. start and end are (x, y) tuples
def a_star(truemap_file, end, start = (0,0)):
    if os.path.exists('./TRUEMAP_new.txt'):
        truemap_file = 'TRUEMAP_new.txt'
    # Convert xy-coordinate to truemap
    # Read from the file as string
    with open(truemap_file) as f:
        data = f.read()
        
    # Reconstruct the data as a dictionary
    js = json.loads(data)    

    # Convert xy points to array indeces
    obstacle_list = []
    start0 = start[0]*10
    start1 = start[1]*10
    end0 = end[0]*10
    end1 = end[1]*10
    start_tuple = (8 - start1/2, 8 + start0/2)
    end_tuple = (8 - end1/2, 8 + end0/2)
    # print(start_tuple)
    # print(end_tuple)

    for pose in js.keys():
        x_coor = js[pose]['x']*10
        y_coor = js[pose]['y']*10
        obstacle_column_index =8 + x_coor/2
        obstacle_row_index = 8 - y_coor/2
        if (obstacle_row_index,obstacle_column_index) != start_tuple and (obstacle_row_index,obstacle_column_index) != end_tuple:
            obstacle_list.append((obstacle_row_index,obstacle_column_index))
            # print((obstacle_row_index,obstacle_column_index))

    # Open the converted truemap txt file to be overwritten
    truemap = open('TRUEMAP(converted).txt','w')

    # Draw the truemap
    rows = 17
    cols = 17    
    for row in range(rows):
        if (row == 0 or row == 16):
            truemap.write("1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n")
        else:
            for column in range(cols):
                if column == 0 or column == 16:
                    truemap.write("1 ")
                else:
                    if (row,column) in obstacle_list:
                        truemap.write("1 ")
                    elif (row,column) == start_tuple:
                        truemap.write("2 ")
                    elif (row,column) == end_tuple:
                        truemap.write("3 ")
                    else:
                        truemap.write("0 ")
            truemap.write("\n")

    truemap.close()
# Convert truemap to tiles
    grid = []

    file = open('TRUEMAP(converted).txt', 'r')
    lines = file.read().split('\n')
    file.close()

    start = start_tuple
    end = end_tuple

    for i in range(rows):
        row = list(map(int, lines[i].split()))
        row_nodes = []
        for j in range(len(row)):
            node = Node(grid, j, i)
            element = row[j]
            if element == 1:
                node.type = 'wall'
            elif element == 2:
                node.type = 'start'   
                start = node         
                # print("Start node column=",j, ",row=",i)
            elif element == 3:
                node.type = 'end'  
                end = node          
                # print("End node column=",j, ",row=",i)

            row_nodes.append(node)
        grid.append(row_nodes)    

# Begin A* implementation    
    open_set = []
    closed_set = []
    came_from = {}

    start.g_score = 0
    start.f_score = h_score(start, end)

    open_set.append(start)

    i = 0
    while len(open_set) > 0:
        i += 1
        current = lowest_f_score(open_set)
        open_set.remove(current)
        closed_set.append(current)

        if current == end:
            return reconstruct_path(grid, came_from, current)

        for neighbor in current.get_neighbors():
            if neighbor in closed_set or neighbor.type == 'wall':
                continue
            # If both adjacent nodes are walls, dont let it be searched
            adj_node_1 = grid[current.y][neighbor.x]
            adj_node_2 = grid[neighbor.y][current.x]
            if adj_node_1.type == 'wall' and adj_node_2.type == 'wall':
                continue
            tentative_g_score = current.g_score + distance(current, neighbor)
            if neighbor not in open_set:
                open_set.append(neighbor)
            elif tentative_g_score > neighbor.g_score:
                # Not a better path
                continue
            # Found a better path
            came_from[str(neighbor.x) + ' ' + str(neighbor.y)] = current
            neighbor.g_score = tentative_g_score
            neighbor.f_score = neighbor.g_score + h_score(neighbor, end)
    # End A* implementation

# TEST DATA
if __name__ == "__main__":
    path = a_star('TRUEMAP.txt', (-1.2 ,0))
    print(path)
    # updateTruemap('TRUEMAP.txt',(0.2,0.4),"orange")
    # path = a_star('TRUEMAP.txt', (1.2 ,1.2), (0.0,0.2))
    # print(path)
    # updateTruemap('TRUEMAP.txt',(0.6,0.8),"capsicum")
    # path = a_star('TRUEMAP.txt', (1.2 ,1.2), (0.4,0.6))
    # print(path)
    # updateTruemap('TRUEMAP.txt',(1.0,1.2),"mango")
    # path = a_star('TRUEMAP.txt', (1.2 ,1.2), (0.8,1.2))
    # print(path)