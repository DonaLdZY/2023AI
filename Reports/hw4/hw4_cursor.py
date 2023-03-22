def heuristic(state):
    # Calculate the Manhattan distance of    distance = 0
    for i in range(4):
        for j in range(4):
            if state[i][j] == 0:
                continue
            distance += abs(i - (state[i][j] - 1) // 4) + abs(j - (state[i][j] - 1) % 4)
    return distance
# Define the Node class
class Node:
    def __init__(self, state, parent=None, action=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.h = h

    def f(self):
        return self.g + self.h

    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.f() < other.f()

# Define the A* algorithm
def astar(start_state):
    start_node = Node(start_state, None, None, 0, heuristic(start_state))
    heap = []
    heapq.heappush(heap, start_node)
    visited = set()

    while heap:
        node = heapq.heappop(heap)
        if node.state == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]:
            actions = []
            while node.parent is not None:
                actions.append(node.action)
                node = node.parent
            return actions[::-1]

        visited.add(str(node.state))
        for action, state in successors(node.state):
            if str(state) not in visited:
                child_node = Node(state, node, action, node.g + 1, heuristic(state))
                heapq.heappush(heap, child_node)
    return None

# Define the successors function
def successors(state):
    for i in range(4):
        for j in range(4):
            if state[i][j] == 0:
                x, y = i, j
                break

    for action in ['up', 'down', 'left', 'right']:
        new_x, new_y = x, y
        if action == 'up':
            new_x -= 1
        elif action == 'down':
            new_x += 1
        elif action == 'left':
            new_y -= 1
        elif action == 'right':
            new_y += 1

        if 0 <= new_x < 4 and 0 <= new_y < 4:
            new_state = [row[:] for row in state]
            new_state[x][y], new_state[new_x][new_y] = new_state[newstart_state = [[5, 1, 2, 3], [9, 6, 7, 4], [13, 10, 11, 8], [14, 15, 0, 12]]
print(astar(start_state))
start_state = [[5, 1, 2, 3], [9, 6, 7, 4], [13, 10, 11, 8], [14, 15, 0, 12]]
print(astar(start_state))
