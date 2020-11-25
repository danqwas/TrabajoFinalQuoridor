import pygame
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from queue import PriorityQueue
from math import inf
from heapq import heappush, heappop
from itertools import count
import time
# Colors
MARRON = (255, 255, 255)
NEGRO = (0, 0, 0)
AZUL = (0, 255, 0)

# Table: 9x9
DIM = 9
RES = 720

# Variable declaration
running = True
editorMode = False
barriers = []
pygame.init()
Graph = nx.Graph()
screen = pygame.display.set_mode((RES, RES))
pygame.display.set_caption("Quoridor")
myfont = pygame.font.SysFont('Comic Sans MS', 30)
textsurface = myfont.render('Modo edicion activado', False, (255, 0, 0))


# Bot Class
class Bot:
    def __init__(self, node, x, y, barriers, name):
        self.node = node
        self.barriers = barriers
        self.name = name
        self.x = x
        self.y = y
        self.type = type
        self.step = 80
        self.width = 60

    def dijkstraTWO(self, start: str, end: str) -> 'List':
        def backtrace(prev, start, end):
            node = end
            path = []
            while node != start:
                path.append(node)
                node = prev[node]
            path.append(node)
            path.reverse()
            return path

        def cost(u, v):
            return Graph.get_edge_data(u, v).get('weight')

        prev = {}
        dist = {v: inf for v in list(nx.nodes(Graph))}
        visited = set()
        pq = PriorityQueue()
        dist[start] = 0
        pq.put((dist[start], start))
        while 0 != pq.qsize():
            curr_cost, curr = pq.get()
            visited.add(curr)
            for neighbor in dict(Graph.adjacency()).get(curr):
                path = dist[curr] + cost(curr, neighbor)
                if path < dist[neighbor]:
                    dist[neighbor] = path
                    prev[neighbor] = curr
                    if neighbor not in visited:
                        visited.add(neighbor)
                        pq.put((dist[neighbor], neighbor))
                    else:
                        _ = pq.get((dist[neighbor], neighbor))
                        pq.put((dist[neighbor], neighbor))
        return backtrace(prev, start, end)

    def dijkstra(self, start, goal):
        grafo = nx.to_dict_of_lists(Graph)
        S = []
        Queue = []

        def cost(u, v):
            return Graph.get_edge_data(u, v).get('weight')

        anterior = [0 for i in range(max(grafo) + 1)]
        distancia = [0 for i in range(max(grafo) + 1)]
        for nodo in grafo:
            distancia[nodo] = float("Inf")
            Queue.append(nodo)
        distancia[start] = 0
        while not len(Queue) == 0:
            distancia_minima = float("Inf")
            for nodo in Queue:
                if distancia[nodo] < distancia_minima:
                    distancia_minima = distancia[nodo]
                    nodo_temporal = nodo
            nodo_distancia_minima = nodo_temporal
            Queue.remove(nodo_distancia_minima)
            for vecino in grafo[nodo_distancia_minima]:
                if distancia[nodo_distancia_minima] == float("Inf"):
                    distancia_temporal = 0
                else:
                    distancia_temporal = distancia[nodo_distancia_minima]
                distancia_con_peso = distancia_temporal + cost(nodo_distancia_minima, vecino)
                if distancia_con_peso < distancia[vecino]:
                    distancia[vecino] = distancia_con_peso
                    anterior[vecino] = nodo_distancia_minima
        if nodo_distancia_minima == goal:
            if anterior[nodo_distancia_minima] != 0 or nodo_distancia_minima == start:
                while nodo_distancia_minima != 0:
                    S.insert(0, nodo_distancia_minima)
                    nodo_distancia_minima = anterior[nodo_distancia_minima]
                return S
        else:
            return S

    def all_start(self, start, goal, heuristic=None, weight="weight"):
        graph_allstar = Graph
        if start not in graph_allstar or goal not in graph_allstar:
            msg = f"Either source {start} or target {goal} is not in G"
            raise nx.NodeNotFound(msg)
        if heuristic is None:
            # The default heuristic is h=0 - same as Dijkstra's algorithm
            def heuristic(u, v):
                return 0
        push = heappush
        pop = heappop
        weight = _weight_function(graph_allstar, weight)
        c = count()
        queue = [(0, next(c), start, 0, None)]
        enqueued = {}

        explored = {}
        while queue:
            # Pop the smallest item from queue.
            _, __, curnode, dist, parent = pop(queue)

            if curnode == goal:
                path = [curnode]
                node = parent
                while node is not None:
                    path.append(node)
                    node = explored[node]
                path.reverse()
                return path

            if curnode in explored:
                # Do not override the parent of starting node
                if explored[curnode] is None:
                    continue

                # Skip bad paths that were enqueued before finding a better one
                qcost, h = enqueued[curnode]
                if qcost < dist:
                    continue

            explored[curnode] = parent

            for neighbor, w in graph_allstar[curnode].items():
                ncost = dist + weight(curnode, neighbor, w)
                if neighbor in enqueued:
                    qcost, h = enqueued[neighbor]
                    # if qcost <= ncost, a less costly path from the
                    # neighbor to the source was already determined.
                    # Therefore, we won't attempt to push this neighbor
                    # to the queue
                    if qcost <= ncost:
                        continue
                else:
                    h = heuristic(neighbor, goal)
                enqueued[neighbor] = ncost, h
                push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

        return nx.NetworkXNoPath(f"Node {goal} not reachable from {start}")

    def make_move(self):
        route = self.dijkstraTWO(self.node, 81)
        print(route)
        if not len(route) == 0:
            if route[1] == self.node + 1:
                self.move_right()
            elif route[1] == self.node - 1:
                self.move_left()
            elif route[1] == self.node + DIM:
                self.move_down()
            elif route[1] == self.node - DIM:
                self.move_up()

    def move_up(self):
        if Graph.has_edge(self.node, self.node - DIM):
            self.y = self.y - self.step
            self.node = self.node - DIM
            return True
        else:
            return False

    def move_left(self):
        if Graph.has_edge(self.node, self.node - 1):
            self.x = self.x - self.step
            self.node = self.node - 1
            return True
        else:
            return False

    def move_right(self):
        if Graph.has_edge(self.node, self.node + 1):
            self.x = self.x + self.step
            self.node = self.node + 1
            return True
        else:
            return False

    def move_down(self):
        if Graph.has_edge(self.node, self.node + DIM):
            self.y = self.y + self.step
            self.node = self.node + DIM
            return True
        else:
            return False

    def check_win(self):
        if self.node >= DIM * 8:
            print("---------------------------")
            print(" Bot won")
            print("---------------------------")
            return True
        else:
            return False

    def draw(self):
        pygame.draw.ellipse(screen, AZUL, (self.x, self.y, self.width, self.width))


# Player Class
class Player:
    def __init__(self, node, x, y, barriers, name):
        self.node = node
        self.barriers = barriers
        self.name = name
        self.x = x
        self.y = y
        self.type = type
        self.step = 80
        self.width = 60
        self.has_played = False

    def move_up(self):
        if Graph.has_edge(self.node, self.node - DIM):
            self.y = self.y - self.step
            self.node = self.node - DIM
            self.has_played = True
            return True
        else:
            return False

    def move_left(self):
        if Graph.has_edge(self.node, self.node - 1):
            self.x = self.x - self.step
            self.node = self.node - 1
            self.has_played = True
            return True
        else:
            return False

    def move_right(self):
        if self.node + 1 == 81:
            return False
        if Graph.has_edge(self.node, self.node + 1):
            self.x = self.x + self.step
            self.node = self.node + 1
            self.has_played = True
            return True
        else:
            return False

    def move_down(self):
        if Graph.has_edge(self.node, self.node + DIM):
            self.y = self.y + self.step
            self.node = self.node + DIM
            self.has_played = True
            return True
        else:
            return False

    def check_win(self):
        if self.node < DIM:
            print("---------------------------")
            print(" Player won")
            print("---------------------------")
            return True
        else:
            return False

    def draw(self):
        pygame.draw.ellipse(screen, NEGRO, (self.x, self.y, self.width, self.width))


# Barrier Class
class Barrier:
    def __init__(self, player_node, x, y):
        self.player_node = player_node
        self.x = x
        self.y = y
        self.angle = 90
        self.step = 80
        self.width = 9
        self.height = 160

    def draw(self):
        if self.angle == 90:  # |
            pygame.draw.rect(screen, NEGRO, (self.x, self.y, self.width, self.height))
        elif self.angle == 180:  # -
            pygame.draw.rect(screen, NEGRO, (self.x + 4, self.y - 4, self.height, self.width))

    def rotate(self):
        if self.angle == 90:
            self.angle = 180
        elif self.angle == 180:
            self.angle = 90

    def move_up(self):
        if self.y == 12:
            return False
        self.y = self.y - self.step
        self.player_node = self.player_node - DIM
        return True

    def move_left(self):
        if self.x == 9:
            return False
        self.x = self.x - self.step
        self.player_node = self.player_node - 1
        return True

    def move_right(self):
        if self.x == 649:
            return False
        self.x = self.x + self.step
        self.player_node = self.player_node + 1
        return True

    def move_down(self):
        if self.y == 652:
            return False
        self.y = self.y + self.step
        self.player_node = self.player_node + DIM
        return True

    def save(self):
        if self.angle == 90:
            Graph.remove_edge(self.player_node, self.player_node - 1)
            Graph.remove_edge(self.player_node + DIM, self.player_node + 8)
        elif self.angle == 180:
            Graph.remove_edge(self.player_node, self.player_node - DIM)
            Graph.remove_edge(self.player_node + 1, self.player_node - 8)


# Add all the nodes
for i in range(DIM * DIM):
    Graph.add_node(i)
Graph.add_node(81)

# Add all the connections (edges)
node = 0
for i in range(DIM):
    for j in range(DIM):
        # (↑ ↓) connections
        if i == DIM - 1:
            Graph.add_edge(node, 81, weight=1)
        if i != (DIM - 1):
            Graph.add_edge(node, node + DIM, weight=1)
        # (← →) connections
        if j != (DIM - 1):
            Graph.add_edge(node, node + 1, weight=1)
        node = node + 1


# Draw map
def draw_map():
    block_size = RES / DIM
    for x in range(DIM):
        for y in range(DIM):
            rect = pygame.Rect(x * block_size, y * block_size, block_size, block_size)
            pygame.draw.rect(screen, MARRON, rect, 1)


# Game Mechanics
bot = Bot(4, 329, 12, 10, "Terminator")
player = Player(76, 329, 652, 10, "Marco")
barrier = Barrier(0, -10, -10)

while running:
    # Check player or bot victory
    if player.check_win():
        running = False
    elif bot.check_win():
        running = False

    # Screen Background
    if not editorMode:
        screen.fill((139, 69, 19))
    else:
        screen.fill((191, 125, 59))

    # Event Handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if not player.has_played:
                if not editorMode:
                    # Player's turn
                    if event.key == pygame.K_SPACE:
                        editorMode = True
                        barrier = Barrier(player.node, player.x - 13, player.y - 12)
                        print("Editor mode: enabled")
                    elif event.key == pygame.K_UP:
                        player.move_up()
                    elif event.key == pygame.K_DOWN:
                        player.move_down()
                    elif event.key == pygame.K_LEFT:
                        player.move_left()
                    elif event.key == pygame.K_RIGHT:
                        player.move_right()
                    print("Player node: ", player.node, "(X: ", player.x, " Y: ", player.y, ")")
                else:
                    # Editor Mode
                    if event.key == pygame.K_ESCAPE:
                        editorMode = False
                        print("Editor mode: disabled")
                    elif event.key == pygame.K_r:
                        barrier.rotate()
                    elif event.key == pygame.K_UP:
                        barrier.move_up()
                    elif event.key == pygame.K_DOWN:
                        barrier.move_down()
                    elif event.key == pygame.K_LEFT:
                        barrier.move_left()
                    elif event.key == pygame.K_RIGHT:
                        barrier.move_right()
                    elif event.key == pygame.K_SPACE:
                        editorMode = False
                        barrier.save()
                        barriers.append(barrier)
                        print("New barrier added, exiting editor mode")
                    print("Barrier node: ", barrier.player_node, "(X: ", barrier.x, " Y: ", barrier.y, ")")
            else:
                t0 = time.time()
                # Bot's turn
                bot.make_move()
                player.has_played = False
                print("Bot node:    ", bot.node, "(X: ", bot.x, " Y: ", bot.y, ")")
                print(time.time() - t0)
    # Drawing
    draw_map()
    bot.draw()
    player.draw()
    barrier.draw()
    player.check_win()
    for obj in barriers:
        obj.draw()
    if editorMode:
        screen.blit(textsurface, (15, 15))
    pygame.display.update()
