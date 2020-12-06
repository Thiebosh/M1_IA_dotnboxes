import pygame
#from pygame.locals import *
import importlib
from pygame import gfxdraw
import sys
from collections import namedtuple
from time import sleep
import time
from random import choice
from builtins import input

ai_player_1 = importlib.import_module("players.Thibaut Juzeau", package = None)
ai_player_2 = importlib.import_module("players.bender", package = None)

print(ai_player_1.name + " VS " + ai_player_2.name)

BOARDSIZE = 8
BS = 80
BLACK = (0, 0, 0)
RED = (255, 128, 0)
BLUE = (0, 0, 255)
OWNER_NONE = 0
OWNER_AI1 = 1
OWNER_AI2 = 2

Point = namedtuple('Point', ['id', 'x', 'y', 'partners'])
# Box = namedtuple("Box", ["p1", "p2", "p3", "p4", "owner"])
# initialize game engine
pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Arial', 50)
score_font = pygame.font.SysFont('Arial', 30)
dot_font = pygame.font.SysFont('Arial', 15)

spoke1 = [(2,6),(10,11),(9,13),(4,5)]
spoke2 = [(1,5),(6,7),(10,14),(8,9)]

# set screen width/height and caption
size = BOARDSIZE * BS + BS
SURF = pygame.display.set_mode((size, size))
pygame.display.set_caption("Dots and  Boxes")
# initialize clock. used later in the loop.
clock = pygame.time.Clock()

# the gameboard is stored as a list of points
# points contain their number, and the number of their connections
board = []
boxes = []
moves_done = []
moves_done_persons = []
score = [0, 0] # AI1, AI2
is_AI1_turn = True

def reset():
    board.clear()
    for i in range(BOARDSIZE):
        for i2 in range(BOARDSIZE):
            board.append( Point(BOARDSIZE * i + i2, i2 * BS + BS, i * BS + BS, []))

    moves_done.clear()
    moves_done_persons.clear()

    #[id haut gauche, id haut droite, id bas gauche, id bas droite, possesseur]
    boxes.clear()
    boxes.extend([[i, i+1, i+BOARDSIZE, i+BOARDSIZE+1, OWNER_NONE] for i in range(0,7)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+BOARDSIZE+1, OWNER_NONE] for i in range(8,15)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+BOARDSIZE+1, OWNER_NONE] for i in range(16,23)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+BOARDSIZE+1, OWNER_NONE] for i in range(24,31)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+BOARDSIZE+1, OWNER_NONE] for i in range(32,39)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+BOARDSIZE+1, OWNER_NONE] for i in range(40,47)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+BOARDSIZE+1, OWNER_NONE] for i in range(48,55)])

    score[0] = 0
    score[1] = 0
    #is_AI1_turn = True

# print(boxes)
def id_to_index(_id):
    for i in range(len(board)):
        if board[i].id == _id:
            return i
    return -1

# print(board)
def disp_board():

    SURF.fill((255, 255, 255))

    # first lets draw the score at the top
    score_AI1 = score_font.render("{}: {}".format(ai_player_1.name , score[0]), True, BLUE)
    w, h = score_font.size("{}: {}".format(ai_player_1.name,score[0]))
    SURF.blit(score_AI1, (size // 2 - w - 10, 10))
    
    score_AI2 = score_font.render("{}: {}".format(ai_player_2.name, score[1]), True, RED)
    SURF.blit(score_AI2, (size // 2 + 10, 10))
        
    for box in boxes:
        x1 = board[id_to_index(box[0])].x
        y1 = board[id_to_index(box[0])].y
        
        if box[4] == OWNER_AI1:
            pygame.draw.rect(SURF,(150,150,255),(x1,y1,BS,BS))
        elif box[4] == OWNER_AI2:
            pygame.draw.rect(SURF,(255,180,180),(x1,y1,BS,BS))

    for i, move in enumerate(moves_done):
        p1 = board[id_to_index(move[0])]
        p2 = board[id_to_index(move[1])]
        thickness = 15 if move == moves_done[-1] else 15
        if moves_done_persons[i]:
            pygame.draw.line(SURF, BLUE, (p1.x, p1.y), (p2.x, p2.y), thickness)
        else:
            pygame.draw.line(SURF, RED, (p1.x, p1.y), (p2.x, p2.y), thickness)

    for i, point in enumerate(board):
        gfxdraw.filled_circle(SURF, point.x, point.y, 7, BLACK)

    x1 = board[27].x
    y1 = board[27].y
    bonus = score_font.render("7", True, (50,50,50))
    text_width, text_height = myfont.size("7")
    SURF.blit(bonus, (x1 + 50 - text_width / 2, y1 + 50 - text_height / 2))

    bonus = score_font.render("3", True, (100,100,100))
    text_width, text_height = myfont.size("3")
    for i in [9, 13, 41, 45]:
        x1 = board[i].x
        y1 = board[i].y
        SURF.blit(bonus, (x1 + 50 - text_width / 2, y1 + 50 - text_height / 2))

    pygame.display.update() 
    sleep(0.02)

def is_connection(id1, id2):
    if (id1, id2) in moves_done:
        return True
    if (id2, id1) in moves_done:
        return True
    return False

def is_valid(id1, id2):
    if is_connection(id1, id2):
        return False
    p1 = board[id_to_index(id1)]
    p2 = board[id_to_index(id2)]
    if (p1.x == p2.x + BS or p1.x == p2.x - BS) and p1.y == p2.y:
        return True
    if p1.x == p2.x and (p1.y == p2.y + BS or p1.y == p2.y - BS):
        return True
    return False
    # return ((id1, id2) not in moves_done and (id2, id1) not in moves_done) and (id2 == id1 + 1 or id2 == id1 - 1 or id2 == id1 + BOARDSIZE or id2 == id1 - BOARDSIZE)

def move(is_AI1, id1, id2):
    # connects id1 and id2
    # depends on somebody else to check if move is valid
    board[id_to_index(id1)].partners.append(id2)
    board[id_to_index(id2)].partners.append(id1)
    moves_done.append((id1, id2))
    moves_done_persons.append(is_AI1)
    return check_move_made_box(is_AI1, id1, id2)

def possible_moves():
    possible = []
    for a in range(0, len(board)):
        for b in range(a, len(board)):
            if b == a:
                continue
            if not is_valid(a, b):
                continue
            possible.append((a, b))

    return possible

def count_connections_box(box):
    # counts the number of lines that exist inside given box
    # note - this is the points on the box itself, NOT an index to the box
    count = 0
    not_connections = []
    
    if (box[0], box[1]):
        count += 1
    else:
        not_connections.append((box[0], box[1]))
        not_connections.append((box[1], box[0]))
    
    if is_connection(box[1], box[3]):
        count += 1
    else:
        not_connections.append((box[1], box[3]))
        not_connections.append((box[3], box[1]))
    
    if is_connection(box[2], box[3]):
        count += 1
    else:
        not_connections.append((box[2], box[3]))
        not_connections.append((box[3], box[2]))
    
    if is_connection(box[2], box[0]):
        count += 1
    else:
        not_connections.append((box[2], box[0]))
        not_connections.append((box[0], box[2]))
        
    return (count, not_connections)

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap

#@timing
def decide_and_move1():
    # randomly pick a valid move
    possible = possible_moves()
    my_choice = ai_player_1.play(boxes, possible, True)
    is_box = move(True, my_choice[0],my_choice[1])

    if is_box:
        disp_board()
        if not check_complete():
            decide_and_move1()

#@timing
def decide_and_move2():
    possible = possible_moves()
    my_choice = ai_player_2.play(boxes, possible, False)
    is_box = move(False, my_choice[0],my_choice[1])

    if is_box:
        disp_board()
        if not check_complete():
            decide_and_move2()

def check_complete():
    possible = possible_moves()
    if len(possible) == 0:
        # game is finished!
        #print("Game over")
        if score[0] > score[1]:
            print("{} won! Score: {} to {}".format(ai_player_1.name, score[0], score[1]))
        elif score[1] > score[0]:
            print("{} won! Score: {} to {}".format(ai_player_2.name, score[0], score[1]))
        else:
            print("DRAW. Score: {} to {}".format(score[0],score[1]))
        sleep(0.5)
        reset()
        return True
    else :
        return False

def check_move_made_box(is_AI1, id1, id2):
    is_box = False
    # check if the connection just make from id1 to id2 made a box
    for i, box in enumerate(boxes):
        temp = list(box[:-1])
        if id1 not in temp or id2 not in temp:
            continue
        temp.remove(id1)
        temp.remove(id2)
        
        if is_connection(temp[0],temp[1]) and ((is_connection(id1, temp[0]) and is_connection(id2, temp[1])) or (is_connection(id1, temp[1]) and is_connection(id2, temp[0]))):
            # yup, we just made a box

            bonus = 1
            if i == 24:               # BONUS 7
                bonus = 7
            elif i in [8, 12, 36, 40]:   # BONUS 3 
                bonus = 3

            if is_AI1:
                score[0] += bonus
                boxes[i][4] = OWNER_AI1
            else:
                score[1] += bonus
                boxes[i][4] = OWNER_AI2
            
            is_box = True

    return is_box

###############################################################################

reset()
disp_board()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    is_AI1_turn = True
    
    decide_and_move1()
    if check_complete():
        continue
    disp_board()
    
    is_AI1_turn = False

    decide_and_move2()
    if check_complete():
        continue
    disp_board()
    