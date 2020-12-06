# ----------------------------------------
#            Import librairies
# ----------------------------------------

import sys
import time
from collections import namedtuple
from importlib import import_module
import pandas as pd

# ----------------------------------------
#            Global variables to define
# ----------------------------------------

AI_PLAYER_1 = "algorithm"
AI_PLAYER_2 = "algorithm"

GRAPHIC_MODE = True
DELAY = 0.01 #0.04

NB_DUEL = 5 #200 max d'un coup ou save tous les 200 -> permet de faire des backups

# ----------------------------------------
#            Global fixed variables
# ----------------------------------------

BOARDSIZE = 8 #nb de points en ligne et colonne
BS = 80 #distance entre points

OWNER_NONE = 0
OWNER_AI1 = 1
OWNER_AI2 = 2

# the gameboard is stored as a list of points
# points contain their number, and the number of their connections
Point = namedtuple('Point', ['id', 'x', 'y', 'id_connected_points'])

init_board = []
init_lineboard = []
init_moves_remaining = []

board = [] #[id_point, coord x, coord y, id_connected_points]
boxes = [] #[id haut gauche, id haut droite, id bas gauche, id bas droite, possesseur]
lineboard = []
moves_done = []
moves_remaining = []

score = [0, 0] # AI1, AI2
is_AI1_turn = True


# ----------------------------------------
#           General functions
# ----------------------------------------

if GRAPHIC_MODE:
    import pygame
    from pygame import gfxdraw
    from time import sleep

    pygame.init() #load modules during loading functions
    pygame.font.init()
    myfont = pygame.font.SysFont('Arial', 50)
    score_font = pygame.font.SysFont('Arial', 30)
    dot_font = pygame.font.SysFont('Arial', 15)
    
    #BS devrait être ici
    LINE_THICKNESS = 15 #epaisseur traits
    DOT_THICKNESS = 7
    size = BOARDSIZE * BS + BS #taille contenu fenêtre
    SURF = pygame.display.set_mode((size, size)) #surface fenêtre
    pygame.display.set_caption("Dots and  Boxes")
    
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 128, 0)
    BLUE = (0, 0, 255)
    
    def disp_board():
        SURF.fill(WHITE)

        # first lets draw the score at the top
        score_AI1 = score_font.render("{}: {}".format(ai_player_1.name , score[0]), True, BLUE)
        w = score_font.size("{}: {}".format(ai_player_1.name,score[0]))
        SURF.blit(score_AI1, (size // 2 - w[0] - 10, 10))
        
        score_AI2 = score_font.render("{}: {}".format(ai_player_2.name, score[1]), True, RED)
        SURF.blit(score_AI2, (size // 2 + 10, 10))

        # then, draw aeras
        for box in boxes:
            x1 = board[box[0]].x
            y1 = board[box[0]].y
            
            if box[4] == OWNER_AI1:
                pygame.draw.rect(SURF,(150,150,255),(x1,y1,BS,BS))
            elif box[4] == OWNER_AI2:
                pygame.draw.rect(SURF,(255,180,180),(x1,y1,BS,BS))

        # ahead, draw ticks with good color
        for i, player in enumerate(lineboard):
            if player != OWNER_NONE:
                point1, point2 = init_moves_remaining[i]
                point1 = board[point1]
                point2 = board[point2]
                pygame.draw.line(SURF, BLUE if player == OWNER_AI1 else RED, (point1.x, point1.y), (point2.x, point2.y), LINE_THICKNESS)

        # on top, draw dots
        for i, point in enumerate(board):
            gfxdraw.filled_circle(SURF, point.x, point.y, DOT_THICKNESS, BLACK)
            
            #display id points
            # point_display = score_font.render(str(point.id), True, WHITE,BLACK)
            # text_width, text_height = myfont.size(str(point.id))
            # SURF.blit(point_display, (int(point.x - text_width / 3), int(point.y - text_height / 3)))

        #display '7' in the middle
        x1 = board[27].x
        y1 = board[27].y
        bonus = score_font.render("7", True, (50,50,50))
        text_width, text_height = myfont.size("7")
        SURF.blit(bonus, (int(x1 + 50 - text_width / 2), int(y1 + 50 - text_height / 2)))

        #display '3' in the squares
        bonus = score_font.render("3", True, (100,100,100))
        text_width, text_height = myfont.size("3")
        for i in [9, 13, 41, 45]:
            x1 = board[i].x
            y1 = board[i].y
            SURF.blit(bonus, (int(x1 + 50 - text_width / 2), int(y1 + 50 - text_height / 2)))

        pygame.display.update()

def timing(f): # permet de calculer temps d'exécution d'une fonction
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap

def init_game():
    global board
    global lineboard
    global moves_remaining

    board.clear()
    boxes.clear()
    lineboard.clear()
    moves_done.clear()
    moves_remaining.clear()

    board = init_board.copy()
    lineboard = init_lineboard.copy()
    moves_remaining = init_moves_remaining.copy()
    
    score[0] = 0
    score[1] = 0

    #[id haut gauche, id haut droite, id bas gauche, id bas droite, possesseur] donc ne peut pas faire range 0,55
    boxes.extend([[i, i+1, i+BOARDSIZE, i+1+BOARDSIZE, OWNER_NONE] for i in range(0,7)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+1+BOARDSIZE, OWNER_NONE] for i in range(8,15)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+1+BOARDSIZE, OWNER_NONE] for i in range(16,23)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+1+BOARDSIZE, OWNER_NONE] for i in range(24,31)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+1+BOARDSIZE, OWNER_NONE] for i in range(32,39)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+1+BOARDSIZE, OWNER_NONE] for i in range(40,47)])
    boxes.extend([[i, i+1, i+BOARDSIZE, i+1+BOARDSIZE, OWNER_NONE] for i in range(48,55)])

def is_set_connected(point1, point2): #identifiants
    return True if (point1, point2) in moves_done or (point2, point1) in moves_done else False
    
def is_move_closing_box(point1, point2):
    is_box = False
    
    for box in [item for item in boxes if (point1 in item and point2 in item)]:
        i = boxes.index(box)

        tmp = list(box) #necessaire pour ne pas detruire boxes
        tmp.remove(point1)
        tmp.remove(point2)

        if  (is_set_connected(tmp[0], tmp[1]) and 
                ((is_set_connected(point1, tmp[0]) and is_set_connected(point2, tmp[1])) or
                 (is_set_connected(point1, tmp[1]) and is_set_connected(point2, tmp[0])))):

            is_box = True #return pas car peut fermer plusieurs boites
            bonus = 1

            if i == 24: # id square bonus 7
                bonus = 7
            elif i in [8, 12, 36, 40]: # id squares bonus 7
                bonus = 3

            #applique gain
            if is_AI1_turn:
                score[0] += bonus
                boxes[i][4] = OWNER_AI1
            else:
                score[1] += bonus
                boxes[i][4] = OWNER_AI2

    return is_box

def apply_move(point1, point2):
    board[point1].id_connected_points.append(point2)
    board[point2].id_connected_points.append(point1)
    moves_done.append((point1, point2))

    lineboard[init_moves_remaining.index((point1, point2))] = OWNER_AI1 if is_AI1_turn else OWNER_AI2

# ----------------------------------------
#           Player call functions
# ----------------------------------------

def decide_and_move(ai_player):
    lineboard_before = lineboard.copy()

    ai_choice = ai_player.play(lineboard, moves_remaining)

    id_pt1 = ai_choice[0]
    id_pt2 = ai_choice[1]

    #verify that move is correct
    if (id_pt1,id_pt2) in moves_remaining:
        moves_remaining.remove((id_pt1,id_pt2))
    elif (id_pt2,id_pt1) in moves_remaining:
        moves_remaining.remove((id_pt2,id_pt1))
        id_pt1, id_pt2 = id_pt2, id_pt1 #swap
    else:
        raise NameError('invalid move')

    apply_move(id_pt1, id_pt2) #met plateau à jour
    
    if GRAPHIC_MODE:
        sleep(DELAY)
        disp_board()

    #if close a square, play again
    if is_move_closing_box(id_pt1, id_pt2):
        if len(moves_remaining) != 0: #fin de jeu
            decide_and_move(ai_player)
        else:
            return True

    #if it remains moves...
    return len(moves_remaining) == 0


# @timing #décommenter pour voir temps d'exécution
def duel(current_duel, player1, player2):
    global is_AI1_turn
    ended = False

    init_game()

    while not ended:
        if GRAPHIC_MODE:
            for event in pygame.event.get(): #change to try except if possible
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        is_AI1_turn = True
        ended = decide_and_move(player1)
        
        if not ended:
            is_AI1_turn = False
            ended = decide_and_move(player2)

    if GRAPHIC_MODE:
        disp_board()

    return score

# ----------------------------------------
#               Main entry
# ----------------------------------------

if __name__ == "__main__":
    #prepare init_board one time only
    for y in range(BOARDSIZE):
        for x in range(BOARDSIZE):
            init_board.append( Point(BOARDSIZE * y + x, x * BS + BS, y * BS + BS, []))

    #prepare init_moves_remaining one time only
    for lineIndex in range (BOARDSIZE-1):
        init_moves_remaining.extend((n,n+1) for n in range(BOARDSIZE*lineIndex, BOARDSIZE*(lineIndex+1)-1))
        init_moves_remaining.extend((n,n+BOARDSIZE) for n in range(BOARDSIZE*lineIndex, BOARDSIZE*(lineIndex+1)))
    init_moves_remaining.extend((n,n+1) for n in range(BOARDSIZE*(BOARDSIZE-1), BOARDSIZE*BOARDSIZE-1))

    #prepare init_lineboard one time only
    init_lineboard = [OWNER_NONE for i in range(len(init_moves_remaining))]


    #instantiate class by loading or creating model
    ai_player_1 = import_module(AI_PLAYER_1).Agent(moves_list=init_moves_remaining.copy())
    ai_player_2 = import_module(AI_PLAYER_2).Agent(moves_list=init_moves_remaining.copy())
    

    #start battle
    for i in range(NB_DUEL):
        duel(i, ai_player_1, ai_player_2)


    #get data into dataframe
    all_boards1, all_moves1 = ai_player_1.get_data()
    all_boards2, all_moves2 = ai_player_2.get_data()
    
    pd.DataFrame(zip(all_boards1 + all_boards2, all_moves1 + all_moves2), columns = ["board", "best_move"]).to_csv("dataframe_parties.csv")
