# ----------------------------------------
#            Import librairies
# ----------------------------------------

import sys
import time
from collections import namedtuple
from importlib import import_module
import random

import matplotlib.pyplot as plt

# ----------------------------------------
#            Global variables to define
# ----------------------------------------

#AI_PLAYER_1 = "players.juzeau_ai_test"
AI_PLAYER_1 = "players.juzeauAI_V11"
#AI_PLAYER_2 = "players.juzeauAI_V"
#AI_PLAYER_1 = "players.algorithm"
#AI_PLAYER_1 = "players.juzeauAI_V3"
AI_PLAYER_2 = "players.algorithm simple"
#AI_PLAYER_2 = "players.juzeau_ai_test"
#AI_PLAYER_2 = "players.c3po"
OTHERS_PLAYERS = [] #["players.c3po", "players.bender", "players.bender"]

TRAIN_MODEL1 = True
TRAIN_MODEL2 = True

GRAPHIC_MODE = True
DELAY = 0.2 #0.04

PERMUTATION_MODE = False #True
PERMUTATION_FREQ = 20

NB_DUEL = 2#200 max d'un coup ou save tous les 200 -> permet de faire des backups

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
    #dot_font = pygame.font.SysFont('Arial', 15)
    
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
        SURF.fill(BLACK)

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
            gfxdraw.filled_circle(SURF, point.x, point.y, DOT_THICKNESS, WHITE)
            
            #display id points
            # point_display = score_font.render(str(point.id), True, WHITE,BLACK)
            # text_width, text_height = myfont.size(str(point.id))
            # SURF.blit(point_display, (int(point.x - text_width / 3), int(point.y - text_height / 3)))

        #display '7' in the middle
        x1 = board[27].x
        y1 = board[27].y
        bonus = score_font.render("7", True, (100,100,100))
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

def decide_and_move(ai_player, nb_moves_chained):
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
    is_box_closed = is_move_closing_box(id_pt1, id_pt2) #met score à jour

    score_AI, score_other = score[0], score[1]

    if GRAPHIC_MODE:
        # if not is_AI1_turn:
        #    input("pause")
        sleep(DELAY)
        disp_board()

    ai_player.play_result(score_AI, score_other, nb_moves_chained, lineboard_before)
    
    #evalue coup de l'adversaire
    if is_AI1_turn:
        ai_player_2.other_play(lineboard_before, ai_choice, score_AI, score_other, nb_moves_chained, lineboard_before)
    else:
        ai_player_1.other_play(lineboard_before, ai_choice, score_AI, score_other, nb_moves_chained, lineboard_before)
    

    #if close a square, play again
    if is_box_closed:
        if len(moves_remaining) != 0: #fin de jeu
            decide_and_move(ai_player, nb_moves_chained + 1)
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
        ended = decide_and_move(player1, 0)
        
        if not ended:
            is_AI1_turn = False
            ended = decide_and_move(player2, 0)

    if GRAPHIC_MODE:
        disp_board()

    return score

@timing #décommenter pour voir temps d'exécution
def tournament():
    global ai_player_1
    global ai_player_2

    result_player1 = []
    result_player2 = []

    result_player1_permuted = []
    result_player2_permuted = []

    permuted = False

    for i in range(NB_DUEL):
        print(f"\nduel {i} - {ai_player_1.name} vs {ai_player_2.name}")
        result = duel(i, ai_player_1, ai_player_2)

        print(f"résultats : {ai_player_1.name if result[0] > result[1] else ai_player_2.name} gagnant - {result[0]} contre {result[1]}")
        # if i > 0 and (i+1) % 10 == 0:
        #     print(f"\n\tmoyenne : {sum(result_player1[-10:])/10} pour {ai_player_1.name} contre {sum(result_player2[-10:])/10} pour {ai_player_2.name}")
        
        ai_player_1.train(lineboard.copy(), result[0], result[1])
        ai_player_2.train(lineboard.copy(), result[1], result[0])

        if not permuted:
            result_player1.append(result[0])
            result_player2.append(result[1])
        else:
            result_player1_permuted.append(result[1])
            result_player2_permuted.append(result[0])

        ai_player_1.clear()
        ai_player_2.clear()

        if PERMUTATION_MODE and (i+1) % PERMUTATION_FREQ == 0: #evite tour 0
            ai_player_1, ai_player_2 = ai_player_2, ai_player_1
            permuted = not permuted
            #result_player1, result_player2 = result_player2, result_player1


    
    return result_player1, result_player1_permuted

def tournament_multi():
    global ai_player_1
    global other_players

    result_player1 = []
    result_player2 = []

    for i in range(NB_DUEL):
        random.shuffle(other_players)
        ai_player_2 = other_players[0]

        print(f"duel {i} : {ai_player_1.name} vs {ai_player_2.name}\n")
        result = duel(i, ai_player_1, ai_player_2)
        
        
        print(f"résultats : {result[0]} contre {result[1]}")
        if i > 0 and (i+1) % 10 == 0:
            print(f"\n\n\tmoyenne : {sum(result_player1[-10:])/10} contre {sum(result_player2[-10:])/10}\n\n")
        
        ai_player_1.train(lineboard.copy(), result[0], result[1])
        ai_player_2.train(lineboard.copy(), result[1], result[0])

        result_player1.append(result[0])
        result_player2.append(result[1])

        ai_player_1.clear()
        ai_player_2.clear()

    
    return result_player1, result_player2


# ----------------------------------------
#               Main entry
# ----------------------------------------

if __name__ == "__main__":
    global ai_player_1
    global ai_player_2
    global other_players

    #prepare init_board one time only
    for i in range(BOARDSIZE):
        for i2 in range(BOARDSIZE):
            init_board.append( Point(BOARDSIZE * i + i2, i2 * BS + BS, i * BS + BS, []))

    #prepare init_moves_remaining one time only
    for lineIndex in range (BOARDSIZE-1):
        init_moves_remaining.extend((n,n+1) for n in range(BOARDSIZE*lineIndex, BOARDSIZE*(lineIndex+1)-1))
        init_moves_remaining.extend((n,n+BOARDSIZE) for n in range(BOARDSIZE*lineIndex, BOARDSIZE*(lineIndex+1)))
    init_moves_remaining.extend((n,n+1) for n in range(BOARDSIZE*(BOARDSIZE-1), BOARDSIZE*BOARDSIZE-1))

    #prepare init_lineboard one time only
    init_lineboard = [OWNER_NONE for i in range(len(init_moves_remaining))]


    #instantiate class by loading or creating model
    ai_player_1 = import_module(AI_PLAYER_1).Agent(train=TRAIN_MODEL1, moves_list=init_moves_remaining.copy())
    ai_player_2 = import_module(AI_PLAYER_2).Agent(train=TRAIN_MODEL2, moves_list=init_moves_remaining.copy())
    
    # other_players = []
    # for player in OTHERS_PLAYERS:
    #     other_players.append(import_module(player).Agent(moves_list=init_moves_remaining.copy()))


    #start battle
    results = tournament()
    
    #if wanted, save model file
    ai_player_1.save()
    ai_player_2.save()


    #display result graph
    plt.plot([x for x in range(len(results[0]))], results[0], c = 'orange')
    plt.scatter([x for x in range(len(results[0]))], results[0], c = 'red', marker = 'x')

    plt.plot([x for x in range(len(results[1]))], results[1], c = 'blue')
    plt.scatter([x for x in range(len(results[1]))], results[1], c = 'green', marker = 'x')

    plt.title(f"{ai_player_1.name} - position 1 : orange - position 2 : bleue\n{ai_player_2.name} - complément")
    plt.ylim(0,65)
    #plt.show()
    plt.savefig(f"{ai_player_1.name} vs {ai_player_2.name}.png")
    
