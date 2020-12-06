# ----------------------------------------
#            Import librairies
# ----------------------------------------

import sys
import time
from collections import namedtuple
import random

import pygame
from pygame import gfxdraw
from time import sleep


# ----------------------------------------
#            Global variables to define
# ----------------------------------------

class Game:
    BOARDSIZE = 8 #nb de points en ligne et colonne
    BS = 80 #distance entre points

    OWNER_NONE = 0
    OWNER_AI1 = 1
    OWNER_AI2 = 2
    
    Point = namedtuple('Point', ['id', 'x', 'y', 'id_linked_pts'])

    def __init__(self, nb_duel = 1, graphic_mode = True, delay = 0.03):
        self.nb_duel = nb_duel

        self.graphic_mode = graphic_mode
        self.delay = delay


        self.init_board = []
        self.init_lineboard = []
        self.init_moves_remaining = []

        #prepare init_board one time only
        for i in range(self.BOARDSIZE):
            for i2 in range(self.BOARDSIZE):
                self.init_board.append( self.Point(self.BOARDSIZE * i + i2, i2 * self.BS + self.BS, i * self.BS + self.BS, []))

        #prepare init_moves_remaining one time only
        for lineIndex in range (self.BOARDSIZE-1):
            self.init_moves_remaining.extend((n,n+1) for n in range(self.BOARDSIZE*lineIndex, self.BOARDSIZE*(lineIndex+1)-1))
            self.init_moves_remaining.extend((n,n+self.BOARDSIZE) for n in range(self.BOARDSIZE*lineIndex, self.BOARDSIZE*(lineIndex+1)))
        self.init_moves_remaining.extend((n,n+1) for n in range(self.BOARDSIZE*(self.BOARDSIZE-1), self.BOARDSIZE*self.BOARDSIZE-1))

        #prepare init_lineboard one time only
        self.init_lineboard = [self.OWNER_NONE for i in range(len(self.init_moves_remaining))]


        self.score = [0, 0] # AI1, AI2
        self.is_AI1_turn = True


        self.board = [] #[id_point, coord x, coord y, id_linked_pts]
        self.boxes = [] #[id haut gauche, id haut droite, id bas gauche, id bas droite, possesseur]
        self.lineboard = []
        self.moves_done = []
        self.moves_remaining = []


        if self.graphic_mode:
            pygame.init() #load modules during loading functions
            pygame.font.init()


    # ----------------------------------------
    #           General functions
    # ----------------------------------------
            
    def disp_board(self):
        myfont = pygame.font.SysFont('Arial', 50)
        score_font = pygame.font.SysFont('Arial', 30)
        
        LINE_THICKNESS = 15 #epaisseur traits
        DOT_THICKNESS = 7
        size = self.BOARDSIZE * self.BS + self.BS #taille contenu fenêtre
        SURF = pygame.display.set_mode((size, size)) #surface fenêtre
        pygame.display.set_caption("Dots and  Boxes")
        
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 128, 0)
        BLUE = (0, 0, 255)


        SURF.fill(WHITE)

        # first lets draw the score at the top
        score_AI1 = score_font.render("{}".format(self.score[0]), True, BLUE)
        w = score_font.size("{}".format(self.score[0]))
        SURF.blit(score_AI1, (size // 2 - w[0] - 10, 10))
        
        score_AI2 = score_font.render("{}".format(self.score[1]), True, RED)
        SURF.blit(score_AI2, (size // 2 + 10, 10))

        # then, draw aeras
        for box in self.boxes:
            x1 = self.board[box[0]].x
            y1 = self.board[box[0]].y
            
            if box[4] == self.OWNER_AI1:
                pygame.draw.rect(SURF,(150,150,255),(x1,y1,self.BS,self.BS))
            elif box[4] == self.OWNER_AI2:
                pygame.draw.rect(SURF,(255,180,180),(x1,y1,self.BS,self.BS))

        # ahead, draw ticks with good color
        for i, player in enumerate(self.lineboard):
            if player != self.OWNER_NONE:
                point1, point2 = self.init_moves_remaining[i]
                point1 = self.board[point1]
                point2 = self.board[point2]
                pygame.draw.line(SURF, BLUE if player == self.OWNER_AI1 else RED, (point1.x, point1.y), (point2.x, point2.y), LINE_THICKNESS)

        # on top, draw dots
        for i, point in enumerate(self.board):
            gfxdraw.filled_circle(SURF, point.x, point.y, DOT_THICKNESS, BLACK)
            

        #display '7' in the middle
        x1 = self.board[27].x
        y1 = self.board[27].y
        bonus = score_font.render("7", True, (50,50,50))
        text_width, text_height = myfont.size("7")
        SURF.blit(bonus, (int(x1 + 50 - text_width / 2), int(y1 + 50 - text_height / 2)))

        #display '3' in the squares
        bonus = score_font.render("3", True, (100,100,100))
        text_width, text_height = myfont.size("3")
        for i in [9, 13, 41, 45]:
            x1 = self.board[i].x
            y1 = self.board[i].y
            SURF.blit(bonus, (int(x1 + 50 - text_width / 2), int(y1 + 50 - text_height / 2)))

        pygame.display.update()


    def init_game(self):
        self.board.clear()
        self.boxes.clear()
        self.lineboard.clear()
        self.moves_done.clear()
        self.moves_remaining.clear()

        self.board = self.init_board.copy()
        self.lineboard = self.init_lineboard.copy()
        self.moves_remaining = self.init_moves_remaining.copy()
        
        self.score[0] = 0
        self.score[1] = 0

        #[id haut gauche, id haut droite, id bas gauche, id bas droite, possesseur] donc ne peut pas faire range 0,55
        self.boxes.extend([[i, i+1, i+self.BOARDSIZE, i+1+self.BOARDSIZE, self.OWNER_NONE] for i in range(0,7)])
        self.boxes.extend([[i, i+1, i+self.BOARDSIZE, i+1+self.BOARDSIZE, self.OWNER_NONE] for i in range(8,15)])
        self.boxes.extend([[i, i+1, i+self.BOARDSIZE, i+1+self.BOARDSIZE, self.OWNER_NONE] for i in range(16,23)])
        self.boxes.extend([[i, i+1, i+self.BOARDSIZE, i+1+self.BOARDSIZE, self.OWNER_NONE] for i in range(24,31)])
        self.boxes.extend([[i, i+1, i+self.BOARDSIZE, i+1+self.BOARDSIZE, self.OWNER_NONE] for i in range(32,39)])
        self.boxes.extend([[i, i+1, i+self.BOARDSIZE, i+1+self.BOARDSIZE, self.OWNER_NONE] for i in range(40,47)])
        self.boxes.extend([[i, i+1, i+self.BOARDSIZE, i+1+self.BOARDSIZE, self.OWNER_NONE] for i in range(48,55)])

    def is_set_connected(self, point1, point2): #identifiants
        return True if (point1, point2) in self.moves_done or (point2, point1) in self.moves_done else False
        
    def is_move_closing_box(self, point1, point2):
        is_box = False
        
        for box in [item for item in self.boxes if (point1 in item and point2 in item)]:
            i = self.boxes.index(box)

            tmp = list(box) #necessaire pour ne pas detruire boxes
            tmp.remove(point1)
            tmp.remove(point2)

            if  (self.is_set_connected(tmp[0], tmp[1]) and 
                    ((self.is_set_connected(point1, tmp[0]) and self.is_set_connected(point2, tmp[1])) or
                    (self.is_set_connected(point1, tmp[1]) and self.is_set_connected(point2, tmp[0])))):

                is_box = True #return pas car peut fermer plusieurs boites
                bonus = 1

                if i == 24: # id square bonus 7
                    bonus = 7
                elif i in [8, 12, 36, 40]: # id squares bonus 7
                    bonus = 3

                #applique gain
                if self.is_AI1_turn:
                    self.score[0] += bonus
                    self.boxes[i][4] = self.OWNER_AI1
                else:
                    self.score[1] += bonus
                    self.boxes[i][4] = self.OWNER_AI2

        return is_box

    def apply_move(self, point1, point2):
        self.board[point1].id_linked_pts.append(point2)
        self.board[point2].id_linked_pts.append(point1)
        self.moves_done.append((point1, point2))

        self.lineboard[self.init_moves_remaining.index((point1, point2))] = self.OWNER_AI1 if self.is_AI1_turn else self.OWNER_AI2

    # ----------------------------------------
    #           Player call functions
    # ----------------------------------------

    def decide_and_move(self, ai_player):
        ai_choice = ai_player.play(self.moves_remaining)

        id_pt1 = ai_choice[0]
        id_pt2 = ai_choice[1]

        #verify that move is correct
        if (id_pt1,id_pt2) in self.moves_remaining:
            self.moves_remaining.remove((id_pt1,id_pt2))
        elif (id_pt2,id_pt1) in self.moves_remaining:
            self.moves_remaining.remove((id_pt2,id_pt1))
            id_pt1, id_pt2 = id_pt2, id_pt1 #swap
        else:
            raise NameError('invalid move')

        self.apply_move(id_pt1, id_pt2) #met plateau à jour
        is_box_closed = self.is_move_closing_box(id_pt1, id_pt2) #met score à jour


        if self.graphic_mode:
            sleep(self.delay)
            self.disp_board()


        #if close a square, play again
        if is_box_closed:
            if len(self.moves_remaining) != 0: #fin de jeu
                self.decide_and_move(ai_player)
            else:
                return True

        #if it remains moves...
        return len(self.moves_remaining) == 0


    def duel(self, current_duel, player1, player2):
        ended = False

        self.init_game()

        while not ended:
            if self.graphic_mode:
                for event in pygame.event.get(): #change to try except if possible
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            self.is_AI1_turn = True
            ended = self.decide_and_move(player1)
            
            if not ended:
                self.is_AI1_turn = False
                ended = self.decide_and_move(player2)

        if self.graphic_mode:
            self.disp_board()


    def tournament(self, ai_player_1, ai_player_2):
        for i in range(self.nb_duel):
            print(f"\nduel {i} - {ai_player_1.name} vs {ai_player_2.name}")
            self.duel(i, ai_player_1, ai_player_2)

            result = self.score
            print(f"résultats : {ai_player_1.name if result[0] > result[1] else ai_player_2.name} gagnant - {result[0]} contre {result[1]}")
            

# ----------------------------------------
#               Main entry
# ----------------------------------------

if __name__ == "__main__":
    #start battle
    
    env = Game(nb_duel = 1, graphic_mode = True, delay = 0.03)
    env.tournament(import_module("bender").Agent(), import_module("bender").Agent())
    