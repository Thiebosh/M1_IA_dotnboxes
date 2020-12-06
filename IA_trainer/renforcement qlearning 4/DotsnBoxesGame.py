from collections import namedtuple
import pygame
from pygame import gfxdraw
from time import sleep

class Game:
    BOARDSIZE = 8 #nb de points en ligne et colonne
    BS = 80 #distance entre points

    OWNER_NONE = 0
    OWNER_AI1 = 1
    OWNER_AI2 = 2
    
    Point = namedtuple('Point', ['id', 'x', 'y', 'id_linked_pts'])

    def __init__(self, graphic_mode = True, delay = 0.03):
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

        self.moves_ref = self.init_moves_remaining

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
    #           private functions
    # ----------------------------------------

    def ligne_haut(self, id_ligne):
        return id_ligne - 15 if id_ligne >= 15 else -1

    def ligne_bas(self, id_ligne):
        return id_ligne + 15 if id_ligne <= 96 else -1


    def colonne_gauche(self, id_colonne):
        return id_colonne - 1 if self.moves_ref[id_colonne][0] % 8 != 0 else -1

    def colonne_droite(self, id_colonne):
        return id_colonne + 1 if self.moves_ref[id_colonne][0] % 8 != 7 else -1


    def ligne_haut_gauche(self, id_colonne):
        id2 = self.moves_ref[id_colonne][0]
        id1 = id2-1 if id2 % 8 != 0 else -1
        return self.moves_ref.index((id1, id2)) if id1 != -1 else -1

    def ligne_bas_gauche(self, id_colonne):
        id2 = self.moves_ref[id_colonne][1]
        id1 = id2-1 if id2 % 8 != 0 else -1 #pb here
        return self.moves_ref.index((id1, id2)) if id1 != -1 else -1

    def ligne_haut_droite(self, id_colonne):
        id1 = self.moves_ref[id_colonne][0]
        id2 = id1+1 if id1 % 8 != 7 else -1
        return self.moves_ref.index((id1, id2)) if id2 != -1 else -1

    def ligne_bas_droite(self, id_colonne):
        id1 = self.moves_ref[id_colonne][1]
        id2 = id1+1 if id1 % 8 != 7 else -1
        return self.moves_ref.index((id1, id2)) if id2 != -1 else -1


    def colonne_haut_gauche(self, id_ligne):
        id2 = self.moves_ref[id_ligne][0]
        id1 = id2-8 if id2 > 7 else -1
        return self.moves_ref.index((id1, id2)) if id1 != -1 else -1

    def colonne_haut_droite(self, id_ligne):
        id2 = self.moves_ref[id_ligne][1]
        id1 = id2-8 if id2 > 7 else -1
        return self.moves_ref.index((id1, id2)) if id1 != -1 else -1

    def colonne_bas_gauche(self, id_ligne):
        id1 = self.moves_ref[id_ligne][0]
        id2 = id1+8 if id1 < 56 else -1
        return self.moves_ref.index((id1, id2)) if id2 != -1 else -1

    def colonne_bas_droite(self, id_ligne):
        id1 = self.moves_ref[id_ligne][1]
        id2 = id1+8 if id1 < 56 else -1
        return self.moves_ref.index((id1, id2)) if id2 != -1 else -1


    def free_ticks_in_neighbours_boxes(self, id_tick):
        free_tick_in_box1 = []
        free_tick_in_box2 = []
        box_out = [False, False]

        # (0,1,2,3,4,5,6), 
        # (7,22,37,52,67,82,97), 
        # (14,29,44,59,74,89,104), 
        # (105,106,107,108,109,110,111)

        if self.moves_ref[id_tick][0] + 1 == self.moves_ref[id_tick][1]: #trait est ligne

            if self.ligne_haut(id_tick) == -1: #-1 si extérieur
                box_out[0] = True
            else:
                for tmp in [self.ligne_haut(id_tick), self.colonne_haut_gauche(id_tick), self.colonne_haut_droite(id_tick)]:
                    if self.moves_ref[tmp] in self.moves_remaining : # libre
                        free_tick_in_box1.append(tmp)
            
            if self.ligne_bas(id_tick) == -1: #-1 si extérieur
                box_out[1] = True
            else:
                for tmp in [self.ligne_bas(id_tick), self.colonne_bas_gauche(id_tick), self.colonne_bas_droite(id_tick)]:
                    if self.moves_ref[tmp] in self.moves_remaining : # libre
                        free_tick_in_box2.append(tmp)

        else: #trait est colonne
            
            if self.colonne_gauche(id_tick) == -1: #-1 si extérieur
                box_out[0] = True
            else:
                for tmp in [self.colonne_gauche(id_tick), self.ligne_haut_gauche(id_tick), self.ligne_bas_gauche(id_tick)]:
                    if self.moves_ref[tmp] in self.moves_remaining : # libre
                        free_tick_in_box1.append(tmp)
            
            if self.colonne_droite(id_tick) == -1: #-1 si extérieur
                box_out[1] = True
            else:
                for tmp in [self.colonne_droite(id_tick), self.ligne_haut_droite(id_tick), self.ligne_bas_droite(id_tick)]:
                    if self.moves_ref[tmp] in self.moves_remaining : # libre
                        free_tick_in_box2.append(tmp)

        return (free_tick_in_box1, free_tick_in_box2), box_out


    def is_set_connected(self, point1, point2): #identifiants
        return True if (point1, point2) in self.moves_done or (point2, point1) in self.moves_done else False


    def is_move_closing_box(self, point1, point2):
        is_box = False
        bonus = 0
        
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

        return is_box#, bonus


    def apply_move(self, point1, point2):
        self.board[point1].id_linked_pts.append(point2)
        self.board[point2].id_linked_pts.append(point1)
        self.moves_done.append((point1, point2))

        self.lineboard[self.init_moves_remaining.index((point1, point2))] = self.OWNER_AI1 if self.is_AI1_turn else self.OWNER_AI2

    # ----------------------------------------
    #           public functions
    # ----------------------------------------

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

        return self.lineboard.copy()


    def get_id_moves_remaining(self):
        return [self.moves_ref.index(move_tuple) for move_tuple in self.moves_remaining]


    def get_moves_ref(self):
        return self.moves_ref


    def disp_board(self):
        myfont = pygame.font.SysFont('Arial', 50)
        score_font = pygame.font.SysFont('Arial', 30)
        
        LINE_THICKNESS = 15 #epaisseur traits
        DOT_THICKNESS = 7
        size = self.BOARDSIZE * self.BS + self.BS #taille contenu fenêtre
        SURF = pygame.display.set_mode((size, size)) #surface fenêtre
        pygame.display.set_caption("Dots and  Boxes")
        
        BLACK = (10, 10, 10)
        WHITE = (200, 200, 200)
        RED = (255, 128, 0)
        BLUE = (0, 0, 255)


        SURF.fill(BLACK)

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
        bonus = score_font.render("7", True, (100,100,100))
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
        for event in pygame.event.get(): #change to try except if possible
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        sleep(self.delay)


    def decide_and_move(self, id_chosen_move):
        # if not self.is_AI1_turn:

        #     return 

        id_pt1, id_pt2 = self.moves_ref[id_chosen_move]

        self.moves_remaining.remove((id_pt1,id_pt2))

        self.apply_move(id_pt1, id_pt2) #met plateau à jour
        #is_box_closed, reward = self.is_move_closing_box(id_pt1, id_pt2) #met score à jour
        is_box_closed = self.is_move_closing_box(id_pt1, id_pt2) #met score à jour

        is_player1 = self.is_AI1_turn

        #if close a square, play again
        if not is_box_closed: #si n'a pas fermé, tour de l'autre joueur
            self.is_AI1_turn = not self.is_AI1_turn

        return is_player1, self.lineboard.copy(), len(self.moves_remaining) == 0


    def nb_boxes_closable(self):
        closable_boxes = 0

        for tick in self.moves_remaining:
            nb_neighbours, out = self.free_ticks_in_neighbours_boxes(self.moves_ref.index(tick))

            for i in range (2):
                if not out[i] and len(nb_neighbours[i]) == 0:
                    closable_boxes += 1
    
        return closable_boxes

    
    def log_results(self):
        print(f"scores de la partie : {self.score[0]} à {self.score[1]}")
        return self.score[0], self.score[1]
