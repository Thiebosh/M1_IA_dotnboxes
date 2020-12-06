import random as rd
import pandas as pd

OWNER_NONE = 0
MIN_ID_LINE = 0
MAX_ID_LINE = 112

class Agent:

	def __init__(self, train, moves_list):
		self.name = "algorithm easy" # put your name here

		self.alea = 0.3
		
		self.moves_ref = moves_list


	def save(self):
		pass
	
	def play_result(self, ai_score, other_score, nb_moves_chained, remaining_moves_before):
		pass

	def clear(self):
		pass

	def train(self, lineboard, ai_score: int, other_score: int):
		pass

	def other_play(self, lineboard, other_move, ai_score, other_score, nb_moves_chained, remaining_moves_before):
		pass
	
	
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




	def play(self, lineboard, moves_remaining):
	# 	move = self.playy(lineboard, moves_remaining)
	# 	input(move)
	# 	return move

	# def playy(self, lineboard, moves_remaining):
		if rd.random() < self.alea: #de temps en temps, fait move aléatoire pour ne pas etre trop fort
			return rd.choice(moves_remaining)


		#analyse chaque coup
		for id_tick in [self.moves_ref.index(x) for x in moves_remaining]:

			#calculer nombre de trait libres dans chacune des deux boites auxquelles il appartient
			free_tick_in_box1 = []
			free_tick_in_box2 = []
			box_out = [False, False]

			if self.moves_ref[id_tick][0] + 1 == self.moves_ref[id_tick][1]: #trait est ligne

				if self.ligne_haut(id_tick) == -1: #-1 si extérieur
					box_out[0] = True
				else:
					for tmp in [self.ligne_haut(id_tick), self.colonne_haut_gauche(id_tick), self.colonne_haut_droite(id_tick)]:
						if self.moves_ref[tmp] in moves_remaining : # libre
							free_tick_in_box1.append(tmp)
				
				if self.ligne_bas(id_tick) == -1: #-1 si extérieur
					box_out[1] = True
				else:
					for tmp in [self.ligne_bas(id_tick), self.colonne_bas_gauche(id_tick), self.colonne_bas_droite(id_tick)]:
						if self.moves_ref[tmp] in moves_remaining : # libre
							free_tick_in_box2.append(tmp)

			else: #trait est colonne
				
				if self.colonne_gauche(id_tick) == -1: #-1 si extérieur
					box_out[0] = True
				else:
					for tmp in [self.colonne_gauche(id_tick), self.ligne_haut_gauche(id_tick), self.ligne_bas_gauche(id_tick)]:
						if self.moves_ref[tmp] in moves_remaining : # libre
							free_tick_in_box1.append(tmp)
				
				if self.colonne_droite(id_tick) == -1: #-1 si extérieur
					box_out[1] = True
				else:
					for tmp in [self.colonne_droite(id_tick), self.ligne_haut_droite(id_tick), self.ligne_bas_droite(id_tick)]:
						if self.moves_ref[tmp] in moves_remaining : # libre
							free_tick_in_box2.append(tmp)

			if len(free_tick_in_box1) == 0 and not box_out[0] or len(free_tick_in_box2) == 0 and not box_out[1]:
				return self.moves_ref[id_tick]


		#sinon, aléatoire
		return rd.choice(moves_remaining)

