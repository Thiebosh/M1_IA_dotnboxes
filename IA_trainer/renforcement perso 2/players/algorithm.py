import random as rd
import pandas as pd

OWNER_NONE = 0
MIN_ID_LINE = 0
MAX_ID_LINE = 112

class Agent:

	def __init__(self, train, moves_list):
		self.name = "algorithm" # put your name here

		self.chain = []
	
		self.moves_ref = moves_list
		self.high_score_boxes = {
			7 : (48, 55, 56, 63),
			3 : (16, 23, 24, 31,
				20, 27, 28, 35,
				76, 83, 84, 91,
				80, 87, 88, 95)
		}

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


	def free_ticks_in_neighbours_boxes(self, id_tick, moves_remaining):
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

		return (free_tick_in_box1, free_tick_in_box2), box_out


	def recursive_close_boxes(self, id_tick, free_ticks_by_neighbours, moves_remaining, chain):
		if id_tick in self.high_score_boxes[7]:
			box_value = 14
		elif id_tick in self.high_score_boxes[3]:
			box_value = 6
		else:
			box_value = 2

		chain.append(id_tick)
		moves_remaining.remove(self.moves_ref[id_tick])

		if len(free_ticks_by_neighbours[0]) + len(free_ticks_by_neighbours[1]) == 1:
			new_tick = free_ticks_by_neighbours[0 if len(free_ticks_by_neighbours[0]) == 1 else 1][0]
			new_free_ticks_by_neighbours = self.free_ticks_in_neighbours_boxes(new_tick, moves_remaining)[0]
			
			return box_value + self.recursive_close_boxes(new_tick, new_free_ticks_by_neighbours, moves_remaining, chain)

		else:
			return box_value


	def evaluate_tick_value(self, id_tick, moves_remaining):
		#calculer nombre de trait libres dans chacune des deux boites auxquelles il appartient
		free_tick_in_boxes, box_out = self.free_ticks_in_neighbours_boxes(id_tick, moves_remaining)

		value_tick = 0
		close_box = False
		value_next_play = 0
		chainable = False
		
		for i, id_free in enumerate(free_tick_in_boxes): #0, 1, 2 ou 3 traits
			if box_out[i]:
				continue

			nb_free = len(id_free)

			if nb_free == 0: #ferme la boite
				close_box = True

				if id_tick in self.high_score_boxes[7]:
					value_tick += 14.0
				elif id_tick in self.high_score_boxes[3]:
					value_tick += 6.0
				else:
					value_tick += 2.0

			elif nb_free == 1: #laisse cuvette (si rejoue, top)
				
				if id_tick in self.high_score_boxes[7] and id_free[0] in self.high_score_boxes[7]:
					value_next_play = 6.9
				elif id_tick in self.high_score_boxes[3] and id_free[0] in self.high_score_boxes[3]:
					value_next_play = 2.9
				else:
					value_next_play = 0.9

				value_tick -= value_next_play

			else: #laisse angle ou entame boite
				if id_tick in self.high_score_boxes[7] and id_free[0] in self.high_score_boxes[7]:
					value_tick += 0.7
				elif id_tick in self.high_score_boxes[3] and id_free[0] in self.high_score_boxes[3]:
					value_tick += 0.3
				else:
					value_tick += 0.1
					
		#evalue strat d'ensemble (enchainement de coups)
		chain = []
		if close_box and value_next_play != 0:
			#value_tick += 2 * value_next_play #ancienne methode pour pressentir chaine sans l'étudier
			value_tick = self.recursive_close_boxes(id_tick, free_tick_in_boxes, moves_remaining.copy(), chain)

		return value_tick, chain

	
	def play(self, lineboard, moves_remaining):
	# 	move = self.playy(lineboard, moves_remaining)
	# 	input(move)
	# 	return move

	# def playy(self, lineboard, moves_remaining):
		# if rd.random() < self.alea: #de temps en temps, fait move aléatoire pour ne pas etre trop fort
		# 	return rd.choice(moves_remaining)

		if len(self.chain) != 0:
			return self.moves_ref[self.chain.pop(0)]

		best_value = -10.0 #ramasse tout
		best_ticks = []
		chained_moves = []

		#analyse chaque coup
		for id_tick in [self.moves_ref.index(x) for x in moves_remaining]:
			#associer valeur au trait
			value_tick, chain = self.evaluate_tick_value(id_tick, moves_remaining)

			#enregistrer traits les plus intéressants
			if value_tick == best_value:
				best_ticks.append(id_tick)
				if len(chain) != 0:
					chained_moves.append((id_tick, chain))
			elif value_tick > best_value:
				best_value = value_tick
				best_ticks.clear()
				best_ticks.append(id_tick)
				if len(chain) != 0:
					chained_moves.clear()
					chained_moves.append((id_tick, chain))

		move = rd.choice(best_ticks)

		for couple in chained_moves:
			if couple[0] == move:
				self.chain = couple[1]
				del self.chain[0]
				break
		
		#retourne un des meilleurs coups
		return self.moves_ref[move]
