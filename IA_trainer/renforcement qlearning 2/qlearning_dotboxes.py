############################
#       Setup
############################

import DotsnBoxesGame as env #environnement du jeu

from math import log
from random import choice
from os import path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

from datetime import date
import matplotlib.pyplot as plt


GRAPHIC_MODE = False
env = env.Game(graphic_mode = GRAPHIC_MODE, delay = 0.01)

IS_PLAYER1 = True

# Configuration paramaters for the whole setup
NB_ROUND_MAX = 500
REWARD_ENOUGH = 100 # max 154 points => entrainer jusque 100 pour commencer

gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
batch_size = 64 # Size of batch taken from replay buffer

version_model = 2
model_storage = "models/qlearner"+str(version_model)+".h5"


############################
#       Implement the Deep Q-Network
############################

num_actions = 112#len(env.get_id_moves_remaining()) #112

def create_q_model():
	inputs = layers.Input(shape=(num_actions,), name='inputs')

	x = layers.Dense(num_actions*2, activation='relu', kernel_initializer='glorot_uniform')(inputs)
	x = layers.Dense(num_actions*4, activation='relu', kernel_initializer='glorot_uniform')(x)
	x = layers.Dense(num_actions*8, activation='relu', kernel_initializer='glorot_uniform')(x)
	x = layers.Dense(num_actions*4, activation='relu', kernel_initializer='glorot_uniform')(x)
	x = layers.Dense(num_actions*2, activation='relu', kernel_initializer='glorot_uniform')(x)
	
	action = layers.Dense(num_actions, activation='linear', kernel_initializer='glorot_uniform', name='q_values')(x)
	
	return keras.Model(inputs = inputs, outputs = action)


# The first model makes the predictions for Q-values which are used to make a action.
#model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
#model_target = create_q_model()

if path.isfile(model_storage):
    print("\nload models")
    model = load_model(model_storage)
    model_target = load_model(model_storage)    
else:
    print("\ncreate models")
    model = create_q_model()
    model_target = create_q_model()
print("\n")

############################
#       Train
############################

# In the Deepmind paper they use RMSProp however then Adam optimizer improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.0025, clipnorm=1.0)

# Experience replay buffers
action_history = [] #coup choisi pour les max 100000 derniers coups
board_history = [] #plateau avant coup pour les max 100000 derniers coups
board_next_history = [] #plateau après coup pour les max 100000 derniers coups
rewards_history = [] #récompense du coup pour les max 100000 derniers coups
end_game_history = [] #coup terminal ou non pour les max 100000 derniers coups
game_reward_history = [] #récompense totale des max 100 dernières partie
running_reward = 0 #moyenne du total des récompenses des max 100 dernières partie
move_count = 0 #nombre de coups, toutes parties confondues
# Number of moves to take random action and observe output
epsilon_random_moves = 5000 #50000 #nombre de coups randoms max sur lesquels apprendre avant de laisser l'ia proposer ses coups
# Number of moves for exploration
epsilon_greedy_moves = 100000.0 #1000000.0 #réducteur de epsilon : plus il est haut, moins vite l'ia pourra tester ses coups
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 10000 #100000 #temps avant vidange des _history (excepté game_reward_history)
# Train the model after 4 actions
update_after_actions = 4 #ne modifie pas le modèle à chaque coup
# How often to update the target network
update_target_network = 1000 #10000 #modifie encore moins souvent le second modele (stabilisteur, chargé d'estimer les récompenses à coup + 2)
# Using huber loss for stability
#loss_function = keras.losses.MeanSquaredError()
loss_function = keras.losses.Huber()

reward_evolution = []

def compute_ai_reward(reward_move, reward_available):
    # penality si points : 0 si aucune, 1 si 63
    # penality si pas points : 1 si aucune, 0 si 63
    # return : 1,2,4,6,8,14,16 - penality
    return (2 * reward_move if reward_move > 0 else 1) - log(1 + reward_available / 7)
    

def normalize_lineboard(lineboard):
    return np.array([int(x != env.OWNER_NONE) for x in lineboard])


for game_count in range(NB_ROUND_MAX): #nb jeu max
    print(f"\n\nGame {game_count} :")

    lineboard = env.init_game()
    game_reward = 0 #somme des récompenses de la partie

    if GRAPHIC_MODE:
        env.disp_board()

    while True: #112 coups / jeu
        if move_count < epsilon_random_moves or epsilon > np.random.rand(1)[0]: #premieres parties sont random, par la suite random se rarifie
            action = choice(env.get_id_moves_remaining())
        else: #prediction du premier modele
            # Predict action Q-values from environment board
            board_tensor = np.array([int(x != env.OWNER_NONE) for x in lineboard]).reshape(1,-1)
            #board_tensor = tf.expand_dims(tf.convert_to_tensor(board), 0) #normalise l'input (image) pour le modele
            action_probs = np.array(model(board_tensor, training=False)[0]) #pas de .predict?
            # discard wrong moves
            action_probs[[i for i in range(len(action_probs)) if lineboard[i] != env.OWNER_NONE]] = -99999
            # Take best action
            action = np.argmax(action_probs) #plus haute valeur

        # Decay probability of taking random action
        epsilon = max(epsilon_min, epsilon - epsilon_interval / epsilon_greedy_moves) #reduit aléatoire jusque valeur min 

        # Apply the sampled action in our environment        
        is_player1, lineboard_next, reward, end_game = env.decide_and_move(action)
        
        if GRAPHIC_MODE:
            env.disp_board()
        
        if is_player1 == IS_PLAYER1:
            move_count += 1 #incremente nb de coups joués

            reward = compute_ai_reward(reward, env.nb_boxes_closable())

            game_reward += reward #cumule récompense des coups de la partie

            # Save actions and boards in replay buffer
            action_history.append(action)
            board_history.append(normalize_lineboard(lineboard))
            board_next_history.append(normalize_lineboard(lineboard_next))
            rewards_history.append(reward)
            end_game_history.append(end_game)

        
            # Update every fourth move and once batch size is over 32
            if move_count % update_after_actions == 0 and len(end_game_history) > batch_size: #si assez de données, tous les 4 coups, met à jour modèle 1

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(end_game_history)), size=batch_size)#sélectionne échantillon au hasard

                # Using list comprehension to sample from replay buffer
                board_sample = np.array([board_history[i] for i in indices])
                board_next_sample = np.array([board_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                end_game_sample = tf.convert_to_tensor([float(end_game_history[i]) for i in indices])

                # Build the updated Q-values for the sampled future boards
                # Use the target model for stability
                future_rewards = model_target.predict(board_next_sample)#, batch_size=batch_size) #prédit avec modèle 2 récompenses à plateau + 2
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1) #pondère récompenses à plateau + 1 avec plus haute récompense de plateau + 2

                # If final move set the last value to -1
                # if end_game_sample: quelque chose ?
                updated_q_values = updated_q_values * (1 - end_game_sample) - end_game_sample #end_game : booléen donc 0 => updated_q_values et 1 => -1. à voir si on adapte pas

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)#crée une matrice de action_sample par 4 = num_actions... ?
                
                with tf.GradientTape() as tape: #parcours chaque gradient comme un fichier?
                    # Train the model on the boards and updated Q-values
                    q_values = model(board_sample) #récupère qvalues pour l'état du plateau
                    
                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1) #somme matricielle de multiplication matricielle de whaaat?
                
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action) #utilise keras.losses.Huber avec y pred = qvalue du modele 1 sur l'état actuel du plateau et y true = qvalue du modele 2 sur l'état suivant du plateau ????

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables) #utilise tape en dehors de la boucle précédente??? pour caluler descente de gradient des variables du modèle 1 = ses poids neuronaux?
                optimizer.apply_gradients(zip(grads, model.trainable_variables)) #keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0) pour appliquer descente de gradient précédemment claculée à travers l'optimizer ?


            if move_count % update_target_network == 0: #tous les 10000 coups, met à jour modèle 2
                # update the the target network with new weights
                model_target.set_weights(model.get_weights()) #met à jour le modèle 2
                # Log details
                print(f"running reward: {running_reward:.2f} at game {game_count}, move count {move_count}")


            # Limit the lineboard and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[0]
                del board_history[0]
                del board_next_history[0]
                del action_history[0]
                del end_game_history[0]

        if end_game:
            env.log_results()
            print(f"game reward : {game_reward:.2f}")
            break

        lineboard = lineboard_next #actualise plateau en mémoire

    if GRAPHIC_MODE:
        env.disp_board()

    ### vérifie qualité de l'ia via ses récompenses : quand suffisamment forte, met fin à l'entrainement
        #dans notre cas, remplacer par % de victoire ?
    # Update running reward to check condition for solving
    game_reward_history.append(game_reward)
    if len(game_reward_history) > NB_ROUND_MAX/10:
        del game_reward_history[0]
    running_reward = np.mean(game_reward_history)

    reward_evolution.append(running_reward)

    game_count += 1

    print(f"End of game - mean level : {running_reward:.3f}")

    if running_reward > REWARD_ENOUGH:
        print("Solved!")
        break


# save progression
model.save(model_storage)


#display result graph
plt.plot([x for x in range(len(reward_evolution))], reward_evolution, c = 'orange')
plt.scatter([x for x in range(len(reward_evolution))], reward_evolution, c = 'red', marker = 'x')

plt.title("Evolution de la récompense moyenne")
plt.ylim(0,154)

plt.savefig(f"training{version_model}.png")
