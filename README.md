# M1_S1_IA_dotnboxes

approches essayées :

qlearning - renforcement sur un choix parmi 112
- avec 1 modèle
- avec 2 modèles (+ de stabilité)
- contre un autre adversaire
- contre l'ia elle même (l'ia appelle le jeu et plus l'inverse)
- avec couches dense
- avec couches lstm
- toute sorte de reward

... peu concluant


classification - génération de 10 000 parties avec un excellent algorithme et entrainement de l'ia pour prédire le coup à partir du plateau

... peu concluant


renforcement sur l'ensemble du plateau
- prédit l'intérêt de chaque coup, choisit le meilleur

... bonne base
- corrections algorithmiques quand nécessaire


essayer couches convolutionnelles ?
- input : matrice 8*8 des points, nombre de voisins en valeur
- kernel : 3*3 pour prendre les voisins directs
- output : 112 traits du plateau et softmax pour avoir trait avec argmax
- reward : les 112 traits avec une valeur entre 0 (trait occupé) et 1 (trait qui rapportera le plus de points)
- hyperparamètres : sans doute learning rate à 0.005, 100 parties pour commencer. Pas de callback pour commencer
- loss : mean_squared_error ou categorical_crossentropy
- optimizer : Adam ou Huber
