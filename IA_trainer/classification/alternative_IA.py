def train():
    data = pd.read_csv("dataframe_parties.csv", index_col=0)

    data["board"] = data["board"].apply(lambda x: literal_eval(x))
    data["board"] = data["board"].apply(lambda x: np.array(x))
    # data["best_move"] = data["best_move"].apply(lambda x: np.array(x))

    X = np.asarray([x for x in data["board"]])
    y = to_categorical(np.asarray([x for x in data["best_move"]]))

    self.model.fit(X, y, batch_size=64, epochs=200, shuffle=True, verbose=1)
    