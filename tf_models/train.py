from tf_models.utils import train, save_model


def train_and_save(name: str, corpus: str, pos_label: str, root: str = ""):
    print("Start training {}...".format(name))

    mlp_model, _, vec = train(corpus, pos_label, root)
    save_model(mlp_model, vec, name, root)


if __name__ == "__main__":
    # Train intent model
    # fmt: off
    train_and_save(
        name="intent",
        corpus="intent_corpus.csv",
        pos_label="weather",
        root="intent"
    )
    # fmt: on

    # Train flow control model
    train_and_save(
        name="flow_control",
        corpus="flow_control_corpus.csv",
        pos_label="continue",
        root="flow_control",
    )
