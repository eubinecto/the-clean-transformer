import wandb
from Korpora import KoreanParallelKOENNewsKorpus
from cleanformer.paths import KORPORA_DIR


def main():
    korpus = KoreanParallelKOENNewsKorpus(root_dir=str(KORPORA_DIR))
    train = list(zip(korpus.train.texts, korpus.train.pairs))
    val = list(zip(korpus.dev.texts, korpus.dev.pairs))
    test = list(zip(korpus.test.texts, korpus.test.pairs))
    train = wandb.Table(columns=["kor", "eng"], data=train)
    val = wandb.Table(columns=["kor", "eng"], data=val)
    test = wandb.Table(columns=["kor", "eng"], data=test)
    with wandb.init(project="cleanformer", tags=[__file__]) as run:
        # save to local, and then to wandb
        artifact = wandb.Artifact(name="kor2eng", type="dataset", metadata={"desc": korpus.description})
        artifact.add(train, "train")
        artifact.add(val, "val")
        artifact.add(test, "test")
        run.log_artifact(artifact)


if __name__ == '__main__':
    main()
