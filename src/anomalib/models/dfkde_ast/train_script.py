from pathlib import Path

import torch
from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters

from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks, LoadModelCallback
import pandas as pd


def get_annotated_train_df(all_df, test_df):
    test_names = test_df["names"].tolist()
    df_train = all_df[~all_df["names"].isin(test_names)]
    df_train = df_train[df_train["is_annotated"] == 1]
    df_train_only_blank = df_train[df_train["annotation_label"] == 0]
    df_train_only_whoop = df_train[df_train["annotation_label"] == 1]
    return df_train_only_blank, df_train_only_whoop

def fit():

    CONFIG_PATH = "config.yaml"
    # with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as file:
    #     print(file.read())

    # pass the config file to model, callbacks and datamodule
    config = get_configurable_parameters(config_path=CONFIG_PATH)

    # datamodule = get_datamodule(config)
    # datamodule.prepare_data()  # Downloads the dataset if it's not in the specified `root` directory
    # datamodule.setup()  # Create train/val/test/prediction sets.
    from data_ast import CustomDataModule

    data_dir = "/Users/christof/Documents/papers/sound/audio_data_simulation/ast_features/30_seconds_split"
    audio_data_dir = "/Users/christof/Documents/papers/sound/audio_data_simulation/real_split_audio/30_seconds_split"
    test_data_file = "/Users/christof/Documents/papers/sound/data/test_set.csv"
    all_data_file = "/Users/christof/Documents/papers/sound/annotation_30_seconds_split.csv"

    # Get the model and callbacks
    model = get_model(config)
    callbacks = get_callbacks(config)
    # model = DfkdeModel()


    all_data_df = pd.read_csv(all_data_file)
    test_data_df = pd.read_csv(test_data_file)
    df_train_only_blank, df_train_only_whoop = get_annotated_train_df(all_data_df, test_data_df)
    # train_list = df_train_only_blank["names"].tolist()[:200]
    # val_list = df_train_only_blank["names"].tolist()[200:] + df_train_only_whoop["names"].tolist()
    # test_list = test_data_df["names"].tolist()
    data = all_data_df["names"].tolist()
    train_list, val_list, test_list = data, data, data

    file_2_label = dict(zip(all_data_df["names"], all_data_df["annotation_label"]))

    datamodule = CustomDataModule(batch_size=16, data_dir=data_dir, file_2_label=file_2_label, train_files=train_list,
                                  val_files=val_list, test_files=test_list, raw_audio=False)

    # start training
    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    test_results1 = trainer.predict(model=model, datamodule=datamodule)

    true_labels = []
    pred_labels = []
    for i in test_results1:
        pred_labels.extend(i["pred_labels"])
        true_labels.extend(i["label"])

    print(trainer.checkpoint_callback.best_model_path)

    output_path = Path(config["project"]["path"])
    model_path = output_path / "weights" / "lightning" / "model.ckpt"
    # config.trainer.accumulate_grad_batches = 0
    model = get_model(config)
    callbacks = get_callbacks(config)
    trainer2 = Trainer(**config.trainer, callbacks=callbacks)
    # trainer2.fit(model=model, datamodule=datamodule)

    load_model_callback = LoadModelCallback(weights_path=model_path)
    trainer2.callbacks.insert(0, load_model_callback)
    test_results2 = trainer2.predict(model=model, datamodule=datamodule)
    pred_labels2 = []
    for i in test_results2:
        pred_labels2.extend(i["pred_labels"])

    pred_labels = [int(i.item()) for i in pred_labels]
    pred_labels2 = [int(i.item()) for i in pred_labels2]
    true_labels = [int(i.item()) for i in true_labels]
    return pred_labels, pred_labels2, true_labels


if __name__ == '__main__':
    l1, l2, t = fit()
