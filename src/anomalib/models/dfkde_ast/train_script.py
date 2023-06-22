
from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters

from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks


def fit():

    MODEL = "padim"  # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
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
    labels_dir = "/Users/christof/Documents/papers/sound/annotation 30_seconds_split.csv"
    datamodule = CustomDataModule(32, data_dir, labels_dir)

    # Get the model and callbacks
    model = get_model(config)
    callbacks = get_callbacks(config)
    # model = DfkdeModel()

    # start training
    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    test_results1 = trainer.test(model=model, datamodule=datamodule)
    print(trainer.checkpoint_callback.best_model_path)
    print(test_results1)

    # load saved model
    # trainer2 = None
    # output_path = Path(config["project"]["path"])
    # model_path = output_path / "weights" / "lightning" / "model.ckpt"
    # # config.trainer.accumulate_grad_batches = 0
    # model = get_model(config)
    # callbacks = get_callbacks(config)
    # trainer2 = Trainer(**config.trainer, callbacks=callbacks)
    # load_model_callback = LoadModelCallback(weights_path=model_path)
    # trainer2.callbacks.insert(0, load_model_callback)
    # test_results2 = trainer2.test(model=model, datamodule=datamodule)

if __name__ == '__main__':
    fit()
