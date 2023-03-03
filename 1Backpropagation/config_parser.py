import tomllib
import os
import layers
from sequential import Sequential
from general import LossFunction, WeightRegularizationType, DEFAULT_SIZE, ActivationFunctions, GLOROT
from dataset.dataset_generator import DatasetGenerator


class NetworkConfigParser:
    def __init__(self, filename: str):
        """
        :param filename: name of the file. The file must be placed in the config_files dir. The file must be .toml
        file and the filename must include the .toml extension.
        """
        self.filename = filename

    def generate_model(self):
        with open(os.path.join("config_files", self.filename), "rb") as f:
            data = tomllib.load(f)
            global_parameters: dict | None = data.get("GLOBALS")

            # parse all global variables
            if global_parameters is not None:
                global_loss: LossFunction | None = None
                global_loss_str: str | None = global_parameters.get("loss")
                if global_loss_str is not None:
                    global_loss = LossFunction[global_loss_str.upper()]
                global_lrate: float | None = global_parameters.get("lrate")
                global_wreg: float | None = global_parameters.get("wreg")
                global_wrt: WeightRegularizationType | None = None
                global_wrt_str: str | None = global_parameters.get("loss")
                global_seed: int | None = global_parameters.get("seed")
                global_iterations: int | None = global_parameters.get("n")
                global_batch_size: int | None = global_parameters.get("batch_size")
                if global_wrt_str is not None:
                    global_wrt = WeightRegularizationType[global_parameters.get("wrt").upper()]

            model = Sequential()

            layer_parameters: dict | None = data.get("LAYERS")
            if layer_parameters is not None:
                input_layer_size: int = layer_parameters.get("input", DEFAULT_SIZE)
                model.add(layers.Input(input_layer_size))
                layer_param_list: list[dict] = layer_parameters.get("layers", [])
                for layer_param in layer_param_list:
                    size = layer_param.get("size", DEFAULT_SIZE)
                    activation = layer_param.get("act", ActivationFunctions.RELU)
                    initial_weight_range = layer_param.get("wr", GLOROT)
                    learning_rate = layer_param.get("lrate", None)
                    bias_range = layer_param.get("br", None)
                    model.add(layers.Dense(size=size,
                                           activation=activation,
                                           initial_weight_range=initial_weight_range,
                                           learning_rate=learning_rate,
                                           bias_range=bias_range))
                use_softmax = layer_parameters.get("use_softmax")
            else:
                model.add(layers.Input(DEFAULT_SIZE))
                model.add(layers.Dense(size=DEFAULT_SIZE))

            if use_softmax:
                model.add(layers.Softmax())

            model.compile(learning_rate=global_lrate,
                          loss=global_loss,
                          regularization=global_wrt,
                          weight_regularization_rate=global_wreg,
                          random_state=global_seed,
                          iterations=global_iterations,
                          batch_size=global_batch_size)

            dataset_parameters: dict | None = data.get("DATASET")

            if dataset_parameters is not None:
                dataset_n_samples: int  = dataset_parameters.get("n_samples", 500)
                dataset_n_size: int = dataset_parameters.get("n_size", 50)
                dataset_seed: int | None = dataset_parameters.get("seed", None)
                dataset_wr: list[int] = dataset_parameters.get("wr", [.4, .8])
                dataset_hr: list[int] = dataset_parameters.get("hr", [.4, .8])
                dataset_noise: float = dataset_parameters.get("noise", 0.01)
                dataset_show: int = dataset_parameters.get("show_n_random_images", 0)
                dataset_center: bool = dataset_parameters.get("center", True)
                dataset_train: float = dataset_parameters.get("train_size", .7)
                dataset_val: float = dataset_parameters.get("val_size", .2)
                dataset_test: float = dataset_parameters.get("test_size", .1)
                d_train, d_val, d_test = DatasetGenerator.generate_dataset(
                    n=dataset_n_size,
                    number=dataset_n_samples,
                    random_seed=dataset_seed,
                    wr=dataset_wr,
                    hr=dataset_hr,
                    noise=dataset_noise,
                    show_n_random_images=dataset_show,
                    center=dataset_center,
                    train=dataset_train,
                    val=dataset_val,
                    test=dataset_test,
                    flat=True)
            else:
                raise ValueError("Dataset must be generated")

            return model, d_train, d_val, d_test


if __name__ == "__main__":
    networkParser = NetworkConfigParser('two_hidden_layers.toml')
    model, train, val, test = networkParser.generate_model()
    model.fit(X=train[0], y=train[1], X_val=val[0], y_val=val[1], verbose=1)
    y_out = model.predict(test[0].T)
    print("Test loss", model._loss(test[1], y_out))
    # DatasetGenerator.generate_dataset(
    #     n=50,
    #     number=100,
    #     random_seed=42,
    #     wr=[0.4, 0.6],
    #     hr=[0.4, 0.6],
    #     noise=0.01,
    #     show_n_random_images=10,
    #     center=False,
    #     )
