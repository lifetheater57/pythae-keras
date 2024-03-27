import os

import pytest

import keras
import numpy as np

from pythae.customexception import BadInheritanceError
from pythae.models import BaseAE, BaseAEConfig
from tests.data.custom_architectures import (
    Decoder_AE_Conv,
    Encoder_AE_Conv,
    NetBadInheritance,
)

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(params=[BaseAEConfig(), BaseAEConfig(latent_dim=(5,))])
def model_configs_no_input_dim(request):
    return request.param


@pytest.fixture(
    params=[
        BaseAEConfig(input_dim=(1, 28, 28), latent_dim=(10,)),
        BaseAEConfig(input_dim=(1, 100, 0), latent_dim=(5,)),
        BaseAEConfig(input_dim=(1, 1e4), latent_dim=(5,)),
    ]
)
def model_config_with_input_dim(request):
    return request.param


@pytest.fixture
def custom_decoder(model_config_with_input_dim):
    return Decoder_AE_Conv(model_config_with_input_dim)


class Test_Model_Building:
    def test_build_model(self, model_config_with_input_dim):
        model = BaseAE(model_config_with_input_dim)
        assert all(
            [
                model.input_dim == model_config_with_input_dim.input_dim,
                model.latent_dim == model_config_with_input_dim.latent_dim,
            ]
        )

    def test_raises_no_input_dim(self, model_configs_no_input_dim, custom_decoder):
        with pytest.raises(AttributeError):
            model = BaseAE(model_configs_no_input_dim)

        model = BaseAE(model_configs_no_input_dim, decoder=custom_decoder)

    def test_build_custom_arch(self, model_config_with_input_dim, custom_decoder):

        model = BaseAE(model_config_with_input_dim, decoder=custom_decoder)

        assert model.decoder == custom_decoder
        assert not model.model_config.uses_default_decoder

        model = BaseAE(model_config_with_input_dim)

        assert model.model_config.uses_default_decoder

        model = BaseAE(model_config_with_input_dim, decoder=custom_decoder)

        assert model.decoder == custom_decoder
        assert not model.model_config.uses_default_decoder


class Test_Model_Saving:
    def test_creates_saving_path(self, tmpdir, model_config_with_input_dim):
        tmpdir.mkdir("saving")
        dir_path = os.path.join(tmpdir, "saving")
        model = BaseAE(model_config_with_input_dim)
        model.save(dir_path=dir_path)

    def test_default_model_saving(self, tmpdir, model_config_with_input_dim):

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")

        model = BaseAE(model_config_with_input_dim)

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.keras"]
        )

        # reload model
        model_rec = BaseAE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                np.array_equal(model_rec.get_weights()[i], model.get_weights()[i])
                for i in range(max(len(model.get_weights()), len(model_rec.get_weights())))
            ]
        )

    #TODO: check why not testing custom encoder
    
    def test_custom_decoder_model_saving(
        self, tmpdir, model_config_with_input_dim, custom_decoder
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = BaseAE(model_config_with_input_dim, decoder=custom_decoder)

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.keras", "decoder.keras"]
        )

        # reload model
        model_rec = BaseAE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                np.array_equal(model_rec.get_weights()[i], model.get_weights()[i])
                for i in range(max(len(model.get_weights()), len(model_rec.get_weights())))
            ]
        )

    def test_raises_missing_files(
        self, tmpdir, model_config_with_input_dim, custom_decoder
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")

        model = BaseAE(model_config_with_input_dim, decoder=custom_decoder)

        model.save(dir_path=dir_path)

        os.remove(os.path.join(dir_path, "decoder.keras"))

        # check raises decoder.keras is missing
        with pytest.raises(FileNotFoundError):
            model_rec = BaseAE.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "model.keras"))

        # check raises model.keras is missing
        with pytest.raises(FileNotFoundError):
            model_rec = BaseAE.load_from_folder(dir_path)

        #TODO: check if required, error raised by Keras
        with pytest.raises(AttributeError):
            keras.models.save_model({"wrong_key": 0.0}, os.path.join(dir_path, "model.keras"))

        #TODO: check if should implement it with Keras
        # # check raises wrong key in model.keras
        # with pytest.raises(KeyError):
        #     model_rec = BaseAE.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "model_config.json"))

        # check raises model_config.json is missing
        with pytest.raises(FileNotFoundError):
            model_rec = BaseAE.load_from_folder(dir_path)
