"""This module contains functions for loading models."""
from os import path
import logging
from typing import Tuple, Union

import torch
from torch import nn

from network.MFNET import MFNET_3D
from network.TorchUtils import TorchModel
from network.anomaly_detector_model import AnomalyDetector
from network.c3d import C3D
from network.resnet import generate_model
from utils.types import Device


def load_feature_extractor(
    features_method: str, feature_extractor_path: str, device: Union[torch.device, str]
) -> nn.Module:
    """Load feature extractor from given path.

    Args:
        features_method (str): The feature extractor model type to use. Either c3d | mfnet | r3d101 | r3d152.
        feature_extractor_path (str): Path to the feature extractor model.
        device (Union[torch.device, str]): Device to use for the model.

    Raises:
        FileNotFoundError: The path to the model does not exist.
        NotImplementedError: The provided feature extractor method is not implemented.

    Returns:
        nn.Module: _description_
    """
    if not path.exists(feature_extractor_path):
        raise FileNotFoundError(
            f"Couldn't find feature extractor {feature_extractor_path}.\n"
            + r"If you are using resnet, download it first from:\n"
            + r"r3d101: https://drive.google.com/file/d/1kQAvOhtL-sGadblfd3NmDirXq8vYQPvf/view?usp=sharing"
            + "\n"
            + r"r3d152: https://drive.google.com/uc?id=17wdy_DS9UY37J9XTV5XCLqxOFgXiv3ZK&export=download"
        )
    logging.info(f"Loading feature extractor from {feature_extractor_path}")

    model = None
    if features_method == "c3d":
        model = C3D(pretrained=feature_extractor_path)
    elif features_method == "mfnet":
        model = MFNET_3D()
        model.load_state(state_dict=feature_extractor_path)
    elif features_method == "r3d101":
        model = generate_model(model_depth=101)
        param_dict = torch.load(feature_extractor_path)["state_dict"]
        param_dict.pop("fc.weight")
        param_dict.pop("fc.bias")
        model.load_state_dict(param_dict)
    elif features_method == "r3d152":
        model = generate_model(model_depth=152)
        param_dict = torch.load(feature_extractor_path)["state_dict"]
        param_dict.pop("fc.weight")
        param_dict.pop("fc.bias")
        model.load_state_dict(param_dict)
    else:
        raise NotImplementedError(
            f"Features extraction method {features_method} not implemented"
        )

    return model.to(device)


def load_anomaly_detector(ad_model_path: str, device: Device) -> AnomalyDetector:
    """Load anomaly detection model from given path.

    Args:
        ad_model_path (str): Path to the anomaly detection model.
        device (Device): Device to use for the model.

    Raises:
        FileNotFoundError: The path to the model does not exist.

    Returns:
        AnomalyDetector
    """
    if not path.exists(ad_model_path):
        raise FileNotFoundError(f"Couldn't find anomaly detector {ad_model_path}.")
    logging.info(f"Loading anomaly detector from {ad_model_path}")

    anomaly_detector = TorchModel.load_model(ad_model_path).to(device)
    return anomaly_detector.eval()


def load_models(
    feature_extractor_path: str,
    ad_model_path: str,
    features_method: str = "c3d",
    device: str = "cuda",
) -> Tuple[nn.Module, nn.Module]:
    """Loads both feature extractor and anomaly detector from the given paths.

    Args:
        feature_extractor_path (str): Path of the features extractor weights to load.
        ad_model_path (str): Path of the anomaly detector weights to load.
        features_method (str, optional): Name of the model to use for features extraction.
            Defaults to "c3d".
        device (str, optional): Device to use for the models. Defaults to "cuda".

    Returns:
        Tuple[nn.Module, nn.Module]
    """
    feature_extractor = load_feature_extractor(
        features_method, feature_extractor_path, device
    )
    anomaly_detector = load_anomaly_detector(ad_model_path, device)
    return anomaly_detector, feature_extractor
