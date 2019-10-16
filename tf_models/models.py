from enum import IntEnum
from abc import abstractmethod
from typing import Tuple
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Model

from tf_models.utils import load_model

logging.getLogger().setLevel(logging.INFO)


class _BinaryModel:
    def __init__(self, name: str, root: str):
        # Load model
        loaded = load_model(name, root=root)  # type: Tuple[Model, TfidfVectorizer]
        self.model, self.vectorizer = loaded

    def _predict(self, text: str) -> int:
        """Return the predicted label (0/1)"""
        vec = self.vectorizer.transform([text])
        [[pred]] = self.model.predict(vec)
        return 1 if pred >= 0.5 else 0

    @abstractmethod
    def predict(self, text: str) -> IntEnum:
        """Leave the actual label to sub-class"""
        raise NotImplementedError


class Intents(IntEnum):
    Stock = 0
    Weather = 1


class _IntentModel(_BinaryModel):
    def __init__(self):
        super().__init__(name="intent", root="tf_models/intent")

    def predict(self, text: str) -> Intents:
        return Intents(self._predict(text))


class FlowControls(IntEnum):
    Stop = 0
    Continue = 1


class _FlowControlModel(_BinaryModel):
    def __init__(self):
        super().__init__(name="flow_control", root="tf_models/flow_control")

    def predict(self, text: str) -> FlowControls:
        return FlowControls(self._predict(text))


intent_model = _IntentModel()
flow_control_model = _FlowControlModel()
