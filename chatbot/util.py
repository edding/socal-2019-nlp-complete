from typing import Any, Dict, Optional
import logging
from subprocess import call

import yaml
import speech_recognition as sr
import spacy

logging.getLogger().setLevel(logging.INFO)


def _parse_configs() -> Dict[str, str]:
    with open("config.yaml") as f:
        config = yaml.full_load(f)
    assert config, "Failed loading config YAML"
    return config


CONFIG = _parse_configs()


def _speed_to_text() -> str:
    text = ""
    while not text:
        try:
            logging.info("Listening...")
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio = r.listen(
                    source,
                    timeout=5,  # wait at most 5 seconds before a phrase starts
                    phrase_time_limit=5,  # wait at most 5 seconds between phrases, otherwise stop and return
                )
            logging.info("Recognizing your voice...")
            text = r.recognize_google(audio)
            logging.info("You said: {}".format(text))
        except:
            logging.info("Voice not recognized")
            output("Sorry I didn't hear you, could you please say it again?")
    return text


def get_user_input(type_: str = CONFIG["input_option"]) -> str:
    if type_ == "voice":
        return _speed_to_text()
    else:
        return input()


def _say(text: str, rate: int = 220):
    logging.info("Bot said: {}".format(text))
    call('say -r {} "{}"'.format(rate, text), shell=True)


def output(text: str, type_: str = CONFIG["input_option"], rate: int = 220):
    if type_ == "voice":
        _say(text, rate)
    else:
        print(text)


def load_spacy_model(model_name: str = CONFIG["spacy_model"]) -> Any:
    """
    Loading Spacy model for name entity recognition.
    Pre-trained models should be downloaded in advance by `python -m spacy download <model_name>`
    """
    logging.info("Loading pre-trained model {}".format(model_name))
    nlp = spacy.load(model_name)
    logging.info("Spacy model loaded")
    return nlp


_NLP = load_spacy_model()


def extract_entity_from_text(text: str, type_: str = "GPE") -> Optional[str]:
    """Extract GPE (location) or ORG (company name) entity from given text."""
    assert type_ in {"GPE", "ORG"}, "Entity type not supported."
    doc = _NLP(text)
    for ent in doc.ents:
        if ent.label_ == type_:
            return str(ent)
    return None
