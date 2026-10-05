"""Classifiers: decide document_type, issuing_country and document_side for one document region.

Only the base class is imported here. The heavy adapters (VLM, embeddings) are imported by name from
config.py only when a configuration asks for them, so their libraries stay optional.
"""

from id_classifier.classifiers.base import Classifier

__all__ = ["Classifier"]
