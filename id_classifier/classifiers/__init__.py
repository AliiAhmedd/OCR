"""Classifiers: decide document_type, issuing_country and document_side for one document region.

Only base and mock are imported here. The heavy adapters (VLM, embeddings) are imported by name from
config.py only when a configuration asks for them, so their libraries stay optional.
"""

from id_classifier.classifiers.base import Classifier

__all__ = ["Classifier"]
