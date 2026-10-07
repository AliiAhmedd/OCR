"""id_classifier: sorts failed 4201 OCR transactions into reviewed, per-country document datasets.

Flow for every image: source (fetch) -> detector (find documents) -> classifier(s) (label them)
-> routing (auto-accept or send to review) -> results report.
"""

__version__ = "0.1.0"  # written into every prediction row as pipeline_version
