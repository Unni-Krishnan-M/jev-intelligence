"""The generic structured-dataset adapter (any long CSV + ``configs/domains/<name>.yaml``)."""

from jev_ml.domains.generic.adapter import GenericAdapter, adapter_for
from jev_ml.domains.generic.config import GenericDomainConfig, load_config, parse_config

__all__ = ["GenericAdapter", "GenericDomainConfig", "adapter_for", "load_config", "parse_config"]
