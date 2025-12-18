from .base import Catalog, BaseCatalog
from .cub_dataset import CUBCatalog
from .nabirds_dataset import NABirdsCatalog
from .fungitastic import FungiTasticCatalog
from .inaturalist import INaturalistCatalog
from .fgvcaircraft import FGVCAircraftCatalog


__all__ = ["Catalog", "BaseCatalog", "FungiTasticCatalog", "CUBCatalog", "NABirdsCatalog", "INaturalistCatalog", "FGVCAircraftCatalog"]

# TODO Lsun, texture, ?tiny imagenet, Turtle? rysove? wild life