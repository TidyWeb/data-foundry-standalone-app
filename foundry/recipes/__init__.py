"""Every file in this folder that defines RECIPE is picked up automatically."""
import importlib
import pkgutil

RECIPES = {}

for _module in pkgutil.iter_modules(__path__):
    _loaded = importlib.import_module(f"{__name__}.{_module.name}")
    _recipe = getattr(_loaded, "RECIPE", None)
    if _recipe is not None:
        RECIPES[_recipe.key] = _recipe
