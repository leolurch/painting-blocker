# Met selection photographs

Downloads the Met benchmark photographs used to choose the epoch and the configuration.
The paper keeps visitor queries from the official validation and test lists that show one
of the SynGallery painting identities, then splits those paintings into the checkpoint
half and the configuration half. `materialize_met_identity_halves.py` builds that split.
This directory does not train or evaluate a recognition model.
