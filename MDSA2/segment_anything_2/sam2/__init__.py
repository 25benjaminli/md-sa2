# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra import initialize_config_module
import hydra
# hydra is initialized on import of sam2, which sets the search path which can't be modified
# so we need to clear the hydra instance
hydra.core.global_hydra.GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
# hydra.initialize_config_module('../', version_base='1.2')


hydra.initialize_config_module("sam2_configs", version_base="1.2")
