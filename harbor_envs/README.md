# Modal environment

`harbor_envs.modal:ModalEnvironment` extends the pinned upstream Harbor Modal provider with
isolated service deployment, read-only model Volumes, input verification, trajectory collection,
and resource cleanup. `abstract_mlplat/` contains shared lifecycle, trajectory, and retry helpers.

There is one supported runtime provider. Authenticate with your own Modal workspace. Model services use IPv4, private sandbox namespaces, and authenticated Modal management.
