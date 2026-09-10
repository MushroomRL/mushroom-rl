class IsaacGpuParams:
    """
    Class for defining the GPU configuration parameters of a PhysX scene.

    A parameter left at ``None`` keeps the value PhysX picks by itself. The table below lists every parameter
    and the size :meth:`per_env` gives it for each environment.

    .. csv-table::
       :header: "Parameter", "Per environment", "Meaning"
       :widths: 44, 14, 42

       "``gpu_found_lost_aggregate_pairs_capacity``", "``128``", "Shape pairs found or lost within an
       aggregate in a step"
       "``gpu_total_aggregate_pairs_capacity``", "``20``", "Aggregate pairs the broadphase reports, one
       aggregate holding one robot"
       "``gpu_max_rigid_patch_count``", "``160``", "Contact patches between rigid bodies"
       "``gpu_temp_buffer_capacity``", "``None``", "Scratch space, in bytes"
       "``gpu_found_lost_pairs_capacity``", "``None``", "Shape pairs found or lost in a step"
       "``gpu_max_rigid_contact_count``", "``None``", "Contacts between rigid bodies"
       "``gpu_collision_stack_size``", "``None``", "Collision detection stack, in bytes"
       "``gpu_heap_capacity``", "``None``", "Heap, in bytes"
       "``gpu_max_num_partitions``", "``None``", "Partitions the solver batches the scene into"
       "``gpu_max_particle_contacts``", "``None``", "Contacts involving particles"
       "``gpu_max_soft_body_contacts``", "``None``", "Contacts involving soft bodies"
       "``gpu_max_deformable_surface_contacts``", "``None``", "Contacts involving deformable surfaces"

    A buffer too small for the scene makes PhysX warn and drop the interactions that do not fit.

    """
    def __init__(self, **overrides):
        """
        Constructor.

        Args:
            **overrides: The capacities to set, named as in the table above. Any other name raises a
                ``ValueError``.

        """
        self._values = self._default_values()

        unknown = set(overrides) - set(self._values)
        if unknown:
            raise ValueError(f"unknown physx gpu parameters: {sorted(unknown)}")

        self._values.update(overrides)

    def __getitem__(self, name):
        return self._values[name]

    def __setitem__(self, name, value):
        if name not in self._values:
            raise ValueError(f"unknown physx gpu parameter: {name}")

        self._values[name] = value

    def __contains__(self, name):
        return name in self._values

    def keys(self):
        """
        Returns:
            The name of every parameter, so that the object can be unpacked with ``**``.

        """
        return self._values.keys()

    @classmethod
    def per_env(cls, num_envs, **overrides):
        """
        Builds the capacities of a scene from the size each environment needs.

        Args:
            num_envs (int): The number of environments the scene holds;
            **overrides: The capacities to set per environment, named as in the table above, overriding the
                per-environment defaults listed there. Any other name raises a ``ValueError``.

        Returns:
            The parameters, each one the per-environment size multiplied by ``num_envs``.

        """
        values = cls._default_values_per_env()

        unknown = set(overrides) - set(cls._default_values())
        if unknown:
            raise ValueError(f"unknown physx gpu parameters: {sorted(unknown)}")

        values.update(overrides)

        return cls(**{name: size * num_envs for name, size in values.items() if size is not None})

    @staticmethod
    def _default_values():
        """
        Returns:
            The default value of every parameter, as a fresh dictionary.

        """
        return dict(
            gpu_collision_stack_size=None,
            gpu_found_lost_aggregate_pairs_capacity=None,
            gpu_found_lost_pairs_capacity=None,
            gpu_heap_capacity=None,
            gpu_max_deformable_surface_contacts=None,
            gpu_max_num_partitions=None,
            gpu_max_particle_contacts=None,
            gpu_max_rigid_contact_count=None,
            gpu_max_rigid_patch_count=None,
            gpu_max_soft_body_contacts=None,
            gpu_temp_buffer_capacity=None,
            gpu_total_aggregate_pairs_capacity=None
        )

    @staticmethod
    def _default_values_per_env():
        """
        Returns:
            The size every parameter is given per environment, as a fresh dictionary.

        """
        return dict(
            gpu_found_lost_aggregate_pairs_capacity=128,
            gpu_total_aggregate_pairs_capacity=20,
            gpu_max_rigid_patch_count=160
        )
