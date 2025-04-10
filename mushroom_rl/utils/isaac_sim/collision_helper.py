from mushroom_rl.core import ArrayBackend
from functools import reduce

class CollisionHelper:
    BASE_ENV_PATH = "/World/envs"
    TEMPLATE_ENV_PATH = BASE_ENV_PATH + "/env"
    ZERO_ENV_PATH = TEMPLATE_ENV_PATH + "_0"

    def __init__(self, collision_groups, backend, num_envs, device, n_intermediate_steps = 1):
        """
        Constructor.

        Args:
            collision_groups (list, None): A list containing groups of prims for which collisions should be checked during
                simulation. The entries are given as ``(key, prim_paths)``, where key is a string for later reference and 
                prim_paths is a list of paths to the prims.
            backend (str): name of the backend for array operations.
            n_envs (int): Number of parallel environments.
            device (str): Compute device (e.g., 'cuda:0').
            n_intermediate_steps (int): Number of intermediate control steps. Defaults to 1.
        """
        self._backend = backend
        self._device = device
        self._num_envs = num_envs
        self.collision_groups = {key: group for key, group in collision_groups} if collision_groups is not None else {}
        self._n_intermediate_steps = n_intermediate_steps
        
    def prepare_env(self, stage):
        """
        Ensures that all objects in the collision groups possess the necessary APIs.

        Args:
            stage: current stage of the isaac sim simulation
        """
        from pxr import PhysxSchema

        for group_name, group in self.collision_groups.items():
            for path in group:
                if path.startswith("/World/"):
                    continue
                prim = stage.GetPrimAtPath(self.ZERO_ENV_PATH + "/Robot" + path)
                if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim)

    def set_up(self):#TODO check if possible One ContactView per Group
        """
        Sets up RigidContactViews for all object of all collisions groups
        """
        from isaacsim.core.api.sensors import RigidContactView

        self._views = {}
        self._collision_groups_indices = {}
        self._collision_group_contains_world = {key: False for key in self.collision_groups}

        for group_name, group in self.collision_groups.items():
            self._collision_group_contains_world[group_name] = any([path.startswith("/World/") for path in group])
            if self._collision_group_contains_world[group_name]:
                continue

            possible_partners = reduce(
                lambda acc, val: acc + val if val not in acc else acc, [value for key, value in self.collision_groups.items() if key != group_name], []
            )
            self._collision_groups_indices[group_name] = {key: self._arr_backend.from_list([possible_partners.index(value) for value in self.collision_groups[key]]) for key in self.collision_groups if key != group_name}
            possible_partners = [self.BASE_ENV_PATH + "/env_*/Robot" + partner if not partner.startswith("/World/") else partner for partner in possible_partners]
            
            paths = [self.BASE_ENV_PATH + "/env_*/Robot" + path for path in group if not path.startswith("/World/")]
            view = RigidContactView(
                prim_paths_expr=paths,
                name=group_name + "_view",
                filter_paths_expr=[possible_partners]*len(paths),
                prepare_contact_sensors=False
            )
            self._views[group_name] = view
        #sensor_count
    
    def post_reset(self):
        """
        Called after world.reset() is completed.
        """
        for group in self._views:
            self._views[group].initialize()

    def get_collision_force(self, group1, group2, selector=None, dt=1.0):
        """
        Computes the collision forces or impulses between two collision groups.

        Args:
            group1 (str): The name of the first collision group.
            group2 (str): The name of the second collision group.
            selector (Callable[[torch.tensor | np.ndarray], torch.tensor | np.ndarray], optional): 
                A function that processes the collision force tensor. 
                Defaults to selecting the maximum force of each environment
            dt (float, optional): The time step duration used for computing forces. 
                The function returns contact impulses if the default dt is used

        Returns:
            A tensor or array containing the computed collision forces between the groups, 
            processed by the `selector` function.
        """
        assert not(self._collision_group_contains_world[group1] and self._collision_group_contains_world[group2])
        
        if selector is None:
            selector = lambda x: self._arr_backend.max(self._arr_backend.max(self._arr_backend.norm(x, dim=3), dim=2), dim=1)

        if self._collision_group_contains_world[group2]:
            group = group1
            indices_prims = self._collision_groups_indices[group1][group2]
        else:
            group = group2
            indices_prims = self._collision_groups_indices[group2][group1]
        n_bodies = len(self.collision_groups[group])
        
        forces = self._views[group].get_contact_force_matrix(clone=False, dt=dt)
        #transform to (num_envs, n_bodies, n_possible_partners, 3)
        forces = forces.view(n_bodies, self._num_envs, -1, 3).transpose(0, 1)
        forces = forces[:, :, indices_prims]

        return selector(forces)
    
    def check_collision(self, group1, group2, threshold, selector=None, dt=1.0):
        """
        Checks whether the collision force between two collision groups exceeds a given threshold.

        Args:
            group1 (str): The name of the first collision group.
            group2 (str): The name of the second collision group.
            threshold (float, torch.tensor, np.ndarray): The threshold value to compare against.
                Can be a scalar or a tensor/array of the same shape as the computed forces.
            selector (Callable[[torch.tensor | np.ndarray], torch.tensor | np.ndarray], optional): 
                A function that processes the collision force tensor or array.
                If None, a default selector is used. Defaults to None.
            dt (float, optional): The time step duration used for computing forces. 
                The function uses impulses if the default dt is used

        Returns:
            A boolean tensor or array indicating where the computed forces exceed the given threshold.
        """
        forces = self.get_collision_force(group1, group2, selector, dt)
        return forces > threshold
    
    def count_collisions(self, group1, group2, threshold, selector=None, dt=1.0):
        """
        Counts the number of collisions between two groups, considering at most one collision between each 
        possible pair of objects from the two groups. 
        For example, the maximum collision count between a group with 3 objects and a group with 4 objects 
        would be 12.

        Args:
            group1 (str): The name of the first collision group.
            group2 (str): The name of the second collision group.
            threshold (float, torch.tensor, np.ndarray): The threshold value for detecting collisions.
                Can be a scalar or a tensor/array of the same shape as the computed forces.
            selector (Callable[[torch.tensor | np.ndarray], torch.tensor | np.ndarray], optional): 
                A function that processes the collision force tensor or array.
                If None, the default selector computes the norm along the collision dimension. Defaults to None.
            dt (float, optional): The time step duration used for computing forces. Defaults to 1.0.

        Returns:
            A tensor or array containing the count of collisions
        """
        if selector is None:
            selector=lambda x: self._arr_backend.norm(x, dim=3)

        forces = self.get_collision_force(group1, group2, selector, dt)
        return self._arr_backend.sum(self._arr_backend.sum(forces > threshold, dim=2), dim=1)
    
    def get_net_contact_forces(self, group, dt=1.0):
        forces = self._views[group].get_net_contact_forces(dt=dt)
        forces = forces.view(-1, self._num_envs, 3).transpose(0, 1)
        
        return forces

    @property
    def _arr_backend(self):
        return ArrayBackend.get_array_backend(self._backend)