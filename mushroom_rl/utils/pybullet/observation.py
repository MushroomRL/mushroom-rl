from enum import Enum


class PyBulletObservationType(Enum):
    """
    An enum indicating the type of data that should be added to the observation
    of the environment, can be Joint/Body/Site positions and velocities.

    """
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)  # Automatically assign ID based on current number of members
        return obj

    BODY_POS = ('body', 7, 'getBasePositionAndOrientation', (0, 1))
    BODY_LIN_VEL = ('body', 3, 'getBaseVelocity', (0,))
    BODY_ANG_VEL = ('body', 3, 'getBaseVelocity', (1,))
    JOINT_POS = ('joint', 1, 'getJointState', (0,))
    JOINT_VEL = ('joint', 1, 'getJointState', (1,))
    LINK_POS = ('link', 7, 'getLinkState', (0, 1))
    LINK_LIN_VEL = ('link', 3, 'getLinkState', (-2,), True)
    LINK_ANG_VEL = ('link', 3, 'getLinkState', (-1,), True)
    CONTACT_FLAG = ('contact', 1)

    def __init__(self, category, length, accessor=None, components=(), compute_link_velocity=False):
        """
        Constructor.

        Args:
            category (str): the category of the observation type, either 'body', 'joint', 'link' or 'contact';
            length (int): the number of values of the observation;
            accessor (str, None): the name of the PyBullet client call serving the observation, or None for the
                contact flags;
            components (tuple, ()): the positions, in the result of the call, of the values forming the
                observation, concatenated in this order;
            compute_link_velocity (bool, False): whether the call has to compute the velocity of the link.

        """
        self.category = category
        self.length = length
        self.accessor = accessor
        self.components = components
        self.compute_link_velocity = compute_link_velocity

    def is_body(self):
        """
        Checks whether the observation type belongs to the 'body' category.

        Returns:
            bool: True if the observation type is category 'body', False otherwise.

        """
        return self.category == 'body'

    def is_joint(self):
        """
        Checks whether the observation type belongs to the 'joint' category.

        Returns:
            bool: True if the observation type is category 'joint', False otherwise.

        """
        return self.category == 'joint'

    def is_link(self):
        """
        Checks whether the observation type belongs to the 'link' category.

        Returns:
            bool: True if the observation type is category 'link', False otherwise.

        """
        return self.category == 'link'

    def is_contact(self):
        """
        Checks whether the observation type belongs to the 'contact' category.

        Returns:
            bool: True if the observation type is category 'contact', False otherwise.

        """
        return self.category == 'contact'

    @property
    def accessor_kwargs(self):
        """
        Returns:
            The keyword arguments of the PyBullet client call serving the observation.

        """
        return dict(computeLinkVelocity=True) if self.compute_link_velocity else dict()
