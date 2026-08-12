import atexit

import isaacsim
from isaacsim import SimulationApp


class IsaacLauncher:
    """
    Owner of the Isaac Sim simulation app.

    Isaac Sim's Carbonite framework only lets its modules be imported once the app is running, so everything
    in MushroomRL's Isaac layer has to be imported *after* :meth:`launch` has been called::

        from mushroom_rl.utils.isaac_sim import IsaacLauncher

        IsaacLauncher.launch(headless=True)

        from mushroom_rl.environments.isaacsim_envs import CartPoleIsaac

    This class is the only part of the layer that can be imported before that point.

    The app is a per-process singleton: it wraps ``omni.kit.app``, which is global to the process, and
    starting a second one crashes the interpreter. There is correspondingly nothing to instantiate here --
    the class is a namespace holding the one app, and every method is a classmethod. It also means the
    settings that belong to the app rather than to a scene -- whether to open a window, and which physics
    engine to simulate with -- are arguments of :meth:`launch` instead of arguments of the environments.

    """
    _app = None
    _disable_rendering = None
    _carb_overrides = None

    def __new__(cls, *args, **kwargs):
        """
        Raises:
            TypeError: Always. There is a single simulation app per process and the class holds it, so an
                instance would carry no state of its own.

        """
        raise TypeError(f"{cls.__name__} cannot be instantiated: it holds the one simulation app of  the process. "
                        f"Call {cls.__name__}.launch() on the class itself.")

    @classmethod
    def launch(cls, headless=True, physics_engine='physx', disable_rendering=False, carb_settings=None):
        """
        Starts Isaac Sim, so that its modules and the MushroomRL environments built on them can be imported.

        Calling this a second time does nothing but return the running app: a process can only hold one.

        Args:
            headless (bool, True): Whether to run without a window.
            physics_engine (str, 'physx'): The physics engine to simulate with, either 'physx' or 'newton'. Isaac Sim
                documents Newton as experimental, and MushroomRL does not test it: the robot assets shipped
                here are tuned for PhysX and the two engines do not produce the same dynamics.
            disable_rendering (booli, False): Whether to skip the carb settings that make the viewer/recorder show
                live data (see :meth:`_apply_carb_settings`), independently of ``headless``.
            carb_settings (dict, None): Overrides for the default carb settings applied at startup, see
                :meth:`_apply_carb_settings`. Keys are carb setting paths (e.g. ``"/physics/fabricEnabled"``);
                values override the corresponding default, and unknown keys are simply added.

        Returns:
            The running simulation app.

        """
        if cls._app is None:
            cls._app = SimulationApp({"headless": headless, "hide_ui": False, "renderer": "RaytracedLighting"})
            cls._carb_overrides = carb_settings
            cls.activate_rendering(not disable_rendering)
            cls._select_physics_engine(physics_engine)

            atexit.register(cls.shutdown)

        return cls._app

    @classmethod
    def shutdown(cls):
        """
        Closes Isaac Sim. This ends the process: the app shuts the Carbonite framework down and terminates,
        so nothing can be simulated afterward and Isaac Sim cannot be launched again. Since this is also
        registered to run at exit, any ``atexit`` callback registered *before* :meth:`launch` is never
        reached; register cleanups afterward, where they run first.

        Does nothing if Isaac Sim is not running.

        """
        if cls._app is not None:
            cls._app.close()

    @classmethod
    def get(cls):
        """
        Returns the running simulation app.

        Returns:
            The simulation app started by :meth:`launch`.

        Raises:
            RuntimeError: If Isaac Sim has not been launched yet.

        """
        cls.require_running()

        return cls._app

    @classmethod
    def require_running(cls):
        """
        Guards a module against being imported before Isaac Sim is running.

        Isaac's Carbonite framework only allows ``isaacsim.*`` submodules to be imported once the app is live, so
        importing a module that does so too early fails deep inside Isaac's own machinery with an opaque
        ``ModuleNotFoundError``. Modules that import such submodules should trigger this check first -- see
        :mod:`mushroom_rl.utils.isaac_sim._require_launched` -- so the failure instead points back to the actual
        cause.

        Skipped when ``isaacsim`` itself is a Sphinx autodoc mock (``autodoc_mock_imports``): every downstream
        ``isaacsim.*`` import then resolves harmlessly to a mock attribute instead of raising, so there is
        nothing left for this check to guard against.

        Raises:
            RuntimeError: If Isaac Sim has not been launched yet.

        """
        if cls._app is None and not getattr(isaacsim, '__sphinx_mock__', False):
            raise RuntimeError("Isaac Sim is not running: call IsaacLauncher.launch() before importing this "
                               "module.")

    @classmethod
    def is_headless(cls):
        """
        Returns:
            Whether Isaac Sim was launched without a window.

        """
        return cls.get().config["headless"]

    @classmethod
    def is_rendering_disabled(cls):
        """
        Returns:
            Whether the carb settings that make the viewer/recorder show live data are currently off.

        """
        cls.require_running()

        return cls._disable_rendering

    @classmethod
    def activate_rendering(cls, flag=True):
        """
        Toggles whether the viewer/recorder show live data, independently of ``headless``. Only affects
        environments created *after* this call: Isaac Sim's render product is bound to whatever this was
        at the time it was built, so an already-running environment keeps showing stale data regardless.

        Args:
            flag (bool, True): Whether to activate rendering (True) or deactivate it (False), i.e. whether
                to apply the carb settings that make the viewer/recorder show live data, see
                :meth:`_apply_carb_settings`.

        """
        cls.require_running()

        cls._disable_rendering = not flag
        cls._apply_carb_settings(cls._app, cls._disable_rendering, cls._carb_overrides)

    @staticmethod
    def _apply_carb_settings(simulation_app, disable_rendering, overrides=None):
        """
        Apply mushroom default settings for optimization.

        Args:
            simulation_app: The running simulation app.
            disable_rendering (bool): Whether to skip the three groups above.
            overrides (dict, None): Carb setting paths overriding the defaults below, or adding new ones.

        """
        headless = simulation_app.config["headless"]
        settings = {
            "/app/useFabricSceneDelegate": not disable_rendering,
            "/app/runLoops/main/rateLimitEnabled": False,
            "/persistent/omnihydra/useSceneGraphInstancing": True,
            "/persistent/simulation/minFrameRate": 15,
            "/exts/omni.replicator.core/Orchestrator/enabled": headless and not disable_rendering,
            "/metricsAssembler/changeListenerEnabled": False,
            "/physics/physxDispatcher": True,
            "/physics/disableContactProcessing": True,
            "/physics/collisionConeCustomGeometry": False,
            "/physics/collisionCylinderCustomGeometry": False,
            "/physics/fabricEnabled": True,
            "/physics/updateToUsd": not disable_rendering,
            "/physics/updateParticlesToUsd": not disable_rendering,
            "/physics/updateVelocitiesToUsd": not disable_rendering,
            "/physics/updateForceSensorsToUsd": not disable_rendering,
            "/physics/outputVelocitiesLocalSpace": False,
            "/physics/useFastCache": False,
            "/physics/visualizationDisplayJoints": False,
            "/physics/fabricUpdateTransformations": not disable_rendering,
            "/physics/fabricUpdateVelocities": not disable_rendering,
            "/physics/fabricUpdateForceSensors": not disable_rendering,
            "/physics/fabricUpdateJointStates": not disable_rendering,
            "/physics/fabricUseGPUInterop": True,
            "/physics/resourcemonitor/timeBetweenQueries": 100,
            "/rtx/hydra/readTransformsFromFabricInRenderDelegate": not disable_rendering,
            "/rtx/translucency/enabled": False,
            "/rtx/reflections/enabled": False,
            "/rtx/indirectDiffuse/enabled": False,
            "/rtx-transient/dlssg/enabled": False,
            "/rtx/directLighting/enabled": True,
            "/rtx/directLighting/sampledLighting/samplesPerPixel": 1,
            "/rtx/shadows/enabled": True,
            "/rtx/ambientOcclusion/enabled": False,
        }

        if overrides is not None:
            settings.update(overrides)

        for path, value in settings.items():
            simulation_app.set_setting(path, value)

    @staticmethod
    def _select_physics_engine(physics_engine):
        """
        Makes the requested physics engine the active one.

        Args:
            physics_engine (str): The physics engine to simulate with, either 'physx' or 'newton'.

        """
        # this module has to be importable before Isaac Sim runs, since it is what starts it, so these two are
        # the one place in the layer where an Isaac import cannot sit at the top of a file
        import isaacsim.core.experimental.utils.app as app_utils
        from isaacsim.core.simulation_manager import SimulationManager

        if physics_engine != SimulationManager.get_active_physics_engine():
            # Newton ships disabled outside its own launch script, so it has to be brought up by hand
            if physics_engine == 'newton':
                app_utils.enable_extension('isaacsim.physics.newton')
                app_utils.enable_extension('isaacsim.physics.newton.tensors')
            SimulationManager.switch_physics_engine(physics_engine)
