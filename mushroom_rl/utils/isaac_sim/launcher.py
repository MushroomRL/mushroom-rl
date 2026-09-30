import atexit
import contextlib
import os

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

    def __new__(cls, *args, **kwargs):
        """
        Raises:
            TypeError: Always. There is a single simulation app per process and the class holds it, so an
                instance would carry no state of its own.

        """
        raise TypeError(f"{cls.__name__} cannot be instantiated: it holds the one simulation app of  the process. "
                        f"Call {cls.__name__}.launch() on the class itself.")

    @classmethod
    def launch(cls, headless=True, physics_engine='physx', log_level='warning', carb_settings=None):
        """
        Starts Isaac Sim, so that its modules and the MushroomRL environments built on them can be imported.

        Calling this a second time does nothing but return the running app: a process can only hold one.

        Args:
            headless (bool, True): Whether to run without a window.
            physics_engine (str, 'physx'): The physics engine to simulate with, either 'physx' or 'newton'. Isaac Sim
                documents Newton as experimental, and MushroomRL does not test it: the robot assets shipped
                here are tuned for PhysX and the two engines do not produce the same dynamics.
            log_level (str, 'warning'): The lowest severity Isaac Sim prints to the console, one of 'verbose',
                'info', 'warning', 'error' or 'fatal'. Below 'info', Isaac Sim's startup messages are not printed
                either. The log file written by Isaac Sim is not affected.
            carb_settings (dict, None): Overrides for the default carb settings applied at startup, see
                :meth:`_apply_carb_settings`. Keys are carb setting paths (e.g. ``"/physics/fabricEnabled"``);
                values override the corresponding default, and unknown keys are simply added.

        Returns:
            The running simulation app.

        """
        if cls._app is None:
            log_args = cls._log_args(log_level)

            with open(os.devnull, 'w') as devnull:
                # Isaac Sim prints its launch arguments with Python's print while starting up
                stdout = contextlib.nullcontext() if cls._is_verbose(log_level) else contextlib.redirect_stdout(devnull)

                with stdout:
                    cls._app = SimulationApp({"headless": headless, "hide_ui": False, "renderer": "RaytracedLighting",
                                              "extra_args": ["--/persistent/app/usd/muteUsdDiagnostics=false",
                                                             *log_args]})
            cls._apply_carb_settings(cls._app, carb_settings)
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

    @staticmethod
    def _log_args(log_level):
        """
        Builds the command line arguments setting the console verbosity of Isaac Sim.

        Args:
            log_level (str): The lowest severity to print, one of 'verbose', 'info', 'warning', 'error' or 'fatal'.

        Returns:
            The list of command line arguments to pass to the simulation app.

        Raises:
            ValueError: If ``log_level`` is not one of the supported levels.

        """
        carb_levels = dict(verbose='Verbose', info='Info', warning='Warning', error='Error', fatal='Fatal')

        if log_level not in carb_levels:
            raise ValueError(f"Unknown log_level '{log_level}', expected one of {list(carb_levels)}.")

        # Kit prints its startup messages (extension startups, app ready, ...) straight to stdout, bypassing the
        # log level, while recording them as Info in its log file
        verbose_stdout = IsaacLauncher._is_verbose(log_level)

        return [f"--/log/outputStreamLevel={carb_levels[log_level]}",
                f"--/app/enableStdoutOutput={str(verbose_stdout).lower()}"]

    @staticmethod
    def _is_verbose(log_level):
        """
        Args:
            log_level (str): The lowest severity to print.

        Returns:
            Whether Isaac Sim's startup messages are printed at ``log_level``.

        """
        return log_level in ['verbose', 'info']

    @staticmethod
    def _apply_carb_settings(simulation_app, overrides=None):
        """
        Apply mushroom default settings for optimization.

        Args:
            simulation_app: The running simulation app.
            overrides (dict, None): Carb setting paths overriding the defaults below, or adding new ones.

        """
        headless = simulation_app.config["headless"]
        settings = {
            "/app/useFabricSceneDelegate": True,
            "/app/runLoops/main/rateLimitEnabled": False,
            "/persistent/omnihydra/useSceneGraphInstancing": True,
            "/persistent/simulation/minFrameRate": 15,
            "/exts/omni.replicator.core/Orchestrator/enabled": headless,
            "/metricsAssembler/changeListenerEnabled": False,
            "/physics/physxDispatcher": True,
            "/physics/disableContactProcessing": True,
            "/physics/collisionConeCustomGeometry": False,
            "/physics/collisionCylinderCustomGeometry": False,
            "/physics/fabricEnabled": True,
            "/physics/updateToUsd": False,
            "/physics/updateParticlesToUsd": False,
            "/physics/updateVelocitiesToUsd": False,
            "/physics/updateForceSensorsToUsd": False,
            "/physics/outputVelocitiesLocalSpace": False,
            "/physics/useFastCache": False,
            "/physics/visualizationDisplayJoints": False,
            "/physics/fabricUpdateTransformations": not headless,
            "/physics/fabricUpdateVelocities": not headless,
            "/physics/fabricUpdateForceSensors": not headless,
            "/physics/fabricUpdateJointStates": not headless,
            "/physics/fabricUseGPUInterop": True,
            "/physics/resourcemonitor/timeBetweenQueries": 100,
            "/rtx/hydra/readTransformsFromFabricInRenderDelegate": True,
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
        # Import basic Isaac Sim libraries.
        import isaacsim.core.experimental.utils.app as app_utils
        from isaacsim.core.simulation_manager import SimulationManager

        if physics_engine != SimulationManager.get_active_physics_engine():
            # Newton ships disabled outside its own launch script, so it has to be brought up by hand
            if physics_engine == 'newton':
                app_utils.enable_extension('isaacsim.physics.newton')
                app_utils.enable_extension('isaacsim.physics.newton.tensors')
            SimulationManager.switch_physics_engine(physics_engine)
