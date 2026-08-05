try:
    from .hit import AirHockeyHitBullet
    from .defend import AirHockeyDefendBullet
    from .prepare import AirHockeyPrepareBullet
    from .repel import AirHockeyRepelBullet

    AirHockeyHitBullet.register()
    AirHockeyDefendBullet.register()
    AirHockeyPrepareBullet.register()
    AirHockeyRepelBullet.register()

    __all__ = ['AirHockeyHitBullet', 'AirHockeyDefendBullet', 'AirHockeyPrepareBullet', 'AirHockeyRepelBullet']
except ImportError:
    pass
