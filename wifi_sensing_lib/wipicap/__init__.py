# Expose the compiled extension
try:
    from . import wipicap
except ImportError:
    # If standard import fails (e.g. if we are in a weird state), try absolute
    try:
        import wipicap
    except ImportError:
        pass
