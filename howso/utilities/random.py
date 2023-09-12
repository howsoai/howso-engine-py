from secrets import randbits


def get_random_seed() -> int:
    """
    Simply return a random seed.

    Provide a cryptographically secure, 64-bit random integer using the
    `secrets` module, which will use hardware-based (when available) entropy to
    generate random states.

    Supplied here for use throughout Howso client software.
    """
    return randbits(64)
