def test_import():
    """Verify that the package can be imported."""
    import fsscore

    assert fsscore


def test_import_version():
    """Verify that the package version can be imported."""
    from fsscore import __version__

    assert __version__


if __name__ == "__main__":
    test_import()
    test_import_version()
    print("Everything passed")
