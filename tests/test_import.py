def test_import():
    """Verify that the package can be imported."""
    import intuitive_sc

    assert intuitive_sc


def test_import_version():
    """Verify that the package version can be imported."""
    from intuitive_sc import __version__

    assert __version__


if __name__ == "__main__":
    test_import()
    test_import_version()
    print("Everything passed")
