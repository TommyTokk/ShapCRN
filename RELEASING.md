# Releasing ShapCRN

1. Confirm that the normalized `shapcrn` project name is available on PyPI. If
   it is unavailable, change only the distribution name in `pyproject.toml` to
   `shapcrn-crn`; keep the `shapcrn` import package and console command.
2. Configure GitHub trusted publishers for the `testpypi` and `pypi`
   environments. Require manual approval for the `pypi` environment.
3. Run the full CI matrix and ensure the distribution-content job passes.
4. Run the **Publish** workflow manually. It publishes to TestPyPI and installs 
   the uploaded wheel back into a clean Python 3.12 environment.
5. Inspect the TestPyPI project page and run a simulation with a real SBML model.
6. Create and push an annotated version tag, for example `v0.1.1`. The tag workflow publishes the same artifacts to production PyPI.

The TestPyPI install uses production PyPI as an additional index because the
scientific runtime dependencies are not guaranteed to exist on TestPyPI.
