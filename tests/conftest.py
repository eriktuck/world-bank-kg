import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Update golden output files"
    )

@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")