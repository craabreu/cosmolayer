"""``import cosmolayer`` must stay cheap: heavy optional dependencies (torch,
lightning) should load only when the attributes that need them are actually
accessed, not merely by importing the package.

Each check runs in a fresh subprocess so an already-populated
``sys.modules`` from other tests in this session can't hide a regression.
"""

import subprocess
import sys


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_import_cosmolayer_does_not_load_torch_or_lightning() -> None:
    out = _run(
        "import sys\n"
        "import cosmolayer\n"
        "print('torch' in sys.modules)\n"
        "print('lightning' in sys.modules)\n"
    )
    assert out == "False\nFalse"


def test_import_cosmolayer_store_does_not_load_torch_or_lightning() -> None:
    out = _run(
        "import sys\n"
        "import cosmolayer.store\n"
        "print('torch' in sys.modules)\n"
        "print('lightning' in sys.modules)\n"
    )
    assert out == "False\nFalse"


def test_accessing_cosmolayer_class_loads_torch() -> None:
    out = _run(
        "import sys\n"
        "import cosmolayer\n"
        "cosmolayer.CosmoLayer\n"
        "print('torch' in sys.modules)\n"
    )
    assert out == "True"


def test_accessing_cosmosac_component_does_not_load_torch() -> None:
    out = _run(
        "import sys\n"
        "from cosmolayer.cosmosac import Component\n"
        "print('torch' in sys.modules)\n"
    )
    assert out == "False"


def test_accessing_cosmosac_datapoint_loads_torch() -> None:
    out = _run(
        "import sys\n"
        "from cosmolayer.cosmosac import CosmoSacMixtureDatapoint\n"
        "print('torch' in sys.modules)\n"
    )
    assert out == "True"


def test_unknown_attribute_still_raises_attribute_error() -> None:
    out = _run(
        "import cosmolayer\n"
        "try:\n"
        "    cosmolayer.NoSuchThing\n"
        "except AttributeError as exc:\n"
        "    print('ok:', exc)\n"
    )
    assert out.startswith("ok:")
