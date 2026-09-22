import json

import pytest

from fluxledger.cli import main


@pytest.mark.parametrize("profile", ["sine", "pulse"])
def test_run_cli(tmp_path, capsys, profile):
    main(["run", "--profile", profile, "--cells", "32", "--output", str(tmp_path)])
    report = json.loads(capsys.readouterr().out)
    assert abs(report["mass_drift"]) < 1e-13
    assert (tmp_path / "solution.csv").read_text().startswith("x,initial,final")
    assert json.loads((tmp_path / "ledger.json").read_text()) == report


def test_cli_invalid_input(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        main(["run", "--diffusivity", "-1", "--output", str(tmp_path)])
    assert raised.value.code == 2
    assert "fluxledger:" in capsys.readouterr().err
    assert not (tmp_path / "solution.csv").exists()


def test_campaign_cli(tmp_path, capsys):
    main(["campaign", "--output", str(tmp_path)])
    report = json.loads(capsys.readouterr().out)
    assert report["passed"]
    assert report["smooth_cases"] == 24
    assert (tmp_path / "transport_audit.png").stat().st_size > 10000
