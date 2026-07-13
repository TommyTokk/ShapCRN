from shapcrn.cli import main
from shapcrn.utils.utils import parse_args


def test_parser_uses_integer_sobol_samples():
    args = parse_args(
        [
            "sensitivity_analysis",
            "model.xml",
            "--input-species",
            "S1",
            "--base-samples",
            "128",
        ]
    )
    assert args.base_samples == 128
    assert isinstance(args.base_samples, int)
    assert not hasattr(args, "operation")
    assert not hasattr(args, "preserve_inputs")


def test_cli_missing_file_has_nonzero_status(capsys):
    status = main(["simulate", "does-not-exist.xml"])
    captured = capsys.readouterr()
    assert status == 2
    assert "does not exist" in captured.err


def test_cli_help_exits_successfully():
    try:
        main(["-h"])
    except SystemExit as error:
        assert error.code == 0
