"""Tests for faithcheck.cli.main."""
from click.testing import CliRunner

from faithcheck.cli.main import cli


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "FaithCheck" in result.output

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--task-suite" in result.output
        assert "--output" in result.output

    def test_run_missing_args(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_dry_run_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert "--dry-run" in result.output
