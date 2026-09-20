"""
Unit tests for the command-line argument list helpers in compass.turbo_compass.
These helpers are used heavily to rewrite the sys.argv-style argument list
before (re-)invoking compass.main.entry().
"""
from compass.turbo_compass import set_argument, pop_argument, get_argument


class TestSetArgument:
    def test_appends_new_argument(self):
        args = ["compass", "--data", "x.tsv"]
        set_argument(args, "--output-dir", "out")
        assert args == ["compass", "--data", "x.tsv", "--output-dir", "out"]

    def test_replaces_existing_argument(self):
        args = ["compass", "--data", "x.tsv", "--output-dir", "old"]
        set_argument(args, "--output-dir", "new")
        assert args == ["compass", "--data", "x.tsv", "--output-dir", "new"]

    def test_replaces_in_place(self):
        args = ["compass", "--data", "x.tsv", "--output-dir", "old"]
        set_argument(args, "--data", "y.tsv")
        # set_argument mutates the list in place (documented behavior)
        assert args == ["compass", "--data", "y.tsv", "--output-dir", "old"]

    def test_non_string_value_raises(self):
        args = ["compass"]
        try:
            set_argument(args, "--num-processes", 10)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for non-string value")


class TestPopArgument:
    def test_pops_existing_argument(self):
        args = ["compass", "--turbo", "0.95", "--data", "x.tsv"]
        new_args, value = pop_argument(args, "--turbo")
        assert new_args == ["compass", "--data", "x.tsv"]
        assert value == "0.95"

    def test_original_arguments_unchanged(self):
        args = ["compass", "--turbo", "0.95"]
        new_args, _ = pop_argument(args, "--turbo")
        assert args == ["compass", "--turbo", "0.95"]
        assert new_args == ["compass"]

    def test_missing_argument_returns_empty_value(self):
        args = ["compass", "--data", "x.tsv"]
        new_args, value = pop_argument(args, "--turbo")
        assert new_args == ["compass", "--data", "x.tsv"]
        assert value == ""

    def test_multiple_arguments(self):
        args = ["compass", "--a", "1", "--b", "2", "--a", "3"]
        new_args, value = pop_argument(args, "--a")
        # The first occurrence is popped
        assert new_args == ["compass", "--b", "2", "--a", "3"]
        assert value == "1"


class TestGetArgument:
    def test_existing_argument(self):
        assert get_argument(["compass", "--data", "x.tsv"], "--data") == "x.tsv"

    def test_missing_argument(self):
        assert get_argument(["compass"], "--data") == ""
