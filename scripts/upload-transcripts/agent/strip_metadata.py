import dataclasses
import pathlib

import inspect_ai.log
import simple_parsing
import tqdm.contrib.concurrent


@dataclasses.dataclass
class Args:
    """Command line arguments for the script."""

    log_dir: pathlib.Path


def get_eval_logs(log_dir: pathlib.Path) -> list[pathlib.Path]:
    # Recursively walk the log directory and return all eval logs.
    # Eval logs are any files with the .eval extension
    return list(log_dir.rglob("*.eval"))


def strip_metadata(eval_log_file: pathlib.Path):
    eval_log = inspect_ai.log.read_eval_log(eval_log_file, resolve_attachments=True)

    # Remove revision information from the eval log.
    eval_log.eval.revision = None

    # Remove old eval log file.
    eval_log_file.unlink()

    # Write the eval log to the same file.
    inspect_ai.log.write_eval_log(eval_log, eval_log_file)


def main(args: Args):
    # Recursively walk the log directory and strip the metadata from each eval log
    # Eval logs are any files with the .eval extension

    eval_logs = get_eval_logs(args.log_dir)

    tqdm.contrib.concurrent.process_map(
        strip_metadata,
        eval_logs,
        desc="Stripping metadata from eval logs",
    )


if __name__ == "__main__":
    args = simple_parsing.parse(
        config_class=Args,
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    )
    main(args)
