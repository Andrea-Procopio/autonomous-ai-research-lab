"""``arl`` — the front door.

Six verbs, and no state of their own::

    arl run CONFIG --root DIR [--lab module:factory] [--stop-after STAGE]
    arl resume [INVESTIGATION] --root DIR [--lab module:factory]
    arl status [INVESTIGATION] --root DIR
    arl verify --root DIR
    arl packet [INVESTIGATION] --root DIR [--out DIR]
    arl manuscript [INVESTIGATION] --root DIR [--lab module:factory]
        [--model NAME] [--out DIR]

``run`` records the config and walks as far as it can. ``resume`` picks
up exactly where a walk stopped, using the config the investigation
recorded rather than whatever the file says now. ``status`` prints the
stage table. ``verify`` re-checks every durable claim under the root —
the snapshots, the facts, the artifact bytes, the ledger, and the event
chains — and says so or lists what is wrong. ``packet`` exports the
evidence packet: it verifies the run from cold, re-derives the
statistician's figures against the record, and writes
``packet/<packet_id>.json`` and ``.md`` under the root — checked, not
copied, and refused outright for a walk that never reached a research
state. ``manuscript`` authors the workshop draft from that packet: a
model writes prose only, behind deterministic gates that refuse any
number the packet does not state and any citation outside its
bibliography; trusted code assembles everything else. Refused for a
walk with nothing to report; re-running replays the recorded draft
without a model call.

Exit codes are meant to be read by a script as well as a person: ``0``
for a walk that ended on its own terms, including an honest scientific
no; ``2`` for a refusal, which is a precondition that was not met rather
than a fault; ``1`` for a failure, an unusable config, or a verification
that found something.

The investigation argument may be omitted when a root holds exactly one.
Where it holds several, the command lists them and stops rather than
guessing which run the operator meant.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ..program.integrity import verify_run
from ..publication.manuscript import ManuscriptError, NothingToReportError
from ..publication.packet import PacketError
from ..runtime.providers import ModelProviderError
from .config import ConfigError, load_config
from .controller import Controller, ControllerError, Outcome, StatusReport
from .investigation import InvestigationStore
from .lab import Lab, LabError, load_lab
from .manuscript import author_manuscript
from .packet import build_packet, write_packet
from .stage import CHAIN_ORDER, MissingFactError, StageName, StageStatus

OK: int = 0
FAILED: int = 1
REFUSED: int = 2

_CONTROL = "control"
_PROGRAM = "program"
_PACKET = "packet"

_EXIT_FOR = {
    Outcome.COMPLETED: OK,
    Outcome.STOPPED: OK,
    Outcome.ENDED: OK,
    Outcome.REFUSED: REFUSED,
    Outcome.FAILED: FAILED,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "run":
            return _run(arguments)
        if arguments.command == "resume":
            return _resume(arguments)
        if arguments.command == "status":
            return _status(arguments)
        if arguments.command == "packet":
            return _packet(arguments)
        if arguments.command == "manuscript":
            return _manuscript(arguments)
        return _verify(arguments)
    except (ConfigError, ControllerError, LabError) as error:
        print(f"FATAL: {error}")
        return FAILED


# -- the verbs ----------------------------------------------------------------


def _run(arguments: argparse.Namespace) -> int:
    config, payload = load_config(arguments.config)
    root = Path(arguments.root).resolve()
    controller = Controller(root)
    stop_after = (
        StageName(arguments.stop_after) if arguments.stop_after else None
    )
    investigation = controller.begin(payload)
    print(f"investigation {investigation.investigation_id}")
    print(f"config        {investigation.config_id}")
    print(f"label         {config.label}")
    print(f"root          {root}")
    print()
    result = controller.walk(
        investigation, lab=_lab(arguments), stop_after=stop_after
    )
    _print_status(controller.status(investigation.investigation_id))
    print()
    print(f"{result.outcome}: {result.detail}")
    return _EXIT_FOR[result.outcome]


def _resume(arguments: argparse.Namespace) -> int:
    root = Path(arguments.root).resolve()
    if not _exists(root):
        return FAILED
    controller = Controller(root)
    investigation_id = _chosen(root, arguments.investigation)
    if investigation_id is None:
        return FAILED
    result = controller.resume(
        investigation_id,
        lab=_lab(arguments),
        stop_after=(
            StageName(arguments.stop_after) if arguments.stop_after else None
        ),
    )
    _print_status(controller.status(investigation_id))
    print()
    print(f"{result.outcome}: {result.detail}")
    return _EXIT_FOR[result.outcome]


def _status(arguments: argparse.Namespace) -> int:
    root = Path(arguments.root).resolve()
    if not _exists(root):
        return FAILED
    investigation_id = _chosen(root, arguments.investigation)
    if investigation_id is None:
        return FAILED
    _print_status(Controller(root).status(investigation_id))
    return OK


def _verify(arguments: argparse.Namespace) -> int:
    root = Path(arguments.root).resolve()
    if not _exists(root):
        return FAILED
    report = verify_run(root, program_root=root / _PROGRAM)
    print(f"root            {root}")
    print(f"states checked  {report.states_checked}")
    print(f"results checked {report.results_checked}")
    print(f"evidence        {report.evidence_checked}")
    print(f"blobs checked   {report.blobs_checked}")
    logs, log_issues = _verify_logs(root)
    print(f"event logs      {logs}")
    for issue in report.issues:
        print(f"  {issue.kind}: {issue.subject_id}: {issue.detail}")
    for detail in log_issues:
        print(f"  event log: {detail}")
    if report.issues or log_issues:
        print()
        print(f"{len(report.issues) + len(log_issues)} issue(s) found")
        return FAILED
    print()
    print("intact")
    return OK


def _packet(arguments: argparse.Namespace) -> int:
    root = Path(arguments.root).resolve()
    if not _exists(root):
        return FAILED
    investigation_id = _chosen(root, arguments.investigation)
    if investigation_id is None:
        return FAILED
    try:
        packet = build_packet(root, investigation_id)
    except MissingFactError as error:
        # A precondition, not a fault: the walk never produced a research
        # state, so there is honestly nothing to export.
        print(f"REFUSED: {error}")
        return REFUSED
    except PacketError as error:
        print(f"FATAL: {error}")
        return FAILED
    out_dir = (
        Path(arguments.out).resolve() if arguments.out else root / _PACKET
    )
    json_path, markdown_path = write_packet(packet, out_dir)
    print(f"packet        {packet.packet_id}")
    print(f"investigation {packet.provenance.investigation_id}")
    print(f"run           {packet.provenance.run_id}")
    print(f"claims        {len(packet.claims)}")
    print(f"json          {json_path}")
    print(f"markdown      {markdown_path}")
    return OK


def _manuscript(arguments: argparse.Namespace) -> int:
    root = Path(arguments.root).resolve()
    if not _exists(root):
        return FAILED
    investigation_id = _chosen(root, arguments.investigation)
    if investigation_id is None:
        return FAILED
    try:
        result = author_manuscript(
            root,
            investigation_id,
            lab=_lab(arguments),
            out_dir=Path(arguments.out).resolve() if arguments.out else None,
            model=arguments.model,
        )
    except (MissingFactError, NothingToReportError) as error:
        # Preconditions, not faults: no research state, or a packet
        # with no claims — there is honestly nothing to write.
        print(f"REFUSED: {error}")
        return REFUSED
    except (PacketError, ManuscriptError, ModelProviderError) as error:
        print(f"FATAL: {error}")
        return FAILED
    manuscript = result.manuscript
    print(f"manuscript    {manuscript.manuscript_id}")
    print(f"packet        {manuscript.packet_id}")
    print(
        f"model         {manuscript.call.requested_model} "
        f"(served {manuscript.call.served_model} via "
        f"{manuscript.call.provider})"
    )
    print(
        f"spend         {manuscript.call.input_tokens}/"
        f"{manuscript.call.output_tokens} tokens in/out, "
        f"{manuscript.call.repair_count} corrective call(s)"
    )
    print(f"replayed      {str(result.replayed).lower()}")
    print(f"json          {result.json_path}")
    print(f"markdown      {result.markdown_path}")
    return OK


# -- helpers ------------------------------------------------------------------


def _verify_logs(root: Path) -> tuple[int, list[str]]:
    """Walk every investigation's event chain. The log verifies itself on
    read, so this is a matter of reading all of them and reporting what
    refuses."""
    store = InvestigationStore(root / _CONTROL)
    issues: list[str] = []
    checked = 0
    try:
        investigations = store.investigations()
    except Exception as error:  # reported, never re-raised
        return 0, [f"{type(error).__name__}: {error}"]
    for investigation in investigations:
        try:
            store.log_for(investigation.investigation_id).events()
            checked += 1
        except Exception as error:  # reported, never re-raised
            issues.append(
                f"{investigation.investigation_id}: "
                f"{type(error).__name__}: {error}"
            )
    return checked, issues


def _exists(root: Path) -> bool:
    """The reading verbs refuse a root that is not there.

    Not merely tidier: every store in this repository creates its own
    directories, so a mistyped path would otherwise be answered with a
    freshly made empty root and the word "intact".
    """
    if root.is_dir():
        return True
    print(f"FATAL: no run root at {root}")
    return False


def _chosen(root: Path, named: str | None) -> str | None:
    """The investigation the operator meant, or a listing and a stop."""
    if named:
        return str(named)
    found = InvestigationStore(root / _CONTROL).investigations()
    if len(found) == 1:
        return found[0].investigation_id
    if not found:
        print(f"FATAL: no investigation under {root}")
        return None
    print(f"FATAL: {root} holds {len(found)} investigations; name one:")
    for investigation in found:
        print(f"  {investigation.investigation_id}  {investigation.label}")
    return None


def _lab(arguments: argparse.Namespace) -> Lab | None:
    return load_lab(arguments.lab) if arguments.lab else None


def _print_status(report: StatusReport) -> None:
    print(f"investigation {report.investigation.investigation_id}")
    print(f"label         {report.investigation.label}")
    print()
    print(f"{'stage':<16} {'status':<10} {'calls':>6}  detail")
    for line in report.lines:
        calls = (
            "-"
            if line.status is StageStatus.PENDING
            else str(line.spend.model_calls)
        )
        print(
            f"{line.stage:<16} {line.status:<10} {calls:>6}  "
            f"{line.detail[:60]}"
        )
    print()
    print(
        f"spend: {report.spend.model_calls} model call(s), "
        f"{report.spend.input_tokens} in / {report.spend.output_tokens} out"
    )
    for name, value in sorted(report.facts.as_mapping().items()):
        print(f"  {name:<24} {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arl", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="start an investigation")
    run.add_argument("config", type=Path, help="the run config, JSON")
    _add_root(run)
    _add_lab(run)
    _add_stop_after(run)

    resume = commands.add_parser("resume", help="continue an investigation")
    resume.add_argument("investigation", nargs="?", help="investigation id")
    _add_root(resume)
    _add_lab(resume)
    _add_stop_after(resume)

    status = commands.add_parser("status", help="what happened so far")
    status.add_argument("investigation", nargs="?", help="investigation id")
    _add_root(status)

    verify = commands.add_parser("verify", help="re-check every durable claim")
    _add_root(verify)

    packet = commands.add_parser("packet", help="export the evidence packet")
    packet.add_argument("investigation", nargs="?", help="investigation id")
    _add_root(packet)
    packet.add_argument(
        "--out",
        type=Path,
        help="directory for the packet files (default: ROOT/packet)",
    )

    manuscript = commands.add_parser(
        "manuscript", help="author the manuscript from the evidence packet"
    )
    manuscript.add_argument("investigation", nargs="?", help="investigation id")
    _add_root(manuscript)
    _add_lab(manuscript)
    manuscript.add_argument(
        "--model",
        help=(
            "model for the writing seat (default: the investigation's "
            "recorded model); the manuscript records both the requested "
            "and the served model"
        ),
    )
    manuscript.add_argument(
        "--out",
        type=Path,
        help="directory for the manuscript files (default: ROOT/manuscript)",
    )
    return parser


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root", type=Path, required=True, help="the run root"
    )


def _add_stop_after(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stop-after",
        choices=[str(stage) for stage in CHAIN_ORDER],
        help=(
            "halt this walk after that stage; a brake, not a scope — "
            "resuming without it continues past"
        ),
    )


def _add_lab(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lab",
        help=(
            "module:factory supplying providers and a runtime; without it "
            "the chain stops at the funded run"
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
