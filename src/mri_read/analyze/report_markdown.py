"""Rendering an AnalysisResult as output/report.md."""

from __future__ import annotations

from mri_read.paths import OUT


def write_markdown(result, study_meta: dict) -> None:
    lines = [
        "# MRI Analysis Report (prototype)",
        "",
        f"> {result.disclaimer}",
        "",
        f"**Study:** {study_meta.get('body_part')} — {study_meta.get('model')} "
        f"@ {study_meta.get('field_T')}T",
        f"**Engine:** {result.engine}",
        f"**Sequences reviewed:** {', '.join(result.sequences_reviewed)}",
        "",
        "## Impression",
        "",
        result.impression or "_(none)_",
        "",
        "## Observations",
        "",
    ]
    if result.observations:
        for o in result.observations:
            lines.append(
                f"- **{o.get('sequence','?')}** — {o.get('finding','?')} "
                f"({o.get('location','?')}; confidence: {o.get('confidence','?')})"
            )
    else:
        lines.append("_(none reported)_")
    if result.flags:
        lines += ["", "## Flags", ""] + [f"- {f}" for f in result.flags]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
