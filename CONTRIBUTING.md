# Contributing

Create a focused branch from main. Install `.[dev]`, write a numerical regression
test or independent reference for solver changes, and run the README checks
before merging. Keep the two methods on a common time-step rule when comparing
spatial dissipation. Do not substitute point samples for reference cell averages.

Rebuild the extension after C++ changes (`python -m pip install -e ".[dev]"`).
An editable installation follows Python edits but does not automatically rebuild
native code. Format Python with Ruff and C++ with the supplied clang-format style.

Regenerate figures with `fluxledger campaign --output results` only when changing
the documented experiment. Review CSV/JSON differences and visually inspect the
figure before committing. Never add course materials or private data.

Keep meaningful commits and real test evidence. CI is required before a remote
merge; local checks suffice for the first local solver merge before publishing.
No fabricated reviewers, authors, dates, or performance numbers.
