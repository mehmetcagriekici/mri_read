"""Writing one montage PNG per series plus a printed tag summary."""

from __future__ import annotations

from mri_read.mri import list_series, load_series
from mri_read.paths import OUT
from mri_read.visualize.montage import montage


def overview() -> None:
    """Write one montage PNG per series and print a one-line tag summary each."""
    OUT.mkdir(exist_ok=True)
    print(f"{'series':7} {'plane':9} {'seq':10} {'TE':>6} {'TR':>7} "
          f"{'thick':>6} {'slices':>6}  protocol")
    print("-" * 78)
    for name in list_series():
        try:
            s = load_series(name)                    # full 3D volume + tags
        except Exception as e:                       # noqa: BLE001
            print(f"{name:7} !! failed to load: {e}")
            continue
        t = s.tags
        montage(s.volume).save(OUT / f"{name}.png")  # one montage per series
        print(f"{name:7} {t['plane'][:9]:9} {t['scanning_sequence'][:10]:10} "
              f"{str(t['echo_time_TE']):>6} {str(t['repetition_TR']):>7} "
              f"{str(t['thickness_mm']):>6} {s.n_slices:>6}  {t['protocol']}")
    print(f"\nMontages written to: {OUT}")
