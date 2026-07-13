"""git_push.py — Commit + push outputs/results/ ke GitHub, dipanggil BERKALI-KALI sepanjang
notebook (bukan cuma sekali di paling akhir).

Kenapa: sebelumnya HANYA ada satu titik push, diletakkan SETELAH seluruh tahap (termasuk
Category C v3 yang paling rawan macet). Akibatnya kalau tahap belakangan gagal/macet
berulang, hasil tahap SEBELUMNYA yang sudah stabil (tes1/tuned_v1/tuned_v2/posthoc) TIDAK
PERNAH tersimpan permanen ke GitHub -- padahal sudah beres dihitung. push_results() dipanggil
di akhir SETIAP tahap supaya progres yang sudah pasti aman langsung terarsip, tak menunggu
seluruh notebook selesai.

commit+push idempotent: kalau tak ada perubahan baru di outputs/results/, tidak melakukan
apa pun (aman dipanggil berkali-kali, termasuk berulang kali dgn tahap yang sama).
"""
from __future__ import annotations

import datetime
import subprocess


def _git(*args: str) -> int:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out)
    return r.returncode


def push_results(stage_label: str, repo_owner: str, repo_name: str) -> bool:
    """Commit + push outputs/results/ dgn pesan menyebut stage_label.

    Return True bila push berhasil ATAU memang tak ada perubahan baru (keduanya dianggap
    'aman', bukan kegagalan). Return False hanya bila commit/push benar-benar gagal.
    """
    subprocess.run(["git", "config", "user.email", "kaggle-runner@example.com"],
                   capture_output=True)
    subprocess.run(["git", "config", "user.name", "Kaggle Runner"], capture_output=True)

    _git("add", "outputs/results")
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if staged.returncode == 0:
        print(f"[push:{stage_label}] tidak ada perubahan baru di outputs/results/ -> skip.")
        return True

    code_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    msg = f"Hasil run Kaggle [{stage_label}] {datetime.datetime.now():%Y-%m-%d %H:%M} (kode @ {code_sha})"
    if _git("commit", "-m", msg) != 0:
        print(f"[push:{stage_label}] commit GAGAL.")
        return False

    token = None
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GH_TOKEN")
    except Exception:
        pass
    push_target = (f"https://{token}@github.com/{repo_owner}/{repo_name}.git"
                   if token else "origin")
    rc = _git("push", push_target, "HEAD:main")
    if rc == 0:
        print(f"[push:{stage_label}] OK -> outputs/results/ ter-push ke GitHub (permanen).")
        return True
    print(f"[push:{stage_label}] push GAGAL (lihat pesan di atas) -- commit tetap "
          f"tersimpan LOKAL di sesi ini, tak hilang, tinggal push manual nanti.")
    return False
