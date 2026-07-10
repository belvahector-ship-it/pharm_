# Tutorial Push ke GitHub (dari nol, Windows)

Panduan ini khusus untuk folder project ini
(`C:\Users\Shireen\Downloads\Claude Work\riset2`). Ikuti berurutan.

> Istilah singkat:
> - **repo** = gudang kode (folder yang dilacak git)
> - **commit** = menyimpan "titik simpan" perubahan
> - **remote** = alamat repo di GitHub
> - **push** = mengunggah commit dari komputer ke GitHub

---

## Langkah 0 — Pastikan identitas git benar (sekali saja)

Cek identitas sekarang:
```bash
git config --global user.name
git config --global user.email
```
Kalau mau ganti agar cocok dengan akun GitHub Anda:
```bash
git config --global user.name "Nama GitHub Anda"
git config --global user.email "email-akun-github@contoh.com"
```
> Email di sini **harus** email yang terdaftar di akun GitHub Anda, supaya commit tercatat
> atas nama Anda.

---

## Langkah 1 — Buat repo kosong di GitHub (lewat web browser)

1. Buka <https://github.com/new>
2. **Repository name**: misal `molprop-ensemble`
3. Pilih **Private** (disarankan, karena ini riset paper) atau Public.
4. **JANGAN** centang "Add a README / .gitignore / license"
   (repo harus benar-benar kosong; kita sudah punya file sendiri).
5. Klik **Create repository**.
6. Salin URL yang muncul, bentuknya:
   `https://github.com/USERNAME/molprop-ensemble.git`

---

## Langkah 2 — Siapkan repo di komputer

Buka terminal **di folder project ini**, lalu:

```bash
cd "C:/Users/Shireen/Downloads/Claude Work/riset2"

git init                 # jadikan folder ini repo git (sekali saja)
git add .                # masukkan semua file ke "keranjang" commit
git commit -m "Pipeline molprop-ensemble sesuai blueprint (Fase 0-7)"
git branch -M main       # namai branch utama jadi 'main'
```

Cek isi yang akan diunggah (opsional, memastikan `outputs/` tidak ikut):
```bash
git status
```

---

## Langkah 3 — Hubungkan ke GitHub & push

Ganti URL di bawah dengan URL repo Anda dari Langkah 1:
```bash
git remote add origin https://github.com/USERNAME/molprop-ensemble.git
git push -u origin main
```

### Saat push pertama → muncul jendela login
Di Windows, biasanya muncul **jendela "Git Credential Manager"**:
- Klik **Sign in with your browser** → login GitHub → **Authorize**.
- Selesai. Kredensial tersimpan, push berikutnya tidak minta login lagi.

Kalau **tidak** muncul jendela dan diminta *username & password* di terminal:
- Username = username GitHub.
- Password = **BUKAN password akun**, tapi **Personal Access Token (PAT)**. Cara buat:
  1. Buka <https://github.com/settings/tokens> → **Generate new token (classic)**.
  2. Centang scope **`repo`**.
  3. Generate, **salin token** (hanya tampil sekali).
  4. Tempel token itu sebagai "password" saat diminta.

---

## Langkah 4 — Update berikutnya (setiap ada perubahan kode)

Cukup 3 baris ini, diulang tiap kali mau menyimpan progres ke GitHub:
```bash
git add .
git commit -m "Deskripsi singkat perubahan"
git push
```

---

## Alur setelah ini: clone ke Kaggle
Setelah kode ada di GitHub, di Kaggle Notebook:
```python
!git clone https://github.com/USERNAME/molprop-ensemble.git
%cd molprop-ensemble
```
Selengkapnya lihat **KAGGLE.md**.

---

## Error yang sering muncul

| Pesan error | Arti & solusi |
|---|---|
| `fatal: not a git repository` | Belum `git init`, atau salah folder. Jalankan `cd` ke folder project dulu. |
| `remote origin already exists` | Remote sudah pernah ditambah. Perbaiki: `git remote set-url origin <URL>`. |
| `Updates were rejected ... fetch first` | Repo GitHub sudah ada isi (mis. README dibuat saat create). Tarik dulu: `git pull origin main --allow-unrelated-histories`, selesaikan bila ada konflik, lalu `git push`. |
| `Authentication failed` | Password salah — pakai **PAT**, bukan password akun (lihat Langkah 3). |
| `Support for password authentication was removed` | Sama seperti di atas: wajib pakai PAT atau login lewat browser. |
| File besar / `outputs` ikut ter-push | Pastikan `.gitignore` ada sebelum `git add .`. Kalau terlanjur: `git rm -r --cached outputs` lalu commit lagi. |

---

## Ringkas (kalau sudah hafal)
```bash
cd "C:/Users/Shireen/Downloads/Claude Work/riset2"
git init
git add .
git commit -m "commit pertama"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```
