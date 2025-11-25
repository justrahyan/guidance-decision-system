import os
import numpy as np
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_from_directory,
)
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# SECRET KEY buat session (boleh kamu ganti)
app.config["SECRET_KEY"] = "ganti-ini-dengan-yang-lebih-random"

# =========================================
# KONFIGURASI
# =========================================
CRITERIA = [
    "AdmYear", "Age", "HSCYear", "Semester", "Scholarship",
    "Study Hours", "Study Frequency", "Learn Mode", "Social Media",
    "English Skill", "Attendance", "Probation", "Consultancy",
    "Skill Hours", "Co Curricular", "Health Issues",
    "SGPA", "CGPA", "Credits", "Family Income",
]

CRITERIA_LABELS = {
    "AdmYear": "Admission Year",
    "Age": "Age",
    "HSCYear": "HSC Passing Year",
    "Semester": "Current Semester",
    "Scholarship": "Scholarship",
    "Study Hours": "Study Hours (per day)",
    "Study Frequency": "Study Frequency (per day)",
    "Learn Mode": "Preferred Learning Mode",
    "Social Media": "Social Media (hours/day)",
    "English Skill": "English Proficiency",
    "Attendance": "Class Attendance (%)",
    "Probation": "Probation History",
    "Consultancy": "Consultancy Attendance",
    "Skill Hours": "Skill Development (hours/day)",
    "Co Curricular": "Co-Curriculum Activities",
    "Health Issues": "Health Issues",
    "SGPA": "Previous SGPA",
    "CGPA": "Current CGPA",
    "Credits": "Completed Credits",
    "Family Income": "Family Income",
}

NEED_ATTR = {
    "AdmYear": "benefit",
    "Age": "neutral",
    "HSCYear": "benefit",
    "Semester": "benefit",
    "Scholarship": "benefit",
    "Study Hours": "benefit",
    "Study Frequency": "benefit",
    "Learn Mode": "benefit",
    "Social Media": "cost",
    "English Skill": "benefit",
    "Attendance": "benefit",
    "Probation": "cost",
    "Consultancy": "benefit",
    "Skill Hours": "benefit",
    "Co Curricular": "benefit",
    "Health Issues": "cost",
    "SGPA": "benefit",
    "CGPA": "benefit",
    "Credits": "benefit",
    "Family Income": "neutral",
}

TOPSIS_ATTR = {
    "AdmYear": "benefit",
    "Age": "cost",
    "HSCYear": "benefit",
    "Semester": "cost",
    "Scholarship": "cost",
    "Study Hours": "cost",
    "Study Frequency": "cost",
    "Learn Mode": "cost",
    "Social Media": "benefit",
    "English Skill": "cost",
    "Attendance": "cost",
    "Probation": "benefit",
    "Consultancy": "cost",
    "Skill Hours": "cost",
    "Co Curricular": "cost",
    "Health Issues": "benefit",
    "SGPA": "cost",
    "CGPA": "cost",
    "Credits": "cost",
    "Family Income": "cost",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)


# =========================================
# FUNGSI UTILITAS
# =========================================

def bin_encode(series, pos_values):
    return series.apply(lambda x: 1 if str(x).strip().lower() in pos_values else 0)


def encode_english_skill(x):
    x = str(x).strip().lower()
    if x == "basic":
        return 1
    elif x == "intermediate":
        return 2
    elif x in ("advanced", "advance"):
        return 3
    return 0


def kmeans_manual(X, k=3, max_iter=100, tol=1e-6):
    X_values = X.values
    n_alt, n_feat = X_values.shape
    centroids = X_values[:k].copy()  # A1,A2,A3

    labels = np.zeros(n_alt, dtype=int)
    last_labels = None
    history = []

    for it in range(1, max_iter + 1):
        diffs = X_values[:, None, :] - centroids[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        labels = np.argmin(dists, axis=1)

        dist_df = pd.DataFrame(
            dists,
            index=X.index,
            columns=[f"C{j+1}" for j in range(k)]
        )
        dist_df["Cluster"] = ["C" + str(c + 1) for c in labels]
        history.append((it, dist_df))

        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            members = X_values[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        if last_labels is not None and np.array_equal(labels, last_labels):
            centroids = new_centroids
            break

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break

        centroids = new_centroids
        last_labels = labels.copy()

    return labels, centroids, history


def score_excel_style(row):
    s = {}
    ay = row["AdmYear"]
    if ay <= 2019: s["AdmYear"] = 50
    elif ay == 2020: s["AdmYear"] = 70
    else: s["AdmYear"] = 90

    age = row["Age"]
    if age >= 23: s["Age"] = 50
    elif age == 22: s["Age"] = 70
    else: s["Age"] = 90

    hsc = row["HSCYear"]
    if hsc <= 2018: s["HSCYear"] = 50
    elif hsc <= 2020: s["HSCYear"] = 70
    else: s["HSCYear"] = 90

    sem = row["Semester"]
    if sem <= 2: s["Semester"] = 60
    elif sem <= 4: s["Semester"] = 75
    elif sem <= 6: s["Semester"] = 85
    else: s["Semester"] = 90

    s["Scholarship"] = 90 if row["Scholarship"] == 1 else 50

    sh = row["Study Hours"]
    if sh <= 1: s["Study Hours"] = 50
    elif sh <= 2: s["Study Hours"] = 60
    elif sh <= 3: s["Study Hours"] = 70
    elif sh <= 4: s["Study Hours"] = 80
    else: s["Study Hours"] = 90

    sf = row["Study Frequency"]
    if sf == 1: s["Study Frequency"] = 60
    elif sf == 2: s["Study Frequency"] = 75
    elif sf == 3: s["Study Frequency"] = 85
    else: s["Study Frequency"] = 90

    lm = row["Learn Mode"]
    if lm == 1: s["Learn Mode"] = 90
    elif lm == 0.5: s["Learn Mode"] = 80
    else: s["Learn Mode"] = 70

    sm = row["Social Media"]
    if sm < 1: s["Social Media"] = 90
    elif sm < 2: s["Social Media"] = 80
    elif sm < 3: s["Social Media"] = 70
    elif sm < 5: s["Social Media"] = 60
    else: s["Social Media"] = 50

    eng = row["English Skill"]
    if eng >= 3: s["English Skill"] = 90
    elif eng == 2: s["English Skill"] = 80
    else: s["English Skill"] = 60

    att = row["Attendance"]
    if att <= 20: s["Attendance"] = 50
    elif att <= 40: s["Attendance"] = 60
    elif att <= 60: s["Attendance"] = 70
    elif att <= 80: s["Attendance"] = 80
    else: s["Attendance"] = 90

    s["Probation"] = 50 if row["Probation"] == 1 else 90
    s["Consultancy"] = 85 if row["Consultancy"] == 1 else 60

    h = row["Skill Hours"]
    if h == 0: s["Skill Hours"] = 60
    elif h <= 1: s["Skill Hours"] = 70
    elif h <= 2: s["Skill Hours"] = 80
    else: s["Skill Hours"] = 90

    s["Co Curricular"] = 85 if row["Co Curricular"] == 1 else 70
    s["Health Issues"] = 60 if row["Health Issues"] == 1 else 90

    sg = row["SGPA"]
    if sg <= 2: s["SGPA"] = 50
    elif sg < 2.5: s["SGPA"] = 60
    elif sg < 3: s["SGPA"] = 70
    elif sg < 3.5: s["SGPA"] = 80
    else: s["SGPA"] = 90

    cg = row["CGPA"]
    if cg <= 2: s["CGPA"] = 50
    elif cg < 2.5: s["CGPA"] = 60
    elif cg < 3: s["CGPA"] = 70
    elif cg < 3.5: s["CGPA"] = 80
    else: s["CGPA"] = 90

    cr = row["Credits"]
    if cr < 30: s["Credits"] = 60
    elif cr < 40: s["Credits"] = 75
    elif cr < 60: s["Credits"] = 85
    else: s["Credits"] = 90

    inc = row["Family Income"]
    if inc <= 20000: s["Family Income"] = 60
    elif inc <= 35000: s["Family Income"] = 70
    elif inc <= 60000: s["Family Income"] = 80
    else: s["Family Income"] = 90

    return pd.Series(s)


def topsis_from_scores(score_df, weights, benefit_flags):
    X = score_df.values.astype(float)
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1
    R = X / norm

    w = weights.reshape(1, -1)
    V = R * w

    ideal_pos = np.where(benefit_flags, V.max(axis=0), V.min(axis=0))
    ideal_neg = np.where(benefit_flags, V.min(axis=0), V.max(axis=0))

    D_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))

    C = D_neg / (D_pos + D_neg + 1e-12)
    return C, D_pos, D_neg, R, V, ideal_pos, ideal_neg


def build_pairwise_from_form(form, criteria):
    n = len(criteria)
    A = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            name = f"w_{criteria[i]}_{criteria[j]}"
            val_str = form.get(name, "").strip()
            if val_str == "":
                raise ValueError(f"Nilai AHP {name} kosong")
            try:
                val = float(val_str)
                if val <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(f"Nilai AHP {name} tidak valid")
            A[i, j] = val
            A[j, i] = 1.0 / val

    return A


def ahp_weights_from_matrix(A):
    n = A.shape[0]
    col_sums = A.sum(axis=0)
    norm_A = A / col_sums
    weights = norm_A.mean(axis=1)
    weights = weights / weights.sum()

    Aw = A @ weights
    lambda_max = (Aw / weights).mean()
    CI = (lambda_max - n) / (n - 1)
    RI = 1.59  # n=20
    CR = CI / RI
    return weights, lambda_max, CI, CR


def load_and_preprocess_data():
    dataset_path = session.get("dataset_path")
    if not dataset_path:
        raise FileNotFoundError("Dataset belum diupload")

    if not os.path.exists(dataset_path):
        session.pop("dataset_path", None)
        raise FileNotFoundError("File dataset tidak ditemukan")

    df_raw = pd.read_excel(dataset_path)

    rename_columns = {
        "University Admission year": "AdmYear",
        "Age": "Age",
        "H.S.C passing year": "HSCYear",
        "Current Semester": "Semester",
        "Do you have meritorious scholarship ?": "Scholarship",
        "How many hour do you study daily?": "Study Hours",
        "How many times do you seat for study in a day?": "Study Frequency",
        "What is your preferable learning mode?": "Learn Mode",
        "How many hour do you spent daily in social media?": "Social Media",
        "Status of your English language proficiency": "English Skill",
        "Average attendance on class": "Attendance",
        "Did you ever fall in probation?": "Probation",
        "Do you attend in teacher consultancy for any kind of academical problems?": "Consultancy",
        "How many hour do you spent daily on your skill development?": "Skill Hours",
        "Are you engaged with any co-curriculum activities?": "Co Curricular",
        "Do you have any health issues?": "Health Issues",
        "What was your previous SGPA?": "SGPA",
        "What is your current CGPA?": "CGPA",
        "How many Credit did you have completed?": "Credits",
        "What is your monthly family income?": "Family Income",
    }
    df = df_raw.rename(columns=rename_columns)

    data = df[CRITERIA].copy()
    alt_ids = [f"A{i}" for i in range(1, len(data) + 1)]
    data.index = alt_ids

    data["Scholarship"] = bin_encode(data["Scholarship"], {"yes", "1"})
    data["Probation"] = bin_encode(data["Probation"], {"yes", "1"})
    data["Consultancy"] = bin_encode(data["Consultancy"], {"yes", "1"})
    data["Co Curricular"] = bin_encode(data["Co Curricular"], {"yes", "1"})
    data["Health Issues"] = bin_encode(data["Health Issues"], {"yes", "1"})

    data["Learn Mode"] = (
        data["Learn Mode"]
        .astype(str).str.strip().str.lower()
        .apply(lambda x: 1 if x == "offline" else 0)
    )

    data["English Skill"] = data["English Skill"].apply(encode_english_skill)

    numeric_cols = [
        "AdmYear", "Age", "HSCYear", "Semester",
        "Study Hours", "Study Frequency", "Social Media",
        "Attendance", "Skill Hours", "SGPA", "CGPA",
        "Credits", "Family Income",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.astype(float)
    data = data.fillna(data.median(numeric_only=True))

    scaler = MinMaxScaler()
    norm_array = scaler.fit_transform(data[CRITERIA].values)
    norm_df = pd.DataFrame(norm_array, index=alt_ids, columns=CRITERIA)

    return data, norm_df, alt_ids


def cluster_and_label(norm_df, data):
    alt_ids = list(norm_df.index)
    labels_idx, centroids, history = kmeans_manual(norm_df, k=3)
    cluster_map = {alt_ids[i]: f"C{labels_idx[i] + 1}" for i in range(len(alt_ids))}

    cluster_ids = [0, 1, 2]
    cluster_names = [f"C{i+1}" for i in cluster_ids]
    cluster_members = {f"C{i+1}": [] for i in cluster_ids}
    for alt_id, c in cluster_map.items():
        cluster_members[c].append(alt_id)

    need_index_list = []
    for j, cname in zip(cluster_ids, cluster_names):
        centroid = centroids[j]
        risk_vals = []
        for col_idx, col in enumerate(CRITERIA):
            val = centroid[col_idx]
            attr = NEED_ATTR[col]
            if attr == "benefit":
                risk = 1.0 - val
            elif attr == "cost":
                risk = val
            else:
                risk = 0.5
            risk_vals.append(risk)
        mean_risk = float(np.mean(risk_vals))
        need_index_list.append(
            {
                "Cluster": cname,
                "Mean_Need_Index": mean_risk,
                "Count": len(cluster_members[cname]),
                "Members": ", ".join(cluster_members[cname]),
            }
        )

    need_index_df = pd.DataFrame(need_index_list).set_index("Cluster")
    ordered_clusters = need_index_df.sort_values("Mean_Need_Index", ascending=False).index.tolist()
    label_map = {}
    if len(ordered_clusters) == 3:
        label_map[ordered_clusters[0]] = "Butuh Bimbingan Tinggi"
        label_map[ordered_clusters[1]] = "Butuh Bimbingan Sedang"
        label_map[ordered_clusters[2]] = "Butuh Bimbingan Rendah"

    need_index_df["Label"] = need_index_df.index.map(label_map)

    alt_cluster_df = pd.DataFrame({
        "Alternatif": alt_ids,
        "Cluster": [cluster_map[a] for a in alt_ids],
    })
    alt_cluster_df["Label"] = alt_cluster_df["Cluster"].map(label_map)
    alt_cluster_df = alt_cluster_df.set_index("Alternatif")

    return need_index_df, alt_cluster_df, history


# =========================================
# ROUTES
# =========================================

# Halaman pertama: penjelasan sistem & kriteria
@app.route("/")
def home():
    return render_template(
        "home.html",
        criteria=CRITERIA,
        labels=CRITERIA_LABELS,
        need_attr=NEED_ATTR,
        topsis_attr=TOPSIS_ATTR,
    )

def _build_preview_vars(dataset_path):
    try:
        df_prev = pd.read_excel(dataset_path)
        preview_columns = list(map(str, df_prev.columns))
        df_head = df_prev.head(5)
        preview_rows = (
            df_head.astype(object)
                  .where(pd.notnull(df_head), "")
                  .to_dict(orient="records")
        )
        n_rows, n_cols = df_prev.shape
        return {
            "preview_columns": preview_columns,
            "preview_rows": preview_rows,
            "preview_shape": (int(n_rows), int(n_cols)),
            "preview_filename": os.path.basename(dataset_path),
            "preview_error": None,
            "uploaded": True,   # <- flag dataset sudah diupload
        }
    except Exception as e:
        return {
            "preview_columns": [],
            "preview_rows": [],
            "preview_shape": None,
            "preview_filename": None,
            "preview_error": f"Gagal membaca file: {e}",
            "uploaded": False,
        }

@app.route("/reset-dataset")
def reset_dataset():
    dataset_path = session.get("dataset_path")

    # hapus file fisik kalau masih ada
    if dataset_path and os.path.exists(dataset_path):
        try:
            os.remove(dataset_path)
        except:
            pass

    # hapus session
    session.pop("dataset_path", None)

    # kembali ke halaman upload
    return redirect(url_for("upload"))

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        dataset_path = session.get("dataset_path")
        if dataset_path and os.path.exists(dataset_path):
            preview_ctx = _build_preview_vars(dataset_path)
        else:
            preview_ctx = {
                "preview_columns": [],
                "preview_rows": [],
                "preview_shape": None,
                "preview_filename": None,
                "preview_error": None,
                "uploaded": False,
            }
        return render_template("upload.html", error=None, **preview_ctx)
    
    # POST
    file = request.files.get("dataset")
    if not file or file.filename == "":
        return render_template(
            "upload.html",
            error="Pilih file .xlsx terlebih dahulu",
            preview_columns=[],
            preview_rows=[],
            preview_shape=None,
            preview_filename=None,
            preview_error=None,
            uploaded=False,
        )

    filename = secure_filename(file.filename)
    dataset_path = os.path.join(UPLOAD_DIR, filename)
    file.save(dataset_path)

    session["dataset_path"] = dataset_path

    # setelah upload: TETAP di upload + tampilkan preview
    preview_ctx = _build_preview_vars(dataset_path)
    return render_template("upload.html", error=None, **preview_ctx)

def _clear_uploaded_dataset():
    """Hapus file dataset yang tersimpan dan bersihkan session."""
    dataset_path = session.get("dataset_path")

    if dataset_path and os.path.exists(dataset_path):
        try:
            os.remove(dataset_path)
        except Exception:
            # kalau gagal hapus file, abaikan saja supaya tidak bikin error
            pass

    session.pop("dataset_path", None)

# Halaman AHP (input pairwise)
@app.route("/ahp", methods=["GET", "POST"])
def index():
    # cek dulu dataset sudah diupload atau belum
    try:
        data, norm_df, alt_ids = load_and_preprocess_data()
    except FileNotFoundError:
        return redirect(url_for("upload"))

    # GET: tampilkan form AHP
    if request.method == "GET":
        return render_template(
            "index.html",
            criteria=CRITERIA,
            labels=CRITERIA_LABELS,
            error=None,
        )

    # POST: proses AHP + TOPSIS
    # validasi SEMUA pairwise harus terisi (i<j)
    for i in range(len(CRITERIA)):
        for j in range(i + 1, len(CRITERIA)):
            field = f"w_{CRITERIA[i]}_{CRITERIA[j]}"
            if not request.form.get(field, "").strip():
                return render_template(
                    "index.html",
                    criteria=CRITERIA,
                    labels=CRITERIA_LABELS,
                    error="Semua nilai perbandingan AHP wajib diisi (untuk i < j).",
                )

    # bangun matriks AHP
    try:
        A = build_pairwise_from_form(request.form, CRITERIA)
    except ValueError as e:
        return render_template(
            "index.html",
            criteria=CRITERIA,
            labels=CRITERIA_LABELS,
            error=str(e),
        )

    # --- AHP: bobot & konsistensi ---
    weights, lambda_max, CI, CR_val = ahp_weights_from_matrix(A)

    # list bobot untuk tampilan
    ahp_weights_list = []
    for crit, w in zip(CRITERIA, weights):
        ahp_weights_list.append({
            "kriteria": crit,
            "label": CRITERIA_LABELS[crit],
            "weight": float(w),
        })

    # clustering (tidak bergantung AHP, jadi tetap jalan)
    need_index_df, alt_cluster_df, kmeans_history = cluster_and_label(norm_df, data)

    # ====== JIKA CR TIDAK MEMENUHI < 0.1 → STOP DI SINI ======
    if CR_val >= 0.1:
        session.pop("ranking_path", None)
        # ⬇️ HAPUS DATASET SETELAH SELESAI PROSES
        _clear_uploaded_dataset()

        return render_template(
            "result.html",
            ranking=[],
            ahp_weights=ahp_weights_list,
            lambda_max=lambda_max,
            CI=CI,
            CR=CR_val,
            need_index_table=need_index_df.to_dict(orient="index"),
            ideal_detail=None,
            vij_detail=None,
            distance_detail=None,
            ahp_inconsistent=True,
            raw_detail=None,
            score_detail=None,
            norm_detail=None,
            kmeans_history=kmeans_history,
            CRITERIA=CRITERIA,
            CRITERIA_LABELS=CRITERIA_LABELS,
        )

    # ====== Lanjut TOPSIS HANYA kalau CR < 0.1 ======
    label_target = "Butuh Bimbingan Tinggi"
    high_alts = alt_cluster_df.index[alt_cluster_df["Label"] == label_target].tolist()

    if len(high_alts) == 0:
        session.pop("ranking_path", None)
        # ⬇️ HAPUS DATASET JUGA DI KASUS INI
        _clear_uploaded_dataset()

        return render_template(
            "result.html",
            ranking=[],
            ahp_weights=ahp_weights_list,
            lambda_max=lambda_max,
            CI=CI,
            CR=CR_val,
            need_index_table=need_index_df.to_dict(orient="index"),
            ideal_detail=None,
            vij_detail=None,
            distance_detail=None,
            ahp_inconsistent=False,
            raw_detail=None,
            score_detail=None,
            norm_detail=None,
            kmeans_history=kmeans_history,
            CRITERIA=CRITERIA,
            CRITERIA_LABELS=CRITERIA_LABELS,
        )

    # =========================
    # DATA ALTERNATIF BUTUH TINGGI
    # =========================
    high_data = data.loc[high_alts, CRITERIA].copy()

    # skor asumsi 50–90 (Excel style)
    score_df = high_data.apply(score_excel_style, axis=1)

    # TOPSIS
    benefit_flags = np.array([TOPSIS_ATTR[c] == "benefit" for c in CRITERIA])
    scores, D_pos, D_neg, R, V, ideal_pos, ideal_neg = topsis_from_scores(
        score_df, weights, benefit_flags
    )

    # DETAIL DATA ASLI per alternatif (sudah encoded – ini yang dipakai TOPSIS)
    raw_detail = []
    for alt_id in high_alts:
        row = high_data.loc[alt_id]
        row_vals = []
        for crit in CRITERIA:
            row_vals.append({
                "kriteria": crit,
                "label": CRITERIA_LABELS[crit],
                "value": float(row[crit]),
            })
        raw_detail.append({
            "alternatif": alt_id,
            "values": row_vals,
        })

    # DETAIL SKOR ASUMSI (50–90)
    score_detail = []
    for alt_id in high_alts:
        row = score_df.loc[alt_id]
        row_vals = []
        for crit in CRITERIA:
            row_vals.append({
                "kriteria": crit,
                "label": CRITERIA_LABELS[crit],
                "value": float(row[crit]),
            })
        score_detail.append({
            "alternatif": alt_id,
            "values": row_vals,
        })

    # DETAIL NORMALISASI R_ij
    norm_detail = []
    for i, alt_id in enumerate(high_alts):
        row_vals = []
        for j, crit in enumerate(CRITERIA):
            row_vals.append({
                "kriteria": crit,
                "label": CRITERIA_LABELS[crit],
                "value": float(R[i, j]),
            })
        norm_detail.append({
            "alternatif": alt_id,
            "values": row_vals,
        })

    # DETAIL V_ij
    vij_detail = []
    for i, alt_id in enumerate(high_alts):
        row_vals = []
        for j, crit in enumerate(CRITERIA):
            row_vals.append({
                "kriteria": crit,
                "label": CRITERIA_LABELS[crit],
                "value": float(V[i, j]),
            })
        vij_detail.append({
            "alternatif": alt_id,
            "values": row_vals,
        })

    # SOLUSI IDEAL
    ideal_detail = []
    for i, crit in enumerate(CRITERIA):
        ideal_detail.append({
            "kriteria": crit,
            "label": CRITERIA_LABELS[crit],
            "pos": float(ideal_pos[i]),
            "neg": float(ideal_neg[i]),
        })

    # DISTANCE
    distance_detail = []
    for i, alt_id in enumerate(high_alts):
        distance_detail.append({
            "alternatif": alt_id,
            "d_pos": float(D_pos[i]),
            "d_neg": float(D_neg[i]),
        })

    # RANKING
    rank_df = pd.DataFrame({
        "Alternatif": high_alts,
        "TOPSIS_Score": scores,
        "D_pos": D_pos,
        "D_neg": D_neg,
    })
    rank_df["Rank"] = rank_df["TOPSIS_Score"].rank(
        ascending=False, method="dense"
    ).astype(int)
    rank_df = rank_df.set_index("Alternatif")
    final_table = rank_df.join(alt_cluster_df[["Cluster", "Label"]], how="left")
    final_table = final_table.sort_values("Rank", ascending=True)

    ranking_json = []
    for alt_id, row in final_table.iterrows():
        ranking_json.append({
            "alternatif": alt_id,
            "score": float(row["TOPSIS_Score"]),
            "rank": int(row["Rank"]),
            "cluster": row["Cluster"],
            "label": row["Label"],
            "d_pos": float(row["D_pos"]),
            "d_neg": float(row["D_neg"]),
        })

    # SIMPAN RANKING KE EXCEL (tetap)
    export_df = final_table.reset_index()
    export_df.rename(columns={
        "Alternatif": "Alternatif",
        "TOPSIS_Score": "Skor_TOPSIS",
        "D_pos": "D_plus",
        "D_neg": "D_minus",
    }, inplace=True)
    ranking_path = os.path.join(EXPORT_DIR, "ranking_topsis.xlsx")
    export_df.to_excel(ranking_path, index=False)
    session["ranking_path"] = ranking_path

    # ⬇️ SETELAH SEMUA PERHITUNGAN BERES, HAPUS DATASET
    _clear_uploaded_dataset()

    return render_template(
        "result.html",
        ranking=ranking_json,
        ahp_weights=ahp_weights_list,
        lambda_max=lambda_max,
        CI=CI,
        CR=CR_val,
        need_index_table=need_index_df.to_dict(orient="index"),
        ideal_detail=ideal_detail,
        vij_detail=vij_detail,
        distance_detail=distance_detail,
        ahp_inconsistent=False,
        raw_detail=raw_detail,
        score_detail=score_detail,
        norm_detail=norm_detail,
        kmeans_history=kmeans_history,
        CRITERIA=CRITERIA,
        CRITERIA_LABELS=CRITERIA_LABELS,
    )

@app.route("/download-dataset")
def download_dataset():
    data_dir = os.path.join(BASE_DIR, "data")
    return send_from_directory(
        data_dir,
        "dataset.xlsx",
        as_attachment=True
    )

@app.route("/panduan")
def panduan():
    return render_template("panduan.html")

@app.route("/download-ranking")
def download_ranking():
    ranking_path = session.get("ranking_path")
    if not ranking_path or not os.path.exists(ranking_path):
        # kalau tidak ada file / session, balikin ke AHP saja
        return redirect(url_for("index"))

    directory, filename = os.path.split(ranking_path)
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
