"""
models.py — Recommendation Engine (MovieLens ml-100k)
User-based CF | Item-based CF | Matrix Factorization (Funk SVD)
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────

def load_data(ratings_path="data/ratings.csv", movies_path="data/movies.csv"):
    df_r = pd.read_csv(ratings_path)
    df_m = pd.read_csv(movies_path)
    info = df_m.set_index("movieId")[["title","genres"]].to_dict("index")
    return df_r, df_m, info


def split_data(df, test_size=0.20):
    """Temporal split: mỗi user giữ 20% rating cuối làm test."""
    df = df.sort_values(["userId","timestamp"])
    train, test = [], []
    for _, g in df.groupby("userId"):
        n = max(1, int(len(g) * test_size))
        train.append(g.iloc[:-n])
        test.append(g.iloc[-n:])
    return (pd.concat(train).reset_index(drop=True),
            pd.concat(test).reset_index(drop=True))


def build_matrix(df_train):
    piv   = df_train.pivot_table(index="userId", columns="movieId", values="rating")
    R     = piv.fillna(0).values.astype(np.float32)
    uid2i = {u: i for i, u in enumerate(piv.index)}
    mid2j = {m: j for j, m in enumerate(piv.columns)}
    return R, uid2i, mid2j, list(piv.index), list(piv.columns)


def evaluate(df_test, predict_fn, uid2i, mid2j):
    preds, actuals = [], []
    for _, row in df_test.iterrows():
        u, m = int(row["userId"]), int(row["movieId"])
        if u in uid2i and m in mid2j:
            preds.append(predict_fn(u, m))
            actuals.append(row["rating"])
    if not preds:
        return {"mae": None, "rmse": None, "coverage": 0}
    return {
        "mae":      round(mean_absolute_error(actuals, preds), 4),
        "rmse":     round(float(np.sqrt(mean_squared_error(actuals, preds))), 4),
        "coverage": round(len(preds) / len(df_test) * 100, 1),
    }


# ─────────────────────────────────────────────────────────
#  PRECISION@K  /  RECALL@K
# ─────────────────────────────────────────────────────────

def precision_recall_at_k(df_test, recommend_fn, uid2i, K=10, threshold=4.0):
    """
    Tính Precision@K và Recall@K trên toàn bộ tập test.
    threshold: rating >= threshold coi là "relevant".
    """
    precisions, recalls = [], []
    for uid in uid2i:
        relevant = set(
            df_test[(df_test["userId"] == uid) & (df_test["rating"] >= threshold)]["movieId"]
        )
        if not relevant:
            continue
        recs = [mid for mid, _ in recommend_fn(uid, n=K)]
        hits = len(set(recs) & relevant)
        precisions.append(hits / K)
        recalls.append(hits / len(relevant))
    return {
        f"precision@{K}": round(float(np.mean(precisions)), 4) if precisions else 0,
        f"recall@{K}":    round(float(np.mean(recalls)),    4) if recalls    else 0,
    }


# ─────────────────────────────────────────────────────────
#  USER-BASED CF
# ─────────────────────────────────────────────────────────

class UserBasedCF:
    def __init__(self, K=30):
        self.K = K

    def fit(self, R, uid2i, mid2j, user_ids, movie_ids, df_train):
        self.R = R; self.uid2i = uid2i; self.mid2j = mid2j
        self.user_ids = user_ids; self.movie_ids = movie_ids
        self.df_train = df_train
        mask = (R != 0).astype(np.float32)
        self.umeans = np.true_divide(R.sum(1), mask.sum(1).clip(min=1))
        Rc   = (R - self.umeans[:, None]) * mask
        nrm  = np.linalg.norm(Rc, axis=1, keepdims=True).clip(min=1e-9)
        Rn   = Rc / nrm
        self.sim = Rn @ Rn.T
        np.fill_diagonal(self.sim, 0)

    def predict(self, user_id, movie_id):
        if user_id not in self.uid2i or movie_id not in self.mid2j:
            return float(self.umeans.mean())
        u = self.uid2i[user_id]; j = self.mid2j[movie_id]
        sims = self.sim[u].copy()
        sims[self.R[:, j] == 0] = 0
        tk = np.argsort(sims)[::-1][:self.K]
        ts = sims[tk]; pos = ts > 0
        if not pos.any(): return float(self.umeans[u])
        tk, ts = tk[pos], ts[pos]
        pred = self.umeans[u] + np.dot(ts, self.R[tk, j] - self.umeans[tk]) / ts.sum()
        return float(np.clip(pred, 1, 5))

    def recommend(self, user_id, n=10):
        if user_id not in self.uid2i: return []
        seen = set(self.df_train[self.df_train["userId"] == user_id]["movieId"])
        sc = [(m, self.predict(user_id, m)) for m in self.movie_ids if m not in seen]
        return sorted(sc, key=lambda x: x[1], reverse=True)[:n]


# ─────────────────────────────────────────────────────────
#  ITEM-BASED CF
# ─────────────────────────────────────────────────────────

class ItemBasedCF:
    def __init__(self, K=30):
        self.K = K

    def fit(self, R, uid2i, mid2j, user_ids, movie_ids, df_train):
        self.R = R; self.uid2i = uid2i; self.mid2j = mid2j
        self.user_ids = user_ids; self.movie_ids = movie_ids
        self.df_train = df_train
        mask  = (R != 0).astype(np.float32)
        umean = np.true_divide(R.sum(1), mask.sum(1).clip(min=1))
        self.umeans = umean
        Radj  = (R - umean[:, None]) * mask
        RT    = Radj.T
        nrm   = np.linalg.norm(RT, axis=1, keepdims=True).clip(min=1e-9)
        RTn   = RT / nrm
        self.sim = RTn @ RTn.T
        np.fill_diagonal(self.sim, 0)

    def predict(self, user_id, movie_id):
        if user_id not in self.uid2i or movie_id not in self.mid2j:
            return float(self.umeans.mean())
        u = self.uid2i[user_id]; j = self.mid2j[movie_id]
        sims = self.sim[j].copy()
        sims[self.R[u, :] == 0] = 0; sims[j] = 0
        tk = np.argsort(sims)[::-1][:self.K]
        ts = sims[tk]; pos = ts > 0
        if not pos.any(): return float(self.umeans[u])
        tk, ts = tk[pos], ts[pos]
        return float(np.clip(np.dot(ts, self.R[u, tk]) / ts.sum(), 1, 5))

    def recommend(self, user_id, n=10):
        if user_id not in self.uid2i: return []
        seen = set(self.df_train[self.df_train["userId"] == user_id]["movieId"])
        sc = [(m, self.predict(user_id, m)) for m in self.movie_ids if m not in seen]
        return sorted(sc, key=lambda x: x[1], reverse=True)[:n]


# ─────────────────────────────────────────────────────────
#  MATRIX FACTORIZATION (FUNK SVD + SGD)
# ─────────────────────────────────────────────────────────

class MatrixFactorization:
    def __init__(self, K=20, alpha=0.005, lambda_=0.02, n_epochs=30, seed=42):
        self.K = K; self.alpha = alpha
        self.lambda_ = lambda_; self.n_epochs = n_epochs; self.seed = seed
        self.train_loss = []; self.val_loss = []

    def fit(self, df_train, n_users, n_items, uid2i, mid2j, df_val=None):
        np.random.seed(self.seed)
        self.uid2i = uid2i; self.mid2j = mid2j; self.df_train = df_train
        self.P  = np.random.normal(0, 0.1, (n_users, self.K)).astype(np.float32)
        self.Q  = np.random.normal(0, 0.1, (n_items, self.K)).astype(np.float32)
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        self.mu = float(df_train["rating"].mean())

        rows = np.array([
            (uid2i[int(r.userId)], mid2j[int(r.movieId)], r.rating)
            for r in df_train.itertuples()
            if int(r.userId) in uid2i and int(r.movieId) in mid2j
        ], dtype=np.float32)

        for ep in range(1, self.n_epochs + 1):
            np.random.shuffle(rows)
            sq = 0.0
            for u, i, rui in rows:
                u, i = int(u), int(i)
                e = rui - (self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])
                sq += e * e
                pu = self.P[u].copy()
                self.P[u]  += self.alpha * (e * self.Q[i] - self.lambda_ * self.P[u])
                self.Q[i]  += self.alpha * (e * pu        - self.lambda_ * self.Q[i])
                self.bu[u] += self.alpha * (e              - self.lambda_ * self.bu[u])
                self.bi[i] += self.alpha * (e              - self.lambda_ * self.bi[i])
            self.train_loss.append(float(np.sqrt(sq / len(rows))))
            if df_val is not None:
                self.val_loss.append(self._rmse(df_val))

    def predict(self, user_id, movie_id):
        if user_id not in self.uid2i or movie_id not in self.mid2j:
            return self.mu
        u = self.uid2i[user_id]; i = self.mid2j[movie_id]
        return float(np.clip(self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i], 1, 5))

    def _rmse(self, df):
        e = [(r.rating - self.predict(int(r.userId), int(r.movieId))) ** 2
             for r in df.itertuples()
             if int(r.userId) in self.uid2i and int(r.movieId) in self.mid2j]
        return float(np.sqrt(np.mean(e))) if e else float("nan")

    def recommend(self, user_id, n=10):
        if user_id not in self.uid2i: return []
        seen = set(self.df_train[self.df_train["userId"] == user_id]["movieId"])
        sc = [(m, self.predict(user_id, m)) for m in self.mid2j if m not in seen]
        return sorted(sc, key=lambda x: x[1], reverse=True)[:n]


# ─────────────────────────────────────────────────────────
#  FACADE
# ─────────────────────────────────────────────────────────

class RecommendationSystem:
    def __init__(self, ratings_path="data/ratings.csv", movies_path="data/movies.csv"):
        print("  [1/4] Đọc dữ liệu ml-100k...")
        self.df_r, self.df_m, self.info = load_data(ratings_path, movies_path)
        self.df_train, self.df_test = split_data(self.df_r)
        self.R, self.uid2i, self.mid2j, self.uids, self.mids = build_matrix(self.df_train)
        nu, ni = len(self.uid2i), len(self.mid2j)

        print("  [2/4] User-based CF...")
        self.ubcf = UserBasedCF(K=30)
        self.ubcf.fit(self.R, self.uid2i, self.mid2j, self.uids, self.mids, self.df_train)

        print("  [3/4] Item-based CF...")
        self.ibcf = ItemBasedCF(K=30)
        self.ibcf.fit(self.R, self.uid2i, self.mid2j, self.uids, self.mids, self.df_train)

        print("  [4/4] Matrix Factorization (SGD 30 epochs)...")
        self.mf = MatrixFactorization(K=20, alpha=0.005, lambda_=0.02, n_epochs=30)
        self.mf.fit(self.df_train, nu, ni, self.uid2i, self.mid2j, self.df_test)

        print("  Đánh giá mô hình...")
        self.metrics = {
            "ubcf": evaluate(self.df_test, self.ubcf.predict, self.uid2i, self.mid2j),
            "ibcf": evaluate(self.df_test, self.ibcf.predict, self.uid2i, self.mid2j),
            "mf":   evaluate(self.df_test, self.mf.predict,   self.uid2i, self.mid2j),
        }

        # Precision@K / Recall@K (dùng subset 200 user để nhanh)
        sample_uids = list(self.uid2i.keys())[:200]
        df_test_sub = self.df_test[self.df_test["userId"].isin(sample_uids)]
        uid2i_sub   = {u: i for u, i in self.uid2i.items() if u in sample_uids}
        for name, model in [("ubcf", self.ubcf), ("ibcf", self.ibcf), ("mf", self.mf)]:
            pr = precision_recall_at_k(df_test_sub, model.recommend, uid2i_sub, K=10)
            self.metrics[name].update(pr)

        print("  Sẵn sàng!\n")

    # ── Public API ──────────────────────────────────────
    def recommend(self, user_id: int, method="mf", n=10):
        model = {"ubcf": self.ubcf, "ibcf": self.ibcf, "mf": self.mf}[method]
        raw   = model.recommend(user_id, n=n)
        result = []
        for mid, score in raw:
            inf = self.info.get(mid, {})
            result.append({
                "movieId":          mid,
                "title":            inf.get("title", f"Movie {mid}"),
                "genres":           inf.get("genres", "Unknown"),
                "predicted_rating": round(score, 2),
            })
        return result

    def history(self, user_id: int, n=8):
        rows = (self.df_train[self.df_train["userId"] == user_id]
                .sort_values("rating", ascending=False).head(n))
        out = []
        for r in rows.itertuples():
            inf = self.info.get(int(r.movieId), {})
            out.append({
                "movieId": int(r.movieId),
                "title":   inf.get("title", f"Movie {r.movieId}"),
                "genres":  inf.get("genres", ""),
                "rating":  int(r.rating),
            })
        return out

    def stats(self):
        return {
            "n_users":   len(self.uid2i),
            "n_movies":  len(self.mid2j),
            "n_ratings": len(self.df_r),
            "n_train":   len(self.df_train),
            "n_test":    len(self.df_test),
            "sparsity":  round((1 - len(self.df_r) / (len(self.uid2i) * len(self.mid2j))) * 100, 2),
            "avg_rating": round(float(self.df_r["rating"].mean()), 2),
            "metrics":   self.metrics,
            "mf_train_loss": self.mf.train_loss,
            "mf_val_loss":   self.mf.val_loss,
        }

    @property
    def all_users(self):
        return sorted(self.uid2i.keys())
