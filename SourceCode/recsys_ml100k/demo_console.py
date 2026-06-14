"""
demo_console.py — Demo giao diện terminal
Chạy: python demo_console.py
Không cần server, không cần trình duyệt.
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from models import RecommendationSystem

S1 = "═" * 60
S2 = "─" * 60
def stars(r): s=int(round(r)); return "★"*s+"☆"*(5-s)

def main():
    print("\n"+S1)
    print("  CINEREC  ·  MovieLens ml-100k  ·  Console Demo")
    print(S1)

    rs = RecommendationSystem()
    st = rs.stats()

    print(f"\n  {'Dataset':<20} ml-100k (GroupLens)")
    print(f"  {'Users':<20} {st['n_users']}")
    print(f"  {'Movies':<20} {st['n_movies']}")
    print(f"  {'Ratings':<20} {st['n_ratings']:,}")
    print(f"  {'Avg rating':<20} {st['avg_rating']}")
    print(f"  {'Sparsity':<20} {st['sparsity']}%")

    print(f"\n  {'Mô hình':<24} {'MAE':>7} {'RMSE':>7} {'P@10':>7} {'R@10':>7}")
    print("  "+S2)
    for k, lab in [("ubcf","User-based CF"),("ibcf","Item-based CF"),("mf","Matrix Factorization")]:
        m = st["metrics"][k]
        p = m.get("precision@10","—"); r = m.get("recall@10","—")
        print(f"  {lab:<24} {m['mae']:>7} {m['rmse']:>7} {str(p):>7} {str(r):>7}")

    METHODS = {"1":"ubcf","2":"ibcf","3":"mf"}
    MLABEL  = {"ubcf":"User-based CF","ibcf":"Item-based CF","mf":"Matrix Factorization"}

    while True:
        print("\n"+S1)
        uid_s = input("  Nhập User ID (1–943) hoặc 'q' thoát: ").strip()
        if uid_s.lower() in ("q","quit","exit"): print("  Tạm biệt!\n"); break
        try: uid = int(uid_s)
        except: print("  ✗ Phải là số nguyên."); continue
        if uid not in rs.uid2i: print(f"  ✗ User {uid} không có trong train set."); continue

        print("  Phương pháp: [1] User-CF  [2] Item-CF  [3] MF (SVD)")
        m  = METHODS.get(input("  Chọn (mặc định 3): ").strip() or "3", "mf")
        try: n = int(input("  Số gợi ý (mặc định 10): ").strip() or "10")
        except: n = 10

        print(f"\n  Đang tính... ({MLABEL[m]})")
        recs = rs.recommend(uid, method=m, n=n)

        print(f"\n  TOP-{len(recs)} GỢI Ý CHO USER {uid}  [{MLABEL[m]}]")
        print(f"  {'#':<4} {'Tên phim':<44} {'⭐':>5} {'Score':>6}")
        print("  "+S2)
        for i,(mid,_) in enumerate([(r["movieId"],r) for r in recs],1):
            r = recs[i-1]
            t = r["title"][:42]
            print(f"  {i:<4} {t:<44} {stars(r['predicted_rating']):>5} {r['predicted_rating']:>6.2f}")

        hist = rs.history(uid, n=5)
        if hist:
            print(f"\n  Lịch sử xem gần đây (rating cao nhất):")
            print("  "+S2)
            for h in hist:
                print(f"  {stars(h['rating'])} ({h['rating']}/5)  {h['title'][:50]}")

if __name__ == "__main__":
    main()
