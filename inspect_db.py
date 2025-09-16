import sqlite3
import base64

DB_PATH = "faces.db"  # adjust path if needed

def inspect_faces(limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, created_at, length(embedding), length(image) FROM faces LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()

    print("---- Stored Faces ----")
    for row in rows:
        id_, name, created_at, emb_len, img_len = row
        print(f"ID={id_}, Name={name}, Created={created_at}, EmbeddingBytes={emb_len}, ImageBytes={img_len}")

def dump_one_image(face_id, out_file="face.jpg"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT image FROM faces WHERE id=?", (face_id,))
    row = c.fetchone()
    conn.close()

    if row and row[0]:
        with open(out_file, "wb") as f:
            f.write(row[0])
        print(f"Image for face {face_id} dumped to {out_file}")
    else:
        print("No image found for that ID.")

if __name__ == "__main__":
    inspect_faces()
    # Example: dump one image with ID=1
    # dump_one_image(1)
