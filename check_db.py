# # simple_browser.py
# import sqlite3
# import os

# def simple_browser():
#     if not os.path.exists('faces.db'):
#         print("No faces.db file found!")
#         return
    
#     conn = sqlite3.connect('faces.db')
#     c = conn.cursor()
    
#     while True:
#         print("\nChoose an option:")
#         print("1. Show all records")
#         print("2. Show table structure")
#         print("3. Count records")
#         print("4. Run custom query")
#         print("5. Exit")
        
#         choice = input("Enter choice (1-5): ")
        
#         if choice == '1':
#             c.execute("SELECT id, name, created_at FROM faces")
#             for row in c.fetchall():
#                 print(f"ID: {row[0]}, Name: {row[1]}, Date: {row[2]}")
                
#         elif choice == '2':
#             c.execute("PRAGMA table_info(faces)")
#             for col in c.fetchall():
#                 print(f"{col[1]}: {col[2]}")
                
#         elif choice == '3':
#             c.execute("SELECT COUNT(*) FROM faces")
#             print(f"Total records: {c.fetchone()[0]}")
            
#         elif choice == '4':
#             query = input("Enter SQL query: ")
#             try:
#                 c.execute(query)
#                 for row in c.fetchall():
#                     print(row)
#             except Exception as e:
#                 print(f"Error: {e}")
                
#         elif choice == '5':
#             break
            
#     conn.close()

# simple_browser()





# inspect_embeddings.py
import sqlite3
import numpy as np
import zlib
from typing import List, Dict, Any

def bytes_to_emb(blob: bytes) -> np.ndarray:
    """Decompress bytes back to float32 numpy array."""
    raw = zlib.decompress(blob)
    arr = np.frombuffer(raw, dtype=np.float16)
    return arr.astype(np.float32)

def inspect_embeddings():
    """Inspect all embeddings in the database with detailed information"""
    
    db_path = 'faces.db'
    
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Check if database and table exist
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='faces'")
        if not c.fetchone():
            print("❌ No 'faces' table found in the database")
            return
        
        # Get all records
        c.execute("SELECT id, name, embedding, created_at FROM faces")
        records = c.fetchall()
        
        if not records:
            print("❌ No face records found in the database")
            return
        
        print(f"🔍 Found {len(records)} face records:")
        print("=" * 80)
        
        for record_id, name, embedding_blob, created_at in records:
            print(f"\n📊 RECORD ID: {record_id}")
            print(f"👤 Name: {name}")
            print(f"📅 Created: {created_at}")
            
            # Decode the embedding
            try:
                embedding = bytes_to_emb(embedding_blob)
                
                print(f"📐 Embedding shape: {embedding.shape}")
                print(f"📏 Embedding norm: {np.linalg.norm(embedding):.8f}")
                print(f"🔢 Data type: {embedding.dtype}")
                
                # Show statistics
                print(f"📈 Min value: {np.min(embedding):.8f}")
                print(f"📈 Max value: {np.max(embedding):.8f}")
                print(f"📈 Mean value: {np.mean(embedding):.8f}")
                print(f"📈 Std deviation: {np.std(embedding):.8f}")
                
                # Show first and last 5 values
                print(f"🔢 First 5 values: {embedding[:5]}")
                print(f"🔢 Last 5 values: {embedding[-5:]}")
                
                # Check if normalized (should be very close to 1.0)
                norm = np.linalg.norm(embedding)
                print(f"✅ Normalization check: {norm:.8f} {'✓' if 0.99 <= norm <= 1.01 else '✗'}")
                
            except Exception as e:
                print(f"❌ Error decoding embedding: {e}")
            
            print("-" * 60)
        
        # Compare embeddings if multiple exist
        if len(records) > 1:
            print("\n" + "=" * 80)
            print("🔁 COMPARING EMBEDDINGS:")
            print("=" * 80)
            
            # Decode all embeddings
            embeddings = []
            names = []
            for record_id, name, embedding_blob, created_at in records:
                try:
                    emb = bytes_to_emb(embedding_blob)
                    embeddings.append(emb)
                    names.append(name)
                except:
                    continue
            
            # Calculate similarity matrix
            print("\n🧮 Cosine Similarity Matrix:")
            print("     " + "".join([f"{n[:8]:<10}" for n in names]))
            
            for i, (emb1, name1) in enumerate(zip(embeddings, names)):
                print(f"{name1[:8]:<5}", end=" ")
                for j, (emb2, name2) in enumerate(zip(embeddings, names)):
                    if i == j:
                        similarity = 1.0
                    else:
                        similarity = float(np.dot(emb1, emb2))
                    print(f"{similarity:.4f}   ", end="")
                print()
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def export_embeddings_to_csv():
    """Export all embeddings to a CSV file for external analysis"""
    
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    
    c.execute("SELECT id, name, embedding FROM faces")
    records = c.fetchall()
    
    if not records:
        print("❌ No records to export")
        return
    
    csv_filename = "embeddings_analysis.csv"
    
    with open(csv_filename, 'w') as f:
        # Write header
        f.write("id,name,embedding_norm")
        for i in range(512):  # Assuming 512-dimensional embeddings
            f.write(f",dim_{i}")
        f.write("\n")
        
        # Write data
        for record_id, name, embedding_blob in records:
            try:
                embedding = bytes_to_emb(embedding_blob)
                norm = np.linalg.norm(embedding)
                
                f.write(f"{record_id},{name},{norm:.8f}")
                for value in embedding:
                    f.write(f",{value:.8f}")
                f.write("\n")
                
            except Exception as e:
                print(f"❌ Error processing record {record_id}: {e}")
    
    conn.close()
    print(f"✅ Embeddings exported to {csv_filename}")

if __name__ == "__main__":
    print("🔍 INSPECTING FACE EMBEDDINGS")
    print("=" * 80)
    
    inspect_embeddings()
    
    print("\n" + "=" * 80)
    print("💾 EXPORTING TO CSV")
    print("=" * 80)
    
    export_embeddings_to_csv()
    
    print("\n✅ Analysis complete!")