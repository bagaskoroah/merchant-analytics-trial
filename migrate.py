import json
import re

# ---------------------------------------------------------
# 1. Update 06_dashboard_gemma.py
# ---------------------------------------------------------
with open('06_dashboard_gemma.py', 'r') as f:
    content = f.read()

# Replace anthropic import
content = content.replace('import anthropic', 'import openai')

# Find the run_agent_simple function
old_func_start = content.find('def run_agent_simple(user_message):')
old_func_end = content.find('# ============================================================', old_func_start)

# New function
new_func = '''def run_agent_simple(user_message):
    """
    Simplified agent untuk dashboard.
    Menggunakan OpenAI API compatibility (Ollama lokal) dengan tool calling.
    """
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        from pathlib import Path
        import os

        load_dotenv(Path(__file__).parent / '.env')

        # Connect to local Ollama
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "get_top_acquisition_targets",
                    "description": "Ambil daftar merchant prioritas akuisisi tertinggi",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {"type": "integer", "description": "Jumlah merchant (default 5)"},
                            "kategori": {"type": "string", "description": "Filter kategori"},
                            "kota": {"type": "string", "description": "Filter kota"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_merchant_network",
                    "description": "Detail affinity merchant berdasarkan kesamaan pelanggan",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "merchant_id": {"type": "string", "description": "ID merchant"}
                        },
                        "required": ["merchant_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_acquisition_impact",
                    "description": "Estimasi potensi MDR dan rekomendasi produk ML untuk akuisisi merchant",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "merchant_id": {"type": "string", "description": "ID merchant"}
                        },
                        "required": ["merchant_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_overview_stats",
                    "description": "Statistik keseluruhan jaringan merchant",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

        import sys
        sys.path.insert(0, PROJECT_ROOT)

        # Tool functions membaca tabel readable dan graph affinity undirected.
        def _get_top_acquisition_targets(n=5, kategori=None, kota=None):
            df = merchants_display[merchants_display['is_nasabah_bni'] == 'Tidak'].copy()
            if kategori: df = df[df['kategori'].str.lower() == kategori.lower()]
            if kota: df = df[df['kota'].str.lower() == kota.lower()]
            if len(df) == 0: return json.dumps({"error": "Tidak ada data"})
            top = df.nlargest(n, 'priority_probability').copy()
            top['rekomendasi_produk'] = top.apply(get_produk_rekomendasi, axis=1)
            return json.dumps({"targets": top[['merchant_id', 'nama', 'kategori', 'kota',
                'priority_probability', 'pagerank', 'degree', 'connected_bni_ratio',
                'n_customers', 'avg_neighbor_jaccard', 'rekomendasi_produk']].to_dict('records')},
                ensure_ascii=False, default=str)

        def _get_merchant_network(merchant_id):
            if merchant_id not in G.nodes(): return json.dumps({"error": "Not found"})
            neighbors = [{
                "id": neighbor,
                "shared_customers": int(G.edges[merchant_id, neighbor].get('shared_customers', 0)),
                "jaccard": float(G.edges[merchant_id, neighbor].get('jaccard', 0)),
            } for neighbor in G.neighbors(merchant_id)]
            neighbors.sort(key=lambda item: item['jaccard'], reverse=True)
            m = merchants_display[merchants_display['merchant_id'] == merchant_id].iloc[0]
            return json.dumps({"merchant": {"id": merchant_id, "nama": m.get('nama', ''),
                "pagerank": float(m.get('pagerank', 0)), "degree": int(m.get('degree', 0)),
                "n_customers": int(m.get('n_customers', 0))},
                "affinity_neighbors": neighbors}, ensure_ascii=False, default=str)

        def _calculate_acquisition_impact(merchant_id):
            info = merchants_display[merchants_display['merchant_id'] == merchant_id]
            if len(info) == 0: return json.dumps({"error": "Not found"})
            row = info.iloc[0]
            monthly_omzet = float(row.get('avg_omzet_bulanan', 0) or 0)
            fee = monthly_omzet * 0.007
            return json.dumps({
                "merchant_id": merchant_id,
                "monthly_omzet": monthly_omzet,
                "est_monthly_fee": float(fee),
                "est_annual_fee": float(fee * 12),
                "rekomendasi_produk": get_produk_rekomendasi(row),
                "recommendation_source": "product_rec_model",
                "affinity_neighbors": int(row.get('degree', 0)),
                "connected_bni_ratio": float(row.get('connected_bni_ratio', 0)),
            }, ensure_ascii=False, default=str)

        def _get_overview_stats():
            total = len(merchants_display); bni = (merchants_display['is_nasabah_bni'] == 'Ya').sum()
            return json.dumps({"total": total, "bni": int(bni), "non_bni": int(total - bni),
                "penetration": round(bni / total * 100, 1),
                "merchant_dalam_affinity_graph": GRAPH_COVERAGE,
                "jumlah_relasi_affinity": G.number_of_edges(),
                "product_model_experimental": PRODUCT_MODEL_EXPERIMENTAL}, default=str)

        tools_map = {
            "get_top_acquisition_targets": _get_top_acquisition_targets,
            "get_merchant_network": _get_merchant_network,
            "calculate_acquisition_impact": _calculate_acquisition_impact,
            "get_overview_stats": _get_overview_stats,
        }

        system = ("Kamu adalah AI assistant untuk tim Sales & Merchant Acquisition di Bank BNI. "
                  "Gunakan tools untuk menjawab. Jawab dalam Bahasa Indonesia yang profesional. "
                  "Sertakan angka konkret dan gunakan rekomendasi produk dari model ML. "
                  "Graph menunjukkan kesamaan pelanggan antarmerchant, bukan transfer dana langsung. "
                  "JANGAN gunakan istilah teknis seperti 'supplier', 'retailer', 'node', 'edge', 'hub', "
                  "'centrality', 'pagerank', atau 'jaccard' — gunakan bahasa bisnis seperti "
                  "'merchant dengan pelanggan serupa', 'tingkat kesamaan pelanggan', dan 'tingkat pengaruh'.")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message}
        ]

        for _ in range(5):
            resp = client.chat.completions.create(
                model="gemma4:e2b",
                messages=messages,
                tools=tool_schemas,
                temperature=0.0
            )
            
            message = resp.choices[0].message
            messages.append(message)

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    fn = tools_map.get(tool_call.function.name)
                    if fn:
                        try:
                            import json
                            args = json.loads(tool_call.function.arguments)
                            result = fn(**args)
                        except Exception as e:
                            result = json.dumps({"error": str(e)})
                    else:
                        result = json.dumps({"error": "Tool not found"})
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": result
                    })
            else:
                return message.content

        return "Tidak dapat menyelesaikan analisis."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (f"⚠️ Asisten tidak tersedia saat ini ({type(e).__name__}). "
                "Gunakan filter dan Top Targets untuk analisis manual.")

'''

new_content = content[:old_func_start] + new_func + content[old_func_end:]
with open('06_dashboard_gemma.py', 'w') as f:
    f.write(new_content)
print("Updated 06_dashboard_gemma.py")


# ---------------------------------------------------------
# 2. Update notebooks/05_genai_integration_gemma.ipynb
# ---------------------------------------------------------
with open('notebooks/05_genai_integration_gemma.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Replace tool schemas
        if 'TOOL_SCHEMAS = [' in source:
            new_source = '''TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_top_acquisition_targets",
            "description": "Ambil daftar merchant prioritas akuisisi tertinggi. Bisa difilter per kategori dan/atau kota.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n"        : {"type": "integer", "description": "Jumlah merchant (default 5)"},
                    "kategori" : {"type": "string", "description": "Filter kategori: F&B, Fashion, Grocery, Electronics, Automotive"},
                    "kota"     : {"type": "string", "description": "Filter kota: Jakarta, Bogor, Bandung, Tangerang, Bekasi, Depok"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_merchant_network",
            "description": "Detail jaringan afinitas satu merchant — tetangga dengan customer overlap dan Jaccard tertinggi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant_id": {"type": "string", "description": "ID merchant (contoh: M0001)"}
                },
                "required": ["merchant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_acquisition_impact",
            "description": "Estimasi dampak finansial jika merchant diakuisisi serta rekomendasi produk dari model ML.",
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant_id": {"type": "string", "description": "ID merchant"}
                },
                "required": ["merchant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_overview_stats",
            "description": "Statistik keseluruhan jaringan: total merchant, penetrasi BNI, performa model.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]
print(f"Defined {len(TOOL_SCHEMAS)} tool schemas")'''
            cell['source'] = [line + "\n" for line in new_source.split('\n')]
            cell['source'][-1] = cell['source'][-1].strip()
            
        # Replace agent logic
        elif 'def run_agent(' in source:
            new_source = '''try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama") if OpenAI is not None else None

SYSTEM_PROMPT = """Kamu adalah Asisten Akuisisi Merchant untuk tim Sales Bank BNI.
Tugasmu membantu tim mengidentifikasi merchant prioritas untuk diakuisisi berdasarkan customer-merchant affinity graph.

Panduan:
- Selalu gunakan tools untuk menjawab — jangan mengarang data
- Jawab dalam Bahasa Indonesia yang profesional
- Sertakan angka konkret (estimasi fee income, jumlah koneksi)
- Gunakan rekomendasi produk ML (QRIS atau QRIS + EDC) dari tool
- Jelaskan MENGAPA merchant diprioritaskan berdasarkan posisi di affinity graph
"""

def run_agent(user_message: str, max_turns: int = 5) -> str:
    """Jalankan agentic tool-calling loop menggunakan OpenAI compatibility (Ollama lokal)."""
    if client is None:
        raise RuntimeError("OpenAI package belum terinstall.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    for _ in range(max_turns):
        response = client.chat.completions.create(
            model="gemma4:e2b",
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.0
        )
        
        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                import json
                try:
                    tool_input = json.loads(tool_call.function.arguments)
                except Exception:
                    tool_input = {}
                    
                print(f"  [Tool dipanggil] {tool_name}({tool_input})")
                if tool_name in TOOLS_REGISTRY:
                    result = TOOLS_REGISTRY[tool_name](**tool_input)
                else:
                    result = json.dumps({"error": f"Tool {tool_name} tidak ditemukan"})
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result,
                })
        else:
            return message.content
            
    return "Maaf, tidak dapat menyelesaikan analisis dalam batas iterasi."

print("Agent function siap." if client is not None else "Agent function tidak siap.")'''
            cell['source'] = [line + "\n" for line in new_source.split('\n')]
            cell['source'][-1] = cell['source'][-1].strip()

with open('notebooks/05_genai_integration_gemma.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("Updated notebooks/05_genai_integration_gemma.ipynb")
