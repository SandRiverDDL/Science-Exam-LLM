import os
import glob
import json
import csv
import time
from typing import List, Tuple, Iterable, Dict, Any
import sys

from core.config import Config
from chunking.token_chunker import chunk_docs_with_tokenizer, tokenize_chunks, tokenize_chunks_ids
from retrieval.embedding_jina_gguf import JinaEmbeddingGGUF
from index.faiss_store import FaissIndex
from transformers import AutoTokenizer
from retrieval.embedding_hf import E5EmbeddingHF, GTEEmbeddingHF, BGEEmbeddingHF, QwenEmbeddingHF


def load_text_files(corpus_dir: str) -> List[Tuple[str, str]]:
    # 支持递归与多扩展名加载，避免仅限于 *.txt 导致空集
    docs: List[Tuple[str, str]] = []
    patterns = ["**/*.txt", "**/*.md", "**/*.text"]
    for pattern in patterns:
        for fp in glob.glob(os.path.join(corpus_dir, pattern), recursive=True):
            if not os.path.isfile(fp):
                continue
            doc_id = os.path.relpath(fp, corpus_dir)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    t = f.read().strip()
                if t:
                    docs.append((doc_id, t))
            except Exception:
                # 读取失败则跳过该文件
                continue
    return docs

def find_csv_files(corpus_dir: str) -> List[str]:
    return [fp for fp in glob.glob(os.path.join(corpus_dir, "**/*.csv"), recursive=True) if os.path.isfile(fp)]

def select_text_columns(header: List[str]) -> List[str]:
    candidates = {"text", "content", "article", "body", "paragraph", "desc", "description"}
    cols = [h for h in header if h and h.lower() in candidates]
    return cols if cols else (header or [])

def iter_csv_texts(csv_path: str, text_columns: List[str] = None, start_row: int = 0) -> Iterable[Tuple[str, str]]:
    # 提升 CSV 单字段最大长度限制，避免超长字段抛错（默认 131072）
    try:
        csv.field_size_limit(min(sys.maxsize, 1_000_000_000))  # 最高提升至约 1GB
    except Exception:
        pass
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        cols = text_columns or select_text_columns(header)
        for i, row in enumerate(reader):
            if i < start_row:
                continue
            parts: List[str] = []
            for c in cols:
                val = row.get(c, "")
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
            text = "\n\n".join(parts).strip()
            if text:
                doc_id = f"{os.path.basename(csv_path)}:row:{i}"
                yield (doc_id, text)


def save_meta(meta_path: str, meta_obj):
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False)


def build_indices_streaming(
    cfg: Config,
    tokenizer,
    docs_iter: Iterable[Tuple[str, str]],
    chunk_size: int,
    overlap: int,
    batch_size: int = 128,
    meta_jsonl_path: str = None,
    skip_keys: set = None,
):
    # 初始化嵌入后端
    jina_enabled = bool(cfg.get("indices", "jina", "enabled", default=False))
    jina_emb = None
    if jina_enabled:
        jina_emb = JinaEmbeddingGGUF(
            repo_id=cfg.embedding["repo_id"],
            filename=cfg.embedding["filename"],
            n_ctx=cfg.embedding.get("n_ctx", 8192),
            n_threads=cfg.embedding.get("n_threads", 8),
            n_batch=cfg.embedding.get("n_batch", 32),
            n_seq_max=cfg.embedding.get("n_seq_max", 128),
        )
    e5_enabled = bool(cfg.get("indices", "e5", "enabled", default=False))
    gte_enabled = bool(cfg.get("indices", "gte", "enabled", default=False))
    bge_enabled = bool(cfg.get("indices", "bge", "enabled", default=False))
    bge_small_enabled = bool(cfg.get("indices", "bge_small", "enabled", default=False))
    qwen3_enabled = bool(cfg.get("indices", "qwen3", "enabled", default=False))
    e5_model_id = cfg.get("embedding_e5", "model_id", default=None)
    gte_model_id = cfg.get("embedding_gte", "model_id", default=None)
    bge_model_id = cfg.get("embedding_bge", "model_id", default=None)
    bge_small_model_id = cfg.get("embedding_bge_small", "model_id", default=None)
    qwen3_model_id = cfg.get("embedding_qwen3", "model_id", default=None)
    if e5_enabled and not e5_model_id:
        print("[warn] E5 enabled but missing embedding_e5.model_id in config; disabling E5.")
        e5_enabled = False
    if gte_enabled and not gte_model_id:
        print("[warn] GTE enabled but missing embedding_gte.model_id in config; disabling GTE.")
        gte_enabled = False
    if bge_enabled and not bge_model_id:
        print("[warn] BGE enabled but missing embedding_bge.model_id in config; disabling BGE.")
        bge_enabled = False
    if bge_small_enabled and not bge_small_model_id:
        print("[warn] BGE-small enabled but missing embedding_bge_small.model_id in config; disabling BGE-small.")
        bge_small_enabled = False
    if qwen3_enabled and not qwen3_model_id:
        print("[warn] Qwen3 enabled but missing embedding_qwen3.model_id in config; disabling Qwen3.")
        qwen3_enabled = False
    e5_emb = E5EmbeddingHF(
        model_id=e5_model_id,
        device=cfg.get("embedding_e5", "device", default=None),
        max_length=cfg.get("embedding_e5", "max_length", default=512),
        dtype=cfg.get("embedding_e5", "dtype", default=None),
    ) if e5_enabled else None
    gte_emb = GTEEmbeddingHF(
        model_id=gte_model_id,
        device=cfg.get("embedding_gte", "device", default=None),
        max_length=cfg.get("embedding_gte", "max_length", default=512),
        dtype=cfg.get("embedding_gte", "dtype", default=None),
    ) if gte_enabled else None
    bge_emb = BGEEmbeddingHF(
        model_id=bge_model_id,
        device=cfg.get("embedding_bge", "device", default=None),
        max_length=cfg.get("embedding_bge", "max_length", default=512),
        dtype=cfg.get("embedding_bge", "dtype", default=None),
    ) if bge_enabled else None
    bge_small_emb = BGEEmbeddingHF(
        model_id=bge_small_model_id,
        device=cfg.get("embedding_bge_small", "device", default=None),
        max_length=cfg.get("embedding_bge_small", "max_length", default=512),
        dtype=cfg.get("embedding_bge_small", "dtype", default=None),
    ) if bge_small_enabled else None
    qwen3_emb = QwenEmbeddingHF(
        model_id=qwen3_model_id,
        device=cfg.get("embedding_qwen3", "device", default=None),
        max_length=cfg.get("embedding_qwen3", "max_length", default=512),
        dtype=cfg.get("embedding_qwen3", "dtype", default=None),
    ) if qwen3_enabled else None
    if e5_emb is not None:
        print(f"[init] E5 device: {getattr(e5_emb, 'device', 'unknown')}")
    if gte_emb is not None:
        print(f"[init] GTE device: {getattr(gte_emb, 'device', 'unknown')}")
    if bge_emb is not None:
        print(f"[init] BGE device: {getattr(bge_emb, 'device', 'unknown')}")
    if bge_small_emb is not None:
        print(f"[init] BGE-small device: {getattr(bge_small_emb, 'device', 'unknown')}")
    if qwen3_emb is not None:
        print(f"[init] Qwen3 device: {getattr(qwen3_emb, 'device', 'unknown')}")

    # 选择用于分块的 tokenizer：当启用直通 token ids 时，必须与目标嵌入模型的 tokenizer 保持一致
    use_token_ids_cfg = bool(cfg.get("pipeline", "embed_from_tokens", default=True))
    embed_tokens_model = str(cfg.get("pipeline", "embed_tokens_model", default="bge_small")).lower()
    chunk_tokenizer = tokenizer
    if use_token_ids_cfg:
        if embed_tokens_model == "bge_small" and bge_small_emb is not None:
            chunk_tokenizer = bge_small_emb.tokenizer
            print("[tokenize] 使用 BGE-small 的 tokenizer 进行分块（直通 token ids）")
        elif embed_tokens_model == "bge" and bge_emb is not None:
            chunk_tokenizer = bge_emb.tokenizer
            print("[tokenize] 使用 BGE 的 tokenizer 进行分块（直通 token ids）")
        elif embed_tokens_model == "gte" and gte_emb is not None:
            chunk_tokenizer = gte_emb.tokenizer
            print("[tokenize] 使用 GTE 的 tokenizer 进行分块（直通 token ids）")
        elif embed_tokens_model == "qwen3" and qwen3_emb is not None:
            chunk_tokenizer = qwen3_emb.tokenizer
            print("[tokenize] 使用 Qwen3 的 tokenizer 进行分块（直通 token ids）")
        else:
            print(f"[tokenize] 直通 token ids 需要 {embed_tokens_model} 的 tokenizer，但模型未启用或不可用，回退为文本路径")
            use_token_ids_cfg = False
        # 为分块目的提高 tokenizer 的 model_max_length，避免 HuggingFace 的超长警告
        try:
            if hasattr(chunk_tokenizer, "model_max_length"):
                chunk_tokenizer.model_max_length = max(getattr(chunk_tokenizer, "model_max_length", 512) or 512, max(chunk_size, 1_000_000_000))
        except Exception:
            pass

    # 初始化索引
    jina_index = FaissIndex(index_type="flat_ip_fp16", save_path=cfg.get("indices", "jina", "index_path")) if jina_enabled else None
    e5_index = FaissIndex(index_type="flat_ip_fp16", save_path=cfg.get("indices", "e5", "index_path")) if e5_enabled else None
    gte_index = FaissIndex(index_type="flat_ip_fp16", save_path=cfg.get("indices", "gte", "index_path")) if gte_enabled else None
    bge_index = FaissIndex(index_type="flat_ip_fp16", save_path=cfg.get("indices", "bge", "index_path")) if bge_enabled else None
    bge_small_index = FaissIndex(index_type="flat_ip_fp16", save_path=cfg.get("indices", "bge_small", "index_path")) if bge_small_enabled else None
    qwen3_index = FaissIndex(index_type="flat_ip_fp16", save_path=cfg.get("indices", "qwen3", "index_path")) if qwen3_enabled else None

    # 断点续跑：若索引文件存在则加载，以便继续追加
    resume_enabled = bool(cfg.get("pipeline", "resume", "enabled", default=False))
    if resume_enabled:
        try:
            if jina_index is not None and os.path.exists(cfg.get("indices", "jina", "index_path")):
                jina_index.load()
                print(f"[resume] Loaded Jina index: {cfg.get('indices','jina','index_path')}")
        except Exception as e:
            print(f"[resume] Skip loading Jina index: {e}")
        try:
            if e5_index is not None and os.path.exists(cfg.get("indices", "e5", "index_path")):
                e5_index.load()
                print(f"[resume] Loaded E5 index: {cfg.get('indices','e5','index_path')}")
        except Exception as e:
            print(f"[resume] Skip loading E5 index: {e}")
        try:
            if gte_index is not None and os.path.exists(cfg.get("indices", "gte", "index_path")):
                gte_index.load()
                print(f"[resume] Loaded GTE index: {cfg.get('indices','gte','index_path')}")
        except Exception as e:
            print(f"[resume] Skip loading GTE index: {e}")
        try:
            if bge_index is not None and os.path.exists(cfg.get("indices", "bge", "index_path")):
                bge_index.load()
                print(f"[resume] Loaded BGE index: {cfg.get('indices','bge','index_path')}")
        except Exception as e:
            print(f"[resume] Skip loading BGE index: {e}")
        try:
            if bge_small_index is not None and os.path.exists(cfg.get("indices", "bge_small", "index_path")):
                bge_small_index.load()
                print(f"[resume] Loaded BGE-small index: {cfg.get('indices','bge_small','index_path')}")
        except Exception as e:
            print(f"[resume] Skip loading BGE-small index: {e}")
        try:
            if qwen3_index is not None and os.path.exists(cfg.get("indices", "qwen3", "index_path")):
                qwen3_index.load()
                print(f"[resume] Loaded Qwen3 index: {cfg.get('indices','qwen3','index_path')}")
        except Exception as e:
            print(f"[resume] Skip loading Qwen3 index: {e}")

    # 超大数据集使用 JSONL 流式写出，避免内存峰值
    chunk_id = 0
    buf_texts: List[str] = []
    buf_ids: List[List[int]] = []
    buf_meta: List[Dict[str, Any]] = []
    out_jsonl = None
    if meta_jsonl_path:
        os.makedirs(os.path.dirname(meta_jsonl_path), exist_ok=True)
        out_jsonl = open(meta_jsonl_path, "w", encoding="utf-8")

    # 进度统计
    start_ts = time.time()
    bytes_total = 0
    flush_count = 0
    last_print_ts = start_ts
    max_line_len = 0

    progress_interval = float(cfg.get("pipeline", "progress_interval_sec", default=5))
    periodic_sec = int(cfg.get("pipeline", "periodic_save", "sec", default=0) or 0)
    periodic_chunks = int(cfg.get("pipeline", "periodic_save", "chunks", default=0) or 0)
    checkpoint_path = cfg.get("pipeline", "resume", "checkpoint_path", default=None)
    last_save_ts = start_ts
    last_save_chunks = 0
    use_token_ids = use_token_ids_cfg
    use_ids_for_gte = use_token_ids and (embed_tokens_model == "gte")
    use_ids_for_bge = use_token_ids and (embed_tokens_model == "bge")
    use_ids_for_bge_small = use_token_ids and (embed_tokens_model == "bge_small")
    use_ids_for_qwen3 = use_token_ids and (embed_tokens_model == "qwen3")

    def print_and_overwrite(text: str):
        nonlocal max_line_len
        # 覆盖上一行输出，控制台更清晰
        max_line_len = max(max_line_len, len(text))
        pad = " " * (max_line_len - len(text))
        sys.stdout.write("\r" + text + pad)
        sys.stdout.flush()

    def flush():
        nonlocal buf_texts, buf_ids, buf_meta, flush_count, bytes_total, last_print_ts, last_save_ts, last_save_chunks
        if not buf_texts and not buf_ids:
            return
        # 嵌入并写入索引
        if jina_enabled and jina_emb is not None:
            jina_vecs = jina_emb.embed(buf_texts)
            jina_index.build(jina_vecs)
        if e5_enabled and e5_emb is not None:
            e5_vecs = e5_emb.embed(buf_texts)
            e5_index.build(e5_vecs)
        if gte_enabled and gte_emb is not None:
            if use_ids_for_gte and buf_ids and hasattr(gte_emb, "embed_from_ids"):
                gte_vecs = gte_emb.embed_from_ids(buf_ids)
            else:
                gte_vecs = gte_emb.embed(buf_texts)
            gte_index.build(gte_vecs)
            # 可选：输出嵌入形状/样本，便于确认数值
            if bool(cfg.get("pipeline", "log_embedding_stats", default=False)):
                try:
                    import numpy as np
                    arr = np.asarray(gte_vecs)
                    sample = arr[0][:6].tolist() if arr.size > 0 else []
                    print(f"[emb] gte batch shape={arr.shape} dtype=float32 sample6={sample}")
                except Exception:
                    pass
        if bge_enabled and bge_emb is not None:
            if use_ids_for_bge and buf_ids and hasattr(bge_emb, "embed_from_ids"):
                bge_vecs = bge_emb.embed_from_ids(buf_ids)
            else:
                bge_vecs = bge_emb.embed(buf_texts)
            bge_index.build(bge_vecs)
        if bge_small_enabled and bge_small_emb is not None:
            if use_ids_for_bge_small and buf_ids and hasattr(bge_small_emb, "embed_from_ids"):
                bge_s_vecs = bge_small_emb.embed_from_ids(buf_ids)
            else:
                bge_s_vecs = bge_small_emb.embed(buf_texts)
            bge_small_index.build(bge_s_vecs)
        if qwen3_enabled and qwen3_emb is not None:
            if use_ids_for_qwen3 and buf_ids and hasattr(qwen3_emb, "embed_from_ids"):
                qwen_vecs = qwen3_emb.embed_from_ids(buf_ids)
            else:
                qwen_vecs = qwen3_emb.embed(buf_texts)
            qwen3_index.build(qwen_vecs)
        # 流式写出 JSONL：每个 chunk 一行，包含 text 与 meta
        if out_jsonl is not None:
            for m, t in zip(buf_meta, buf_texts):
                out_jsonl.write(json.dumps({"text": t, **m}, ensure_ascii=False) + "\n")
            try:
                out_jsonl.flush()
            except Exception:
                pass
        # 进度打印
        flush_count += 1
        now = time.time()
        elapsed = now - start_ts
        # 估算本次文本字节数
        # 对于 token-id 路径，用 token 数量近似字节数以保持进度估计
        batch_bytes = sum(len(t.encode("utf-8")) for t in buf_texts if t) + sum(len(ids) for ids in buf_ids)
        bytes_total += batch_bytes
        rate = chunk_id / max(elapsed, 1e-6)
        mb_total = bytes_total / (1024 * 1024)
        if now - last_print_ts >= progress_interval:
            msg = f"[progress] flush={flush_count} chunks={chunk_id} bytes~{mb_total:.2f}MB speed={rate:.1f} chunks/s"
            print_and_overwrite(msg)
            last_print_ts = now

        # 周期性保存索引与检查点
        should_save = False
        if periodic_chunks > 0 and (chunk_id - last_save_chunks) >= periodic_chunks:
            should_save = True
        if periodic_sec > 0 and (now - last_save_ts) >= periodic_sec:
            should_save = True
        if should_save:
            try:
                if jina_index is not None:
                    jina_index.save()
                if e5_index is not None:
                    e5_index.save()
                if gte_index is not None:
                    gte_index.save()
                if bge_index is not None:
                    bge_index.save()
                if bge_small_index is not None:
                    bge_small_index.save()
                if qwen3_index is not None:
                    qwen3_index.save()
            except Exception as e:
                print(f"[periodic] save error: {e}")
            if resume_enabled and checkpoint_path:
                try:
                    cp = {
                        "timestamp": now,
                        "chunk_id": chunk_id,
                        "flush_count": flush_count,
                    }
                    csv_positions = {}
                    for m in buf_meta:
                        doc_id = m.get("doc_id", "")
                        if ":row:" in doc_id:
                            base, row_str = doc_id.split(":row:")
                            try:
                                row = int(row_str)
                                csv_positions[base] = max(csv_positions.get(base, -1), row + 1)
                            except Exception:
                                pass
                    if csv_positions:
                        cp["csv_positions"] = csv_positions
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(cp, f, ensure_ascii=False)
                    print(f"\n[checkpoint] saved at chunks={chunk_id} -> {checkpoint_path}")
                except Exception as e:
                    print(f"[checkpoint] write error: {e}")
            last_save_ts = now
            last_save_chunks = chunk_id
        buf_texts = []
        buf_ids = []
        buf_meta = []

    # 如果非直通路径或其他嵌入仍需文本，则需要在直通路径下是否解码文本
    need_texts = (
        jina_enabled or e5_enabled or
        (gte_enabled and not use_ids_for_gte) or
        (bge_enabled and not use_ids_for_bge) or
        (bge_small_enabled and not use_ids_for_bge_small)
    )

    for doc_id, text in docs_iter:
        if use_token_ids:
            doc_chunks = tokenize_chunks_ids(text, chunk_tokenizer, chunk_size, overlap)
            for ch in doc_chunks:
                ids = ch["ids"]
                if not ids:
                    continue
                # 跳过已处理的 chunk（doc_id + position）
                if skip_keys and (doc_id, ch.get("position", 0)) in skip_keys:
                    continue
                buf_ids.append(ids)
                if need_texts:
                    # 仅在其他模型需要文本时解码一次，避免重复分词成本
                    t_dec = chunk_tokenizer.decode(ids, skip_special_tokens=True)
                    buf_texts.append(t_dec)
                else:
                    buf_texts.append("")
                buf_meta.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "position": ch["position"],
                })
                chunk_id += 1
                if len(buf_ids) >= batch_size:
                    flush()
        else:
            doc_chunks = tokenize_chunks(text, chunk_tokenizer, chunk_size, overlap)
            for ch in doc_chunks:
                t = ch["text"].strip()
                if not t:
                    continue
                # 跳过已处理的 chunk（doc_id + position）
                if skip_keys and (doc_id, ch.get("position", 0)) in skip_keys:
                    continue
                buf_texts.append(t)
                buf_meta.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "position": ch["position"],
                })
                chunk_id += 1
                if len(buf_texts) >= batch_size:
                    flush()
    flush()

    # 保存索引前打印一次最终进度并换行
    final_elapsed = time.time() - start_ts
    final_rate = chunk_id / max(final_elapsed, 1e-6)
    final_msg = f"[progress] flush={flush_count} chunks={chunk_id} bytes~{bytes_total/(1024*1024):.2f}MB speed={final_rate:.1f} chunks/s"
    print_and_overwrite(final_msg)
    print("")

    # 保存索引
    if jina_index is not None:
        jina_index.save()
    if e5_index is not None:
        e5_index.save()
    if gte_index is not None:
        gte_index.save()
    if bge_index is not None:
        bge_index.save()
    if bge_small_index is not None:
        bge_small_index.save()
    if qwen3_index is not None:
        qwen3_index.save()
    if out_jsonl is not None:
        out_jsonl.close()

    # 最终检查点（可选）
    if resume_enabled and checkpoint_path:
        try:
            cp = {
                "timestamp": time.time(),
                "chunk_id": chunk_id,
                "flush_count": flush_count,
                "completed": True,
            }
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(cp, f, ensure_ascii=False)
        except Exception as e:
            print(f"[checkpoint] final write error: {e}")

    return {"chunk_count": chunk_id}


def main():
    cfg = Config()
    corpus_dir = cfg.data["corpus_dir"]
    chunk_size = cfg.data["chunking"]["chunk_size_tokens"]
    overlap = cfg.data["chunking"]["chunk_overlap_tokens"]

    hf_token = os.environ.get("huggingface_token", None)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.qwen["model_id"],
        trust_remote_code=cfg.qwen.get("trust_remote_code", True),
        use_auth_token=hf_token,
    )

    docs_text = load_text_files(corpus_dir)
    print(f"Loaded text docs: {len(docs_text)} from {corpus_dir}")

    csv_files = find_csv_files(corpus_dir)
    print(f"Found CSV files: {len(csv_files)} from {corpus_dir}")

    # 断点续跑：加载 JSONL，构建已处理 chunk 键集合（doc_id, position）
    skip_keys = set()
    resume_enabled = bool(cfg.get("pipeline", "resume", "enabled", default=False))
    # 优先使用 bge_small 的 JSONL 路径，其次回退到 Jina 的 JSONL 路径
    meta_jsonl_path_pref = cfg.get("indices", "bge_small", "meta_jsonl_path", default=None) or cfg.get("indices", "jina", "meta_jsonl_path", default=None)
    if resume_enabled and meta_jsonl_path_pref and os.path.exists(meta_jsonl_path_pref):
        try:
            with open(meta_jsonl_path_pref, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        did = obj.get("doc_id")
                        pos = obj.get("position")
                        if did is not None and pos is not None:
                            skip_keys.add((did, int(pos)))
                    except Exception:
                        continue
            print(f"[resume] Loaded processed chunks from JSONL: {len(skip_keys)} keys")
        except Exception as e:
            print(f"[resume] skip_keys load error: {e}")

    def docs_iter() -> Iterable[Tuple[str, str]]:
        for doc_id, t in docs_text:
            yield (doc_id, t)
        for csv_fp in csv_files:
            print(f"正在分块CSV：{os.path.relpath(csv_fp)}")
            # 加强健壮性：逐行读取，避免单条异常中断整体
            try:
                for doc in iter_csv_texts(csv_fp, start_row=0):
                    yield doc
            except Exception as e:
                print(f"[warn] CSV迭代异常 {csv_fp}: {e}")

    result = build_indices_streaming(
        cfg=cfg,
        tokenizer=tokenizer,
        docs_iter=docs_iter(),
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=cfg.get("pipeline", "batch_size", default=128),
        meta_jsonl_path=meta_jsonl_path_pref,
        skip_keys=skip_keys,
    )
    # 改为 JSONL 流式写出，不再汇总到内存
    meta_jsonl_path = meta_jsonl_path_pref
    if meta_jsonl_path:
        print(f"Chunks: {result['chunk_count']}; Meta JSONL saved to: {meta_jsonl_path}")
    else:
        print(f"Chunks: {result['chunk_count']}; (no meta JSONL path configured)")
    if cfg.get("indices", "jina", "enabled", default=False):
        print(f"Jina index saved to: {cfg.get('indices','jina','index_path')}")
    if cfg.get("indices", "bge", "enabled", default=False):
        print(f"BGE index saved to: {cfg.get('indices','bge','index_path')}")
    if cfg.get("indices", "e5", "enabled", default=False):
        print(f"E5 index saved to: {cfg.get('indices','e5','index_path')}")
    if cfg.get("indices", "gte", "enabled", default=False):
        print(f"GTE index saved to: {cfg.get('indices','gte','index_path')}")
    if cfg.get("indices", "qwen3", "enabled", default=False):
        print(f"Qwen3 index saved to: {cfg.get('indices','qwen3','index_path')}")


if __name__ == "__main__":
    main()
