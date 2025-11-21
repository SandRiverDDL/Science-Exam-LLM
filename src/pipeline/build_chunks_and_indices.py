import os
import glob
import json
import csv
import time
from typing import List, Tuple, Iterable, Dict, Any
import sys

from core.config import Config
from chunking.token_chunker import chunk_docs_with_tokenizer, tokenize_chunks, tokenize_chunks_ids
from transformers import AutoTokenizer
from retrieval.embedding_builder import EmbeddingManager
from index.index_manager import IndexManager


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
    # 初始化嵌入模型管理器
    emb_mgr = EmbeddingManager(cfg)
    
    # 获取用于分块的 tokenizer
    chunk_tokenizer, use_token_ids_cfg, embed_tokens_model = emb_mgr.get_chunking_tokenizer(tokenizer)
    
    # 初始化索引管理器
    idx_mgr = IndexManager(cfg, emb_mgr.enabled)
    
    # 断点续跑：加载已有索引
    resume_enabled = bool(cfg.get("pipeline", "resume", "enabled", default=False))
    if resume_enabled:
        idx_mgr.load_existing()

    # 超大数据集使用 JSONL 流式写出，避免内存峰值
    chunk_id = 0
    chunk_id_offset = 0  # 断点续跑时的起始偏移
    buf_texts: List[str] = []
    buf_ids: List[List[int]] = []
    buf_meta: List[Dict[str, Any]] = []
    out_jsonl = None
    
    # 断点续跑：从 checkpoint 恢复 chunk_id 和统计信息
    checkpoint_path = cfg.get("pipeline", "resume", "checkpoint_path", default=None)
    if resume_enabled and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
                chunk_id_offset = cp.get("chunk_id", 0)
                chunk_id = chunk_id_offset
                print(f"[resume] 从检查点恢复: chunk_id={chunk_id_offset}")
        except Exception as e:
            print(f"[resume] 无法加载检查点: {e}")
    
    if meta_jsonl_path:
        os.makedirs(os.path.dirname(meta_jsonl_path), exist_ok=True)
        # 断点续跑时使用追加模式
        mode = "a" if (resume_enabled and os.path.exists(meta_jsonl_path)) else "w"
        out_jsonl = open(meta_jsonl_path, mode, encoding="utf-8")

    # 进度统计
    start_ts = time.time()
    bytes_total = 0
    flush_count = 0
    last_print_ts = start_ts
    max_line_len = 0

    progress_interval = float(cfg.get("pipeline", "progress_interval_sec", default=5))
    periodic_sec = int(cfg.get("pipeline", "periodic_save", "sec", default=0) or 0)
    periodic_chunks = int(cfg.get("pipeline", "periodic_save", "chunks", default=0) or 0)
    last_save_ts = start_ts
    last_save_chunks = chunk_id_offset
    use_token_ids = use_token_ids_cfg
    use_ids_for_model = {
        "gte": use_token_ids and (embed_tokens_model == "gte"),
        "bge": use_token_ids and (embed_tokens_model == "bge"),
        "bge_small": use_token_ids and (embed_tokens_model == "bge_small"),
        "qwen3": use_token_ids and (embed_tokens_model == "qwen3"),
    }
    
    # 性能监控
    emb_time_total = 0.0
    io_time_total = 0.0
    tokenize_time_total = 0.0

    def print_and_overwrite(text: str):
        nonlocal max_line_len
        # 覆盖上一行输出，控制台更清晰
        max_line_len = max(max_line_len, len(text))
        pad = " " * (max_line_len - len(text))
        sys.stdout.write("\r" + text + pad)
        sys.stdout.flush()

    def flush():
        nonlocal buf_texts, buf_ids, buf_meta, flush_count, bytes_total, last_print_ts, last_save_ts, last_save_chunks
        nonlocal emb_time_total, io_time_total
        if not buf_texts and not buf_ids:
            return
        
        # 嵌入并写入索引（带性能监控）
        emb_start = time.time()
        
        for model_name in ["jina", "e5", "gte", "bge", "bge_small", "qwen3"]:
            if not emb_mgr.is_enabled(model_name):
                continue
            
            model = emb_mgr.get_model(model_name)
            index = idx_mgr.get_index(model_name)
            
            if model is None or index is None:
                continue
            
            # 选择嵌入方式：token ids 或文本
            if use_ids_for_model.get(model_name, False) and buf_ids and hasattr(model, "embed_from_ids"):
                vecs = model.embed_from_ids(buf_ids)
            else:
                vecs = model.embed(buf_texts)
            
            index.build(vecs)
        
        emb_time_total += (time.time() - emb_start)
        # 流式写出 JSONL（批量写入优化，减少 flush 频率）
        io_start = time.time()
        if out_jsonl is not None:
            # 批量构建 JSON 字符串，减少逐行写入开销
            lines = []
            for m, t in zip(buf_meta, buf_texts):
                lines.append(json.dumps({"text": t, **m}, ensure_ascii=False))
            out_jsonl.write("\n".join(lines) + "\n")
            # 每 10 次 flush 才真正刷盘一次，减少 I/O
            if flush_count % 10 == 0:
                try:
                    out_jsonl.flush()
                except Exception:
                    pass
        io_time_total += (time.time() - io_start)
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
            # 计算性能指标
            emb_pct = (emb_time_total / max(elapsed, 1e-6)) * 100
            io_pct = (io_time_total / max(elapsed, 1e-6)) * 100
            gpu_util = f"emb={emb_pct:.0f}% io={io_pct:.0f}%"
            # 显示本次运行的 chunks 和累计总 chunks
            chunks_current = chunk_id - chunk_id_offset
            msg = f"[progress] flush={flush_count} chunks={chunks_current}(+{chunk_id_offset}={chunk_id}) bytes~{mb_total:.2f}MB speed={rate:.1f} chunks/s {gpu_util}"
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
                idx_mgr.save_all()
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

    # 判断是否需要解码文本（仅当某些模型不支持 token ids 时）
    need_texts = (
        emb_mgr.is_enabled("jina") or 
        emb_mgr.is_enabled("e5") or
        (emb_mgr.is_enabled("gte") and not use_ids_for_model.get("gte", False)) or
        (emb_mgr.is_enabled("bge") and not use_ids_for_model.get("bge", False)) or
        (emb_mgr.is_enabled("bge_small") and not use_ids_for_model.get("bge_small", False)) or
        (emb_mgr.is_enabled("qwen3") and not use_ids_for_model.get("qwen3", False)) or
        (out_jsonl is not None)  # JSONL 需要文本
    )

    for doc_id, text in docs_iter:
        tok_start = time.time()
        if use_token_ids:
            doc_chunks = tokenize_chunks_ids(text, chunk_tokenizer, chunk_size, overlap)
        else:
            doc_chunks = tokenize_chunks(text, chunk_tokenizer, chunk_size, overlap)
        tokenize_time_total += (time.time() - tok_start)
        
        for ch in doc_chunks:
            if use_token_ids:
                ids = ch.get("ids", [])
                if not ids:
                    continue
                # 跳过已处理的 chunk
                if skip_keys and (doc_id, ch.get("position", 0)) in skip_keys:
                    continue
                buf_ids.append(ids)
                # 关键优化：仅在真正需要时解码文本
                if need_texts:
                    t_dec = chunk_tokenizer.decode(ids, skip_special_tokens=True)
                    buf_texts.append(t_dec)
                else:
                    buf_texts.append("")  # 占位符
            else:
                t = ch.get("text", "").strip()
                if not t:
                    continue
                if skip_keys and (doc_id, ch.get("position", 0)) in skip_keys:
                    continue
                buf_texts.append(t)
            
            buf_meta.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "position": ch.get("position", 0),
            })
            chunk_id += 1
            
            # 使用统一的批次判断
            if (use_token_ids and len(buf_ids) >= batch_size) or (not use_token_ids and len(buf_texts) >= batch_size):
                flush()
    flush()

    # 保存索引前打印一次最终进度并换行
    final_elapsed = time.time() - start_ts
    final_rate = chunk_id / max(final_elapsed, 1e-6)
    emb_pct = (emb_time_total / max(final_elapsed, 1e-6)) * 100
    io_pct = (io_time_total / max(final_elapsed, 1e-6)) * 100
    tok_pct = (tokenize_time_total / max(final_elapsed, 1e-6)) * 100
    chunks_current = chunk_id - chunk_id_offset
    final_msg = f"[progress] flush={flush_count} chunks={chunks_current}(+{chunk_id_offset}={chunk_id}) bytes~{bytes_total/(1024*1024):.2f}MB speed={final_rate:.1f} chunks/s"
    print_and_overwrite(final_msg)
    print("")
    print(f"[perf] 总耗时={final_elapsed:.1f}s | 嵌入={emb_pct:.1f}% | IO={io_pct:.1f}% | 分词={tok_pct:.1f}%")

    # 保存所有索引
    idx_mgr.save_all()
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
    skip_duplicate_check = bool(cfg.get("pipeline", "resume", "skip_duplicate_check", default=False))
    # 优先使用 bge_small 的 JSONL 路径，其次回退到 Jina 的 JSONL 路径
    meta_jsonl_path_pref = cfg.get("indices", "bge_small", "meta_jsonl_path", default=None) or cfg.get("indices", "jina", "meta_jsonl_path", default=None)
    
    # 优化：如果启用 skip_duplicate_check，完全信任 checkpoint，不加载 skip_keys
    load_skip_keys = resume_enabled and not skip_duplicate_check and meta_jsonl_path_pref and os.path.exists(meta_jsonl_path_pref)
    
    if skip_duplicate_check and resume_enabled:
        print("[resume] 已启用 skip_duplicate_check，完全信任 checkpoint，不加载 skip_keys（性能优化）")
    
    # 如果 checkpoint 存在且完整，可以跳过 skip_keys 加载（性能优化）
    checkpoint_path_tmp = cfg.get("pipeline", "resume", "checkpoint_path", default=None)
    if load_skip_keys and checkpoint_path_tmp and os.path.exists(checkpoint_path_tmp):
        try:
            with open(checkpoint_path_tmp, "r", encoding="utf-8") as f:
                cp = json.load(f)
                # 如果 checkpoint 标记为已完成，则无需加载 skip_keys
                if cp.get("completed", False):
                    load_skip_keys = False
                    print("[resume] Checkpoint 标记已完成，跳过 skip_keys 加载")
        except Exception:
            pass
    
    if load_skip_keys:
        try:
            print(f"[resume] 正在加载 skip_keys...")
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
            print(f"[resume] Loaded processed chunks from JSONL: {len(skip_keys):,} keys")
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
