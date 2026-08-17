import os
import re
import json
import sqlite3
import random
from datetime import datetime
from typing import List, Literal
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tavily import TavilyClient

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("⚠️  rapidfuzz not installed — falling back to substring dedup. Run: pip install rapidfuzz")

# -------------------------------------------------------------------
# 1. Helper Functions for String Normalization & Fuzzy Deduplication
# -------------------------------------------------------------------
def clean_string(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

def normalize_url(url: str) -> str:
    parsed = urlparse(url.lower().strip())
    netloc = parsed.netloc.replace("www.", "")
    return f"{parsed.scheme}://{netloc}{parsed.path}".rstrip("/")

def get_domain(normalized_url: str) -> str:
    if "://" in normalized_url:
        return normalized_url.split("://", 1)[1].split("/", 1)[0]
    return normalized_url.split("/", 1)[0]

LOCALE_SEGMENTS = {
    "en", "de", "en-us", "en-gb", "de-de", "en-en",
    "index", "index.html", "index.php", "home"
}

def get_path_stem(normalized_url: str) -> str:
    """Extract the identity-bearing path, stripping locale/noise segments
    (e.g. /laisf/en and /laisf/de both reduce to 'entrepreneurship/laisf')."""
    if "://" in normalized_url:
        parts = normalized_url.split("://", 1)[1].split("/", 1)
        path = parts[1] if len(parts) > 1 else ""
    else:
        parts = normalized_url.split("/", 1)
        path = parts[1] if len(parts) > 1 else ""

    segments = [s for s in path.split("/") if s]
    while segments and segments[-1].lower() in LOCALE_SEGMENTS:
        segments.pop()
    return "/".join(segments).lower()

def fuzzy_score(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if HAS_RAPIDFUZZ:
        return fuzz.token_set_ratio(a, b)
    return 100 if (a in b or b in a) else 0

def fuzzy_match(a: str, b: str, threshold: int = 80) -> bool:
    return fuzzy_score(a, b) >= threshold

# -------------------------------------------------------------------
# 2. Pydantic Schema (Taxonomy) — includes "loan" tag options
# -------------------------------------------------------------------
class ProgramItem(BaseModel):
    name: str = Field(description="Official name of the program, grant, accelerator, or entity")
    provider: str = Field(description="Managing organization, university, state agency, or fund")
    url: str = Field(description="Direct official link to the program or application page")
    description: str = Field(description="Brief single-sentence overview of the offer")

    type_tag: Literal["grant", "accelerator", "incubator", "vc", "loan"] = Field(
        description="Program Structure: grant, accelerator, incubator, vc, or loan"
    )
    funding_tag: Literal["equity-free", "zero-equity", "equity", "subsidy", "loan"] = Field(
        description="Funding Mechanics: equity-free, zero-equity, equity, subsidy, or loan"
    )
    stage_tag: Literal["pre-seed", "seed", "scale"] = Field(
        description="Startup Stage: pre-seed, seed, or scale"
    )
    eligibility_tag: Literal["de-academic", "de-entity", "international"] = Field(
        description="Eligibility: de-academic, de-entity, or international"
    )
    backing_tag: Literal["public", "ppp", "private"] = Field(
        description="Backing Capital: public, ppp, or private"
    )
    bmwk_tag: Literal["direct", "mandated", "none"] = Field(
        description="BMWK Affiliation: direct, mandated, or none"
    )
    exist_tag: Literal["grant", "partner", "none"] = Field(
        description="EXIST Ecosystem: grant, partner, or none"
    )
    focus_tags: List[Literal["software", "electronics", "ai", "deeptech"]] = Field(
        description="Technology Focus array subset of software, electronics, ai, deeptech"
    )

class ProgramList(BaseModel):
    programs: List[ProgramItem]

class QueryMutation(BaseModel):
    mutated_query: str = Field(description="A more specific, sub-program-level version of the base query")

class MutationBatch(BaseModel):
    mutations: List[QueryMutation]

# -------------------------------------------------------------------
# 3. Seed Taxonomy — cross-product instead of hardcoded strings
# -------------------------------------------------------------------
BUNDESLAENDER = [
    "Baden-Württemberg", "Bavaria", "Berlin", "Brandenburg", "Bremen",
    "Hamburg", "Hesse", "Mecklenburg-Vorpommern", "Lower Saxony",
    "North Rhine-Westphalia", "Rhineland-Palatinate", "Saarland",
    "Saxony", "Saxony-Anhalt", "Schleswig-Holstein", "Thuringia"
]

PROGRAM_TYPES = ["grant", "accelerator", "incubator", "loan"]

SECTOR_MODIFIERS = ["deep tech", "AI", "cleantech", "life sciences", "hardware startup"]

def build_seed_queries():
    seeds = []
    for state in BUNDESLAENDER:
        for ptype in PROGRAM_TYPES:
            seeds.append((f"state:{state}", f"{state} startup {ptype} program Germany"))
    for sector in SECTOR_MODIFIERS:
        seeds.append((f"sector:{sector}", f"Germany {sector} startup funding program"))
    seeds.append(("national:EXIST", "EXIST Gründungsstipendium Forschungstransfer 2026"))
    seeds.append(("national:BMWK", "BMWK German federal startup grant program"))
    return seeds

# -------------------------------------------------------------------
# 4. Composite SQLite Database Engine
# -------------------------------------------------------------------
class ProgramDatabase:
    def __init__(self, db_path="german_startup_programs.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("PRAGMA table_info(programs)")
            columns = [info[1] for info in cursor.fetchall()]
            if columns and "clean_name" not in columns:
                print("🔄 Updating SQLite schema (programs)...")
                cursor.execute("DROP TABLE programs")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS programs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    clean_name TEXT NOT NULL,
                    provider TEXT,
                    clean_provider TEXT,
                    url TEXT,
                    normalized_url TEXT,
                    path_stem TEXT,
                    description TEXT,
                    type_tag TEXT,
                    funding_tag TEXT,
                    stage_tag TEXT,
                    eligibility_tag TEXT,
                    backing_tag TEXT,
                    bmwk_tag TEXT,
                    exist_tag TEXT,
                    focus_tags TEXT,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_pool (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT UNIQUE,
                    category TEXT,
                    hit_count INTEGER DEFAULT 0,
                    threshold INTEGER DEFAULT 3,
                    novelty_last_run INTEGER DEFAULT -1,
                    consecutive_zero_novelty INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    last_searched_at TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

    def seed_pool_if_empty(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM query_pool")
            count = cursor.fetchone()[0]
            if count > 0:
                return
            seeds = build_seed_queries()
            for category, query_text in seeds:
                threshold = 2 if category.startswith("national:") else 3
                try:
                    conn.execute(
                        "INSERT INTO query_pool (query_text, category, threshold) VALUES (?, ?, ?)",
                        (query_text, category, threshold)
                    )
                except sqlite3.IntegrityError:
                    pass
            print(f"🌱 Seeded query_pool with {len(seeds)} queries.")

    # -----------------------------------------------------------
    # Fingerprint-based dedup
    # -----------------------------------------------------------
    def is_duplicate(self, prog: ProgramItem) -> bool:
        norm_url = normalize_url(prog.url)
        norm_domain = get_domain(norm_url)
        norm_stem = get_path_stem(norm_url)
        norm_name = clean_string(prog.name)
        norm_provider = clean_string(prog.provider)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT clean_name, clean_provider, normalized_url, path_stem,
                       type_tag, funding_tag, stage_tag, bmwk_tag, exist_tag
                FROM programs
            """)
            all_rows = cursor.fetchall()

        for (existing_name, existing_provider, existing_url, existing_stem,
             e_type, e_funding, e_stage, e_bmwk, e_exist) in all_rows:

            existing_domain = get_domain(existing_url) if existing_url else ""
            same_domain = bool(norm_domain) and norm_domain == existing_domain
            same_stem = same_domain and norm_stem == (existing_stem or "") and norm_stem != ""

            name_score = fuzzy_score(norm_name, existing_name)
            provider_score = fuzzy_score(norm_provider, existing_provider)

            # Field agreement: how many taxonomy tags match exactly
            tag_pairs = [
                (prog.type_tag, e_type), (prog.funding_tag, e_funding),
                (prog.stage_tag, e_stage), (prog.bmwk_tag, e_bmwk),
                (prog.exist_tag, e_exist),
            ]
            tag_matches = sum(1 for a, b in tag_pairs if a == b)

            # --- Tier 1: same domain + same path stem = structurally the
            # same page family (e.g. /laisf/en vs /laisf/de). Name is just
            # a sanity floor here, not the primary signal.
            if same_stem and name_score >= 50:
                return True

            # --- Tier 2: field-fingerprint match — name is reasonably close
            # AND provider matches AND at least 4/5 taxonomy tags agree.
            # This is the language/domain-independent check: catches
            # exist.de vs exist.com, or a program mirrored on an aggregator
            # site, purely from extracted facts rather than URL shape.
            if name_score >= 70 and provider_score >= 70 and tag_matches >= 4:
                return True

            # --- Tier 3: same domain, DIFFERENT path stem = siblings from
            # the same org (e.g. laisf vs ai-academy vs incubator-ignition).
            # Domain gives no bonus here — require a high name bar so
            # distinct sibling programs don't get merged.
            if same_domain and not same_stem and name_score >= 88:
                return True

            # --- Tier 4: different domain entirely, name similarity only.
            if not same_domain and name_score >= 80:
                return True

        return False

    def insert_program(self, prog: ProgramItem) -> bool:
        if self.is_duplicate(prog):
            return False

        norm_url = normalize_url(prog.url)
        norm_name = clean_string(prog.name)
        norm_provider = clean_string(prog.provider)
        stem = get_path_stem(norm_url)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO programs (
                    name, clean_name, provider, clean_provider, url, normalized_url,
                    path_stem, description, type_tag, funding_tag, stage_tag,
                    eligibility_tag, backing_tag, bmwk_tag, exist_tag, focus_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prog.name, norm_name, prog.provider, norm_provider, prog.url, norm_url,
                stem, prog.description, prog.type_tag, prog.funding_tag, prog.stage_tag,
                prog.eligibility_tag, prog.backing_tag, prog.bmwk_tag, prog.exist_tag,
                ",".join(prog.focus_tags)
            ))
        return True

    # ---- Query pool mechanics ----
    def get_next_batch(self, n=5):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, query_text, category, hit_count, threshold
                FROM query_pool
                WHERE status = 'active'
                ORDER BY hit_count ASC, RANDOM()
                LIMIT ?
            """, (n,))
            return cursor.fetchall()

    def update_pool_stats(self, query_id, added_count):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT hit_count, threshold, consecutive_zero_novelty FROM query_pool WHERE id = ?", (query_id,))
            hit_count, threshold, zero_streak = cursor.fetchone()

            hit_count += 1
            zero_streak = zero_streak + 1 if added_count == 0 else 0

            status = 'active'
            if hit_count >= threshold or zero_streak >= 2:
                status = 'exhausted'

            conn.execute("""
                UPDATE query_pool
                SET hit_count = ?, novelty_last_run = ?, consecutive_zero_novelty = ?,
                    status = ?, last_searched_at = ?
                WHERE id = ?
            """, (hit_count, added_count, zero_streak, status, datetime.utcnow().isoformat(), query_id))

    def count_active(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM query_pool WHERE status = 'active'")
            return cursor.fetchone()[0]

    def get_all_exhausted(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, query_text, category FROM query_pool WHERE status = 'exhausted'")
            return cursor.fetchall()

    def insert_mutated_query(self, query_text, category, threshold=2):
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO query_pool (query_text, category, threshold) VALUES (?, ?, ?)",
                    (query_text, category, threshold)
                )
                return True
            except sqlite3.IntegrityError:
                return False

    # ---- Cleanup ----
    def deduplicate_database(self):
        """Exact clean_name matches only — fast pass."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM programs
                WHERE id NOT IN (SELECT MIN(id) FROM programs GROUP BY clean_name)
            """)
            deleted = cursor.rowcount
            print("\n" + "=" * 80)
            print("🧹 AUTOMATED HOUSEKEEPING REPORT (exact match)")
            print("=" * 80)
            print(f"  • Purged {deleted} stray duplicate record(s)." if deleted else "  • Database clean.")

    def fuzzy_deduplicate_database(self, name_threshold=80, tag_match_min=4):
        """Full pairwise sweep using the same fingerprint logic (name +
        provider + tag agreement + path stem), applied retroactively across
        the whole table. Keeps the earliest-discovered row of each cluster."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, clean_name, clean_provider, normalized_url, path_stem,
                       type_tag, funding_tag, stage_tag, bmwk_tag, exist_tag
                FROM programs ORDER BY discovered_at ASC
            """)
            rows = cursor.fetchall()

        to_delete = set()
        n = len(rows)
        for i in range(n):
            (id_a, name_a, prov_a, url_a, stem_a,
             type_a, fund_a, stage_a, bmwk_a, exist_a) = rows[i]
            if id_a in to_delete:
                continue
            for j in range(i + 1, n):
                (id_b, name_b, prov_b, url_b, stem_b,
                 type_b, fund_b, stage_b, bmwk_b, exist_b) = rows[j]
                if id_b in to_delete:
                    continue

                domain_a, domain_b = get_domain(url_a or ""), get_domain(url_b or "")
                same_domain = bool(domain_a) and domain_a == domain_b
                same_stem = same_domain and (stem_a or "") == (stem_b or "") and stem_a

                name_score = fuzzy_score(name_a, name_b)
                provider_score = fuzzy_score(prov_a, prov_b)
                tag_matches = sum(1 for x, y in [
                    (type_a, type_b), (fund_a, fund_b), (stage_a, stage_b),
                    (bmwk_a, bmwk_b), (exist_a, exist_b)
                ] if x == y)

                is_dupe = False
                if same_stem and name_score >= 50:
                    is_dupe = True
                elif name_score >= 70 and provider_score >= 70 and tag_matches >= tag_match_min:
                    is_dupe = True
                elif same_domain and not same_stem and name_score >= 88:
                    is_dupe = True
                elif not same_domain and name_score >= name_threshold:
                    is_dupe = True

                if is_dupe:
                    to_delete.add(id_b)

        if to_delete:
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ",".join("?" * len(to_delete))
                conn.execute(f"DELETE FROM programs WHERE id IN ({placeholders})", tuple(to_delete))
            print(f"🧹 Fuzzy sweep: removed {len(to_delete)} near-duplicate program(s).")
        else:
            print("🧹 Fuzzy sweep: no near-duplicates found.")
        return len(to_delete)

    def maybe_run_fuzzy_sweep(self, interval_days=0, force=False):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM meta WHERE key = 'last_fuzzy_sweep'")
            row = cursor.fetchone()

        if not force and interval_days > 0 and row:
            last_run = datetime.fromisoformat(row[0])
            if (datetime.utcnow() - last_run).total_seconds() < interval_days * 86400:
                print(f"⏭️  Fuzzy sweep skipped (last ran {row[0]}, interval={interval_days}d).")
                return 0

        removed = self.fuzzy_deduplicate_database()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO meta (key, value) VALUES ('last_fuzzy_sweep', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (datetime.utcnow().isoformat(),))
        return removed

    def list_all_programs(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, provider, type_tag, funding_tag, stage_tag, eligibility_tag,
                       backing_tag, bmwk_tag, exist_tag, focus_tags, url, description
                FROM programs ORDER BY discovered_at DESC
            """)
            return cursor.fetchall()

    def print_terminal_summary(self):
        programs = self.list_all_programs()
        print("\n" + "=" * 80)
        print(f"📊 LIVE DATABASE DIRECTORY ({len(programs)} TOTAL ACTIVE PROGRAMS)")
        print("=" * 80)
        for idx, row in enumerate(programs, 1):
            name, provider, type_tag, funding_tag, stage_tag, eligibility_tag, backing_tag, bmwk_tag, exist_tag, focus, url, desc = row
            tags_str = f"type:{type_tag} | funding:{funding_tag} | stage:{stage_tag} | bmwk:{bmwk_tag} | exist:{exist_tag} | focus:{focus}"
            print(f"\n[{idx}] {name} ({provider})")
            print(f"    Tags: {tags_str}")
            print(f"    URL:  {url}")
            print(f"    Info: {desc}")
        print("\n" + "=" * 80)

    def print_pool_summary(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, COUNT(*) FROM query_pool GROUP BY status")
            rows = cursor.fetchall()
        print("\n📦 QUERY POOL STATUS:", dict(rows))

# -------------------------------------------------------------------
# 5. Discovery Engine
# -------------------------------------------------------------------
class GermanEcosystemAgent:
    def __init__(self):
        self.db = ProgramDatabase()
        self.db.seed_pool_if_empty()
        self.ai = genai.Client()  # Uses GEMINI_API_KEY
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def search_and_categorize(self, search_query: str, angle_label: str) -> int:
        """Returns number of NEW (non-duplicate) programs added."""
        print(f"\n🔍 Searching [{angle_label}]: '{search_query}'...")

        search_response = self.tavily.search(
            query=search_query,
            search_depth="basic",
            include_answer="advanced",
            max_results=10
        )

        tavily_answer = search_response.get("answer", "")
        web_snippets = "\n\n".join([
            f"Source URL: {res['url']}\nContent: {res['content']}"
            for res in search_response.get("results", [])
        ])

        prompt = f"""
        System Role: You are an expert German Venture Capital & Innovation Ecosystem Analyst.

        Task: Extract active German startup grants, accelerators, incubators, loans, and VC programs
        from the web context below. Categorize each program strictly according to the schema.
        Write a clear, informative one-sentence description for each program suitable for public
        display on a directory website. If the same organization runs multiple distinct programs,
        list each as a separate entry with its own specific URL. If no valid programs are present,
        return an empty list — do not invent entries.

        Tavily AI Executive Summary:
        {tavily_answer}

        Web Snippets:
        {web_snippets}
        """

        response = self.ai.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProgramList,
                temperature=0.0
            )
        )

        extracted_data: ProgramList = response.parsed

        added, duplicates = 0, 0
        for program in extracted_data.programs:
            if self.db.insert_program(program):
                tags_str = f"type:{program.type_tag} | funding:{program.funding_tag} | stage:{program.stage_tag} | bmwk:{program.bmwk_tag} | exist:{program.exist_tag}"
                print(f"  ✅ Appended: {program.name} ({program.provider})")
                print(f"     Tags -> [{tags_str}]")
                added += 1
            else:
                print(f"  ⏭️ Skipped (Duplicate): {program.name}")
                duplicates += 1

        print(f"  --> '{angle_label}': {added} added | {duplicates} duplicates")
        return added

    def trigger_mutation_cycle(self):
        exhausted = self.db.get_all_exhausted()
        if not exhausted:
            print("⚠️  No exhausted queries to mutate — pool is genuinely empty.")
            return

        print(f"\n🧬 MUTATION CYCLE: expanding {len(exhausted)} exhausted queries...")

        BATCH_SIZE = 10
        for i in range(0, len(exhausted), BATCH_SIZE):
            chunk = exhausted[i:i + BATCH_SIZE]
            base_list = "\n".join([f"- ({cat}) {q}" for _, q, cat in chunk])

            prompt = f"""
            You are refining a German startup-funding search query pool.
            Below are BASE queries that have been fully searched and exhausted (returning no new results).
            For EACH base query, write exactly ONE more specific, sub-program-level search query —
            target a specific sub-initiative, target group (women, migrants, students), funding round,
            or 2026 cohort rather than repeating the general topic.
            Return one mutation per base query, in the same order.

            Base queries:
            {base_list}
            """

            response = self.ai.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MutationBatch,
                    temperature=0.7
                )
            )

            mutation_data: MutationBatch = response.parsed
            for (qid, orig_query, category), mutation in zip(chunk, mutation_data.mutations):
                inserted = self.db.insert_mutated_query(
                    mutation.mutated_query, f"{category}:mutated", threshold=2
                )
                if inserted:
                    print(f"  🌿 Mutated ({category}): '{orig_query}' -> '{mutation.mutated_query}'")

    def run_until_novel_target(self, target_new=5, max_batches=15, batch_size=5):
        total_new = 0
        batches_run = 0

        print("\n🚀 STARTING GERMAN STARTUP PROGRAM DISCOVERY ENGINE...")
        print("=" * 80)

        while total_new < target_new and batches_run < max_batches:
            batch = self.db.get_next_batch(n=batch_size)

            if not batch:
                if self.db.count_active() == 0:
                    self.trigger_mutation_cycle()
                    batch = self.db.get_next_batch(n=batch_size)
                if not batch:
                    print("🏁 Pool truly exhausted — no more queries to try.")
                    break

            for query_id, query_text, category, hit_count, threshold in batch:
                added = self.search_and_categorize(query_text, category)
                total_new += added
                self.db.update_pool_stats(query_id, added)

                if total_new >= target_new:
                    break

            batches_run += 1
            print(f"\n— Batch {batches_run} complete. Running total: {total_new}/{target_new} new programs —")

        print("\n" + "=" * 80)
        if total_new >= target_new:
            print(f"✅ TARGET MET: {total_new} new programs added across {batches_run} batch(es).")
        else:
            print(f"⚠️  STOPPED at max_batches ({max_batches}). Only {total_new}/{target_new} new programs found.")
            print("    This may mean the pool is genuinely near-exhausted — check pool status below.")
        print("=" * 80)

        self.db.print_pool_summary()
        return total_new

    def export_to_json(self, json_path="grants.json"):
        programs = self.db.list_all_programs()
        data = []
        for row in programs:
            name, provider, type_tag, funding_tag, stage_tag, eligibility_tag, backing_tag, bmwk_tag, exist_tag, focus, url, desc = row
            data.append({
                "name": name, "provider": provider, "type": type_tag, "funding": funding_tag,
                "stage": stage_tag, "eligibility": eligibility_tag, "backing": backing_tag,
                "bmwk": bmwk_tag, "exist": exist_tag,
                "focus": focus.split(",") if focus else [],
                "url": url, "description": desc
            })
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"🌐 Web Dashboard Export: Written {len(data)} unique record(s) to '{json_path}'.")

# -------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    agent = GermanEcosystemAgent()

    agent.run_until_novel_target(target_new=5, max_batches=15, batch_size=5)

    agent.db.deduplicate_database()               # exact clean_name matches
    agent.db.maybe_run_fuzzy_sweep(interval_days=0)  # 0 = always run for now
    agent.export_to_json()
    agent.db.print_terminal_summary()
