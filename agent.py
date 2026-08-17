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

def fuzzy_match(a: str, b: str, threshold: int = 85) -> bool:
    if HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(a, b) >= threshold
    return a in b or b in a  # fallback

# -------------------------------------------------------------------
# 2. Pydantic Schema (Taxonomy) — added "loan" to type/funding tags
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
# 3. Seed Taxonomy — cross-product instead of 5 hardcoded strings
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
    """State x program-type cross product (primary seed), sector modifiers layered in lightly."""
    seeds = []
    for state in BUNDESLAENDER:
        for ptype in PROGRAM_TYPES:
            seeds.append((f"state:{state}", f"{state} startup {ptype} program Germany"))
    # A handful of sector-first queries, not a full cross product (keeps pool sane at start)
    for sector in SECTOR_MODIFIERS:
        seeds.append((f"sector:{sector}", f"Germany {sector} startup funding program"))
    # National/high-profile programs, seeded once — pool mechanics handle repeats & cooldown
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
                    url TEXT,
                    normalized_url TEXT,
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

    def seed_pool_if_empty(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM query_pool")
            count = cursor.fetchone()[0]
            if count > 0:
                return
            seeds = build_seed_queries()
            for category, query_text in seeds:
                # National/high-profile queries get a lower threshold so they
                # don't dominate search budget; regional gets a slightly higher one
                threshold = 2 if category.startswith("national:") else 3
                try:
                    conn.execute(
                        "INSERT INTO query_pool (query_text, category, threshold) VALUES (?, ?, ?)",
                        (query_text, category, threshold)
                    )
                except sqlite3.IntegrityError:
                    pass
            print(f"🌱 Seeded query_pool with {len(seeds)} queries.")

    # ---- Program dedup / insert ----
    def is_duplicate(self, prog: ProgramItem) -> bool:
        norm_url = normalize_url(prog.url)
        norm_name = clean_string(prog.name)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM programs WHERE normalized_url = ?", (norm_url,))
            if cursor.fetchone():
                return True

            cursor.execute("""
                SELECT clean_name FROM programs
                WHERE type_tag = ? AND stage_tag = ? AND bmwk_tag = ?
            """, (prog.type_tag, prog.stage_tag, prog.bmwk_tag))

            for (existing_clean_name,) in cursor.fetchall():
                if fuzzy_match(norm_name, existing_clean_name):
                    return True
        return False

    def insert_program(self, prog: ProgramItem) -> bool:
        if self.is_duplicate(prog):
            return False

        norm_url = normalize_url(prog.url)
        norm_name = clean_string(prog.name)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO programs (
                    name, clean_name, provider, url, normalized_url, description,
                    type_tag, funding_tag, stage_tag, eligibility_tag,
                    backing_tag, bmwk_tag, exist_tag, focus_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prog.name, norm_name, prog.provider, prog.url, norm_url, prog.description,
                prog.type_tag, prog.funding_tag, prog.stage_tag, prog.eligibility_tag,
                prog.backing_tag, prog.bmwk_tag, prog.exist_tag,
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

            # Exhaust if threshold hit OR two consecutive dead runs (novelty exhaustion)
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
                return False  # already exists, skip

    def deduplicate_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM programs
                WHERE id NOT IN (SELECT MIN(id) FROM programs GROUP BY clean_name)
            """)
            deleted = cursor.rowcount
            print("\n" + "=" * 80)
            print("🧹 AUTOMATED HOUSEKEEPING REPORT")
            print("=" * 80)
            print(f"  • Purged {deleted} stray duplicate record(s)." if deleted else "  • Database clean.")

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
        If no valid programs are present, return an empty list — do not invent entries.

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
        """When the active pool is empty, mutate every exhausted query into a sub-program query."""
        exhausted = self.db.get_all_exhausted()
        if not exhausted:
            print("⚠️  No exhausted queries to mutate — pool is genuinely empty.")
            return

        print(f"\n🧬 MUTATION CYCLE: expanding {len(exhausted)} exhausted queries...")

        # Batch these to Gemini in groups to keep prompts manageable
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
        """
        Keeps running batches (pulling lowest hit_count queries, mutating when
        the pool exhausts) until at least `target_new` unique programs have been
        added, or `max_batches` is hit (safety valve against unlimited API spend).
        """
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

    agent.db.deduplicate_database()
    agent.export_to_json()
    agent.db.print_terminal_summary()
