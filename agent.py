import os
import re
import json
import sqlite3
import random
from datetime import datetime
from typing import List, Literal, Optional
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
# 1. Helper Functions
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
# 2. Pydantic Schema (Taxonomy)
# -------------------------------------------------------------------
class ProgramItem(BaseModel):
    name: str = Field(description="Official name of the program, grant, accelerator, or entity")
    provider: str = Field(description="Managing organization, university, state agency, or fund")
    url: str = Field(description="Direct official link to the program or application page")
    description: str = Field(description="Brief single-sentence overview of the offer")
    deadline: str = Field(
        description=(
            "Application deadline or cycle info, in whatever form the source states it "
            "(e.g. '04.03.2026', 'March 2026', 'rolling', 'ongoing', 'next call Q3 2026'). "
            "If the source gives no deadline information at all, use 'not stated'."
        )
    )

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
# 3. Seed Taxonomy
# -------------------------------------------------------------------
BUNDESLAENDER = [
    "Baden-Württemberg", "Bavaria", "Berlin", "Brandenburg", "Bremen",
    "Hamburg", "Hesse", "Mecklenburg-Vorpommern", "Lower Saxony",
    "North Rhine-Westphalia", "Rhineland-Palatinate", "Saarland",
    "Saxony", "Saxony-Anhalt", "Schleswig-Holstein", "Thuringia"
]

PROGRAM_TYPES = ["grant", "accelerator", "incubator", "loan"]

SECTOR_MODIFIERS = ["deep tech", "AI", "cleantech", "life sciences", "hardware startup"]

NATIONAL_PROGRAMS = [
    ("national:EXIST-Gruendungsstipendium", "EXIST Gründungsstipendium application requirements 2026"),
    ("national:EXIST-Women", "EXIST Women 2026 application Germany"),
    ("national:KfW-StartGeld", "KfW Gründerkredit StartGeld startup loan Germany"),
    ("national:KfW-Universell", "KfW Gründerkredit Universell startup loan Germany"),
    ("national:KfW-Capital", "KfW Capital venture capital fund Germany startup"),
    ("national:Gruendungszuschuss", "Gründungszuschuss Federal Employment Agency startup grant"),
    ("national:INVEST-Wagniskapital", "INVEST Zuschuss für Wagniskapital Germany venture capital grant"),
    ("national:ERP-Startfonds", "ERP-Startfonds KfW venture capital Germany startup"),
    ("national:WIPANO", "WIPANO patent innovation grant Germany startup"),
    ("national:go-digital", "go-digital BMWK digitalization grant Germany SME"),
    ("national:High-Tech-Gruenderfonds", "High-Tech Gründerfonds HTGF seed investment Germany"),
    ("national:Deutschlandfonds", "Deutschlandfonds federal startup investment Germany"),
    ("national:DTCF", "DeepTech and Climate Fonds Germany startup funding"),
    ("national:Mikromezzaninfonds", "Mikromezzaninfonds Deutschland startup financing"),
    ("national:EXIST-Potentiale", "EXIST-Potentiale university startup infrastructure grant Germany"),
    ("national:BAFA-grants", "BAFA federal grant program startup innovation Germany"),
    ("national:Kultur-Kreativpiloten", "Kultur- und Kreativpiloten Deutschland federal grant"),
    ("national:BMWK-Coaching", "Bund Coaching-Programme Gründer Zuschuss Germany"),
]

NATIONAL_BROAD = [
    ("national:broad-1", "list of German federal startup funding programs 2026"),
    ("national:broad-2", "Bundesförderung Startup Zuschuss Übersicht"),
    ("national:broad-3", "German government grants for founders not state specific"),
    ("national:broad-4", "new federal startup funding program Germany 2026"),
    ("national:broad-5", "Förderdatenbank Bund startup grant"),
]

EU_PROGRAMS = [
    ("eu:EIC-Accelerator", "EIC Accelerator European Innovation Council funding Germany"),
    ("eu:Horizon-Europe", "Horizon Europe startup funding Germany"),
    ("eu:EIT-Digital", "EIT Digital accelerator funding Germany startup"),
    ("eu:EIF-VC", "European Investment Fund venture capital Germany startup"),
    ("eu:Eurostars", "Eurostars programme German startups R&D funding"),
    ("eu:EU-Innovation-Fund", "EU Innovation Fund cleantech Germany startup"),
]

def build_seed_queries():
    seeds = []
    for state in BUNDESLAENDER:
        for ptype in PROGRAM_TYPES:
            seeds.append((f"state:{state}", f"{state} startup {ptype} program Germany"))
    for sector in SECTOR_MODIFIERS:
        seeds.append((f"sector:{sector}", f"Germany {sector} startup funding program"))
    seeds.extend(NATIONAL_PROGRAMS)
    seeds.extend(NATIONAL_BROAD)
    seeds.extend(EU_PROGRAMS)
    return seeds

def priority_for_category(category: str) -> int:
    if category.startswith("national:"):
        return 0
    if category.startswith("eu:"):
        return 2
    return 1

def threshold_for_category(category: str) -> int:
    if category.startswith("national:broad"):
        return 4
    if category.startswith("national:") or category.startswith("eu:"):
        return 2
    return 3

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
                    deadline TEXT,
                    deadline_checked_at TIMESTAMP,
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

            # Additive migration: add any columns missing from an older
            # schema version instead of dropping the table (which would
            # destroy already-discovered programs).
            cursor.execute("PRAGMA table_info(programs)")
            existing_columns = {info[1] for info in cursor.fetchall()}

            required_columns = {
                "clean_name": "TEXT", "clean_provider": "TEXT", "path_stem": "TEXT",
                "deadline": "TEXT", "deadline_checked_at": "TIMESTAMP",
            }
            for col_name, col_type in required_columns.items():
                if col_name not in existing_columns:
                    print(f"🔧 Adding missing column '{col_name}' to programs table...")
                    cursor.execute(f"ALTER TABLE programs ADD COLUMN {col_name} {col_type}")

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

    def seed_pool(self):
        seeds = build_seed_queries()
        added = 0
        with sqlite3.connect(self.db_path) as conn:
            for category, query_text in seeds:
                threshold = threshold_for_category(category)
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO query_pool (query_text, category, threshold) VALUES (?, ?, ?)",
                    (query_text, category, threshold)
                )
                if cursor.rowcount > 0:
                    added += 1
        if added:
            print(f"🌱 Seeded {added} new quer{'y' if added == 1 else 'ies'} into the pool (total seed set: {len(seeds)}).")
        else:
            print(f"🌱 Pool already contains all {len(seeds)} seed queries — nothing new to add.")

    # -----------------------------------------------------------
    # Fingerprint-based dedup — now returns the matched row's id (or None)
    # so the caller can refresh deadline info on rediscovery instead of
    # just silently skipping.
    # -----------------------------------------------------------
    def find_duplicate_id(self, prog: ProgramItem) -> Optional[int]:
        norm_url = normalize_url(prog.url)
        norm_domain = get_domain(norm_url)
        norm_stem = get_path_stem(norm_url)
        norm_name = clean_string(prog.name)
        norm_provider = clean_string(prog.provider)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, clean_name, clean_provider, normalized_url, path_stem,
                       type_tag, funding_tag, stage_tag, bmwk_tag, exist_tag
                FROM programs
            """)
            all_rows = cursor.fetchall()

        for (row_id, existing_name, existing_provider, existing_url, existing_stem,
             e_type, e_funding, e_stage, e_bmwk, e_exist) in all_rows:

            existing_domain = get_domain(existing_url) if existing_url else ""
            same_domain = bool(norm_domain) and norm_domain == existing_domain
            same_stem = same_domain and norm_stem == (existing_stem or "") and norm_stem != ""

            name_score = fuzzy_score(norm_name, existing_name)
            provider_score = fuzzy_score(norm_provider, existing_provider)

            tag_pairs = [
                (prog.type_tag, e_type), (prog.funding_tag, e_funding),
                (prog.stage_tag, e_stage), (prog.bmwk_tag, e_bmwk),
                (prog.exist_tag, e_exist),
            ]
            tag_matches = sum(1 for a, b in tag_pairs if a == b)

            if same_stem and name_score >= 50:
                return row_id
            if name_score >= 70 and provider_score >= 70 and tag_matches >= 4:
                return row_id
            if same_domain and not same_stem and name_score >= 88:
                return row_id
            if not same_domain and name_score >= 80:
                return row_id

        return None

    def refresh_deadline(self, row_id: int, deadline: str):
        """Called on rediscovery of an existing program — updates the
        deadline and stamps when it was last checked, so the website can
        judge staleness from deadline_checked_at."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE programs SET deadline = ?, deadline_checked_at = ? WHERE id = ?
            """, (deadline, datetime.utcnow().isoformat(), row_id))

    def insert_program(self, prog: ProgramItem) -> bool:
        """Returns True if a NEW row was inserted. If the program already
        exists, its deadline is refreshed in place instead, and this
        returns False (counts as a duplicate for novelty tracking, but the
        deadline data still gets updated)."""
        existing_id = self.find_duplicate_id(prog)
        if existing_id is not None:
            self.refresh_deadline(existing_id, prog.deadline)
            return False

        norm_url = normalize_url(prog.url)
        norm_name = clean_string(prog.name)
        norm_provider = clean_string(prog.provider)
        stem = get_path_stem(norm_url)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO programs (
                    name, clean_name, provider, clean_provider, url, normalized_url,
                    path_stem, description, deadline, deadline_checked_at,
                    type_tag, funding_tag, stage_tag, eligibility_tag,
                    backing_tag, bmwk_tag, exist_tag, focus_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prog.name, norm_name, prog.provider, norm_provider, prog.url, norm_url,
                stem, prog.description, prog.deadline, datetime.utcnow().isoformat(),
                prog.type_tag, prog.funding_tag, prog.stage_tag, prog.eligibility_tag,
                prog.backing_tag, prog.bmwk_tag, prog.exist_tag,
                ",".join(prog.focus_tags)
            ))
        return True

    # ---- Query pool mechanics ----
    def get_next_batch(self, n=12):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, query_text, category, hit_count, threshold
                FROM query_pool
                WHERE status = 'active'
            """)
            rows = cursor.fetchall()

        rows.sort(key=lambda r: (priority_for_category(r[2]), r[3], random.random()))
        return rows[:n]

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
                       backing_tag, bmwk_tag, exist_tag, focus_tags, url, description,
                       deadline, deadline_checked_at
                FROM programs ORDER BY discovered_at DESC
            """)
            return cursor.fetchall()

    def print_terminal_summary(self):
        programs = self.list_all_programs()
        print("\n" + "=" * 80)
        print(f"📊 LIVE DATABASE DIRECTORY ({len(programs)} TOTAL ACTIVE PROGRAMS)")
        print("=" * 80)
        for idx, row in enumerate(programs, 1):
            (name, provider, type_tag, funding_tag, stage_tag, eligibility_tag, backing_tag,
             bmwk_tag, exist_tag, focus, url, desc, deadline, checked_at) = row
            tags_str = f"type:{type_tag} | funding:{funding_tag} | stage:{stage_tag} | bmwk:{bmwk_tag} | exist:{exist_tag} | focus:{focus}"
            print(f"\n[{idx}] {name} ({provider})")
            print(f"    Tags: {tags_str}")
            print(f"    URL:  {url}")
            print(f"    Deadline: {deadline}  (checked: {checked_at})")
            print(f"    Info: {desc}")
        print("\n" + "=" * 80)

    def print_pool_summary(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, COUNT(*) FROM query_pool GROUP BY status")
            rows = cursor.fetchall()
            cursor.execute("""
                SELECT
                    CASE
                        WHEN category LIKE 'national:%' THEN 'national'
                        WHEN category LIKE 'eu:%' THEN 'eu'
                        ELSE 'state/sector'
                    END as tier,
                    status,
                    COUNT(*)
                FROM query_pool GROUP BY tier, status
            """)
            tier_rows = cursor.fetchall()
        print("\n📦 QUERY POOL STATUS:", dict(rows))
        print("📦 BY PRIORITY TIER:")
        for tier, status, count in tier_rows:
            print(f"    {tier:<14} {status:<10} {count}")

# -------------------------------------------------------------------
# 5. Discovery Engine
# -------------------------------------------------------------------
class GermanEcosystemAgent:
    def __init__(self):
        self.db = ProgramDatabase()
        self.db.seed_pool()
        self.ai = genai.Client()  # Uses GEMINI_API_KEY
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def search_and_categorize(self, search_query: str, angle_label: str) -> int:
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
        (including EU-level programs available to German startups) from the web context below.
        Categorize each program strictly according to the schema. Write a clear, informative
        one-sentence description for each program suitable for public display on a directory website.
        Extract the application deadline exactly as the source states it (a date, a month/quarter,
        "rolling", "ongoing", etc.) — if no deadline is mentioned anywhere in the source, use "not stated".
        If the same organization runs multiple distinct programs, list each as a separate entry with
        its own specific URL. If no valid programs are present, return an empty list — do not invent entries.

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
                print(f"  ✅ Appended: {program.name} ({program.provider}) — deadline: {program.deadline}")
                print(f"     Tags -> [{tags_str}]")
                added += 1
            else:
                print(f"  ⏭️ Skipped (Duplicate, deadline refreshed): {program.name} — deadline: {program.deadline}")
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

    def run_until_novel_target(self, target_new=5, max_batches=15, batch_size=12):
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
            (name, provider, type_tag, funding_tag, stage_tag, eligibility_tag, backing_tag,
             bmwk_tag, exist_tag, focus, url, desc, deadline, checked_at) = row
            data.append({
                "name": name, "provider": provider, "type": type_tag, "funding": funding_tag,
                "stage": stage_tag, "eligibility": eligibility_tag, "backing": backing_tag,
                "bmwk": bmwk_tag, "exist": exist_tag,
                "focus": focus.split(",") if focus else [],
                "url": url, "description": desc,
                "deadline": deadline,
                "deadline_checked_at": checked_at
            })
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"🌐 Web Dashboard Export: Written {len(data)} unique record(s) to '{json_path}'.")

# -------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    agent = GermanEcosystemAgent()

    agent.run_until_novel_target(target_new=5, max_batches=15, batch_size=12)

    agent.db.deduplicate_database()
    agent.db.maybe_run_fuzzy_sweep(interval_days=0)
    agent.export_to_json()
    agent.db.print_terminal_summary()
