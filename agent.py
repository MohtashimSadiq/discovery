import os
import re
import json
import sqlite3
from typing import List, Literal
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tavily import TavilyClient

# -------------------------------------------------------------------
# 1. Helper Functions for String Normalization & Fuzzy Deduplication
# -------------------------------------------------------------------
def clean_string(text: str) -> str:
    """Removes spaces, special characters, and casing for fuzzy matching."""
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

def normalize_url(url: str) -> str:
    """Extracts base domain and clean path without trailing slashes or www."""
    parsed = urlparse(url.lower().strip())
    netloc = parsed.netloc.replace("www.", "")
    return f"{parsed.scheme}://{netloc}{parsed.path}".rstrip("/")

# -------------------------------------------------------------------
# 2. Pydantic Schema Enforcing Your 8 Taxonomy Dimensions
# -------------------------------------------------------------------
class ProgramItem(BaseModel):
    name: str = Field(description="Official name of the program, grant, accelerator, or entity")
    provider: str = Field(description="Managing organization, university, state agency, or fund")
    url: str = Field(description="Direct official link to the program or application page")
    description: str = Field(description="Brief single-sentence overview of the offer")
    
    # 8 Taxonomy Categories
    type_tag: Literal["grant", "accelerator", "incubator", "vc"] = Field(
        description="Program Structure: grant, accelerator, incubator, or vc"
    )
    funding_tag: Literal["equity-free", "zero-equity", "equity", "subsidy"] = Field(
        description="Funding Mechanics: equity-free, zero-equity, equity, or subsidy"
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

# -------------------------------------------------------------------
# 3. Composite SQLite Database Engine with Automated Housekeeping
# -------------------------------------------------------------------
class ProgramDatabase:
    def __init__(self, db_path="german_startup_programs.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Upgrade database structure if using an older schema version
            cursor.execute("PRAGMA table_info(programs)")
            columns = [info[1] for info in cursor.fetchall()]
            if columns and "clean_name" not in columns:
                print("🔄 Updating SQLite database schema to support clean name indexing...")
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

    def is_duplicate(self, prog: ProgramItem) -> bool:
        """Checks if a program exists by normalized URL OR name + taxonomy fingerprint."""
        norm_url = normalize_url(prog.url)
        norm_name = clean_string(prog.name)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Exact Normalized URL Match
            cursor.execute("SELECT id FROM programs WHERE normalized_url = ?", (norm_url,))
            if cursor.fetchone():
                return True

            # 2. Name & Taxonomy Fingerprint Overlap Match
            cursor.execute("""
                SELECT clean_name FROM programs 
                WHERE type_tag = ? AND stage_tag = ? AND bmwk_tag = ?
            """, (prog.type_tag, prog.stage_tag, prog.bmwk_tag))
            
            for (existing_clean_name,) in cursor.fetchall():
                if norm_name in existing_clean_name or existing_clean_name in norm_name:
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

    def deduplicate_database(self):
        """Automated Housekeeping: Purges stray duplicate records from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM programs 
                WHERE id NOT IN (
                    SELECT MIN(id) 
                    FROM programs 
                    GROUP BY clean_name
                )
            """)
            deleted_count = cursor.rowcount
            print("\n" + "=" * 80)
            print("🧹 AUTOMATED HOUSEKEEPING REPORT")
            print("=" * 80)
            if deleted_count > 0:
                print(f"  • Purged {deleted_count} stray duplicate record(s) from SQLite database.")
            else:
                print("  • Database is clean. 0 duplicates found.")

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
        """Prints full database contents directly to stdout console."""
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

# -------------------------------------------------------------------
# 4. Discovery Engine with Regional Query Angles & Deep Parsing
# -------------------------------------------------------------------
class GermanEcosystemAgent:
    def __init__(self):
        self.db = ProgramDatabase()
        self.ai = genai.Client()  # Uses GEMINI_API_KEY
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def search_and_categorize(self, search_query: str, angle_label: str):
        print(f"\n🔍 Searching [{angle_label}]: '{search_query}'...")

        # Deeper search limit (max_results=10) with AI summary
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
        
        Task: Extract active German startup grants, accelerators, incubators, and VC programs from the web context below.
        Categorize each program strictly according to the schema.

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
                temperature=0.0  # Zero temperature for deterministic tagging
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
                print(f"  ⏭️ Skipped (Duplicate Match): {program.name}")
                duplicates += 1

        print(f"  --> Results for '{angle_label}': {added} Added | {duplicates} Duplicates Skipped")

    def run(self):
        # Targeted Regional & Sector Query Rotation Angles
        target_queries = [
            ("National / BMWK", "German startup grants EXIST Gründungsstipendium Forschungstransfer 2026"),
            ("Bavaria (South)", "Bavaria university startup grant FLÜGGE LfA Bayern Innovativ Bayern Kapital"),
            ("Hesse (Central)", "Hesse AI startup funding hessian.ai WIBank AI Startup Rising StartUpSecure"),
            ("Baden-Württemberg", "Baden-Württemberg startup grant L-Bank Junge Innovatoren CyberLab VCBW"),
            ("Berlin / Capital", "Berlin Brandenburg startup subsidy IBB Ventures ProFIT Innovation Seed")
        ]

        print("\n🚀 STARTING GERMAN STARTUP PROGRAM DISCOVERY ENGINE...")
        print("=" * 80)

        for angle_label, query in target_queries:
            self.search_and_categorize(query, angle_label)

    def export_to_json(self, json_path="grants.json"):
        """Exports SQLite database to grants.json for GitHub Pages."""
        programs = self.db.list_all_programs()
        data = []
        for row in programs:
            name, provider, type_tag, funding_tag, stage_tag, eligibility_tag, backing_tag, bmwk_tag, exist_tag, focus, url, desc = row
            data.append({
                "name": name,
                "provider": provider,
                "type": type_tag,
                "funding": funding_tag,
                "stage": stage_tag,
                "eligibility": eligibility_tag,
                "backing": backing_tag,
                "bmwk": bmwk_tag,
                "exist": exist_tag,
                "focus": focus.split(",") if focus else [],
                "url": url,
                "description": desc
            })
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"🌐 Web Dashboard Export: Written {len(data)} unique record(s) to '{json_path}'.")

# -------------------------------------------------------------------
# Execution & Terminal Output Reporting
# -------------------------------------------------------------------
if __name__ == "__main__":
    agent = GermanEcosystemAgent()
    
    # 1. Execute Regional Discovery
    agent.run()
    
    # 2. Execute Automated Housekeeping
    agent.db.deduplicate_database()
    
    # 3. Export to Web Dashboard JSON
    agent.export_to_json()
    
    # 4. Print Complete Directory Summary Directly to Terminal Logs
    agent.db.print_terminal_summary()
