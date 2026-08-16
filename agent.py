import os
import sqlite3
import json
from typing import List, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tavily import TavilyClient

# -------------------------------------------------------------------
# 1. Pydantic Schema Enforcing Your 8 Taxonomy Dimensions
# -------------------------------------------------------------------
class ProgramItem(BaseModel):
    name: str = Field(description="Official name of the program, grant, accelerator, or entity")
    provider: str = Field(description="Managing organization, university, state agency, or fund")
    url: str = Field(description="Direct official link to the program or application page")
    description: str = Field(description="Brief single-sentence overview of the offer")
    
    # 8 Taxonomy Categories
    type_tag: Literal["grant", "accelerator", "incubator", "vc"] = Field(
        description="Program Structure: grant (non-dilutive payout), accelerator (cohort-based), incubator (open-ended support/lab), vc (venture capital equity fund)"
    )
    funding_tag: Literal["equity-free", "zero-equity", "equity", "subsidy"] = Field(
        description="Funding Mechanics: equity-free (cash/stipend no shares), zero-equity (mentorship/credits only), equity (shares/SAFE/convertible), subsidy (reimbursement/matching)"
    )
    stage_tag: Literal["pre-seed", "seed", "scale"] = Field(
        description="Startup Stage: pre-seed (idea/MVP/prototyping), seed (working product/early traction), scale (post-revenue expansion)"
    )
    eligibility_tag: Literal["de-academic", "de-entity", "international"] = Field(
        description="Eligibility: de-academic (German university students/graduates/researchers), de-entity (registered German UG/GmbH required), international (open to foreign founders)"
    )
    backing_tag: Literal["public", "ppp", "private"] = Field(
        description="Backing Capital: public (tax money), ppp (public-private partnership like HTGF), private (corporate/private VC)"
    )
    bmwk_tag: Literal["direct", "mandated", "none"] = Field(
        description="BMWK Affiliation: direct (direct ministry initiative e.g. EXIST, Digital Hub), mandated (heavily backed e.g. HTGF), none (independent/regional state)"
    )
    exist_tag: Literal["grant", "partner", "none"] = Field(
        description="EXIST Ecosystem: grant (actual EXIST federal grant), partner (university incubator/network sponsor), none (outside EXIST)"
    )
    focus_tags: List[Literal["software", "electronics", "ai", "deeptech"]] = Field(
        description="Technology Focus: software (SaaS/cloud), electronics (hardware/embedded/IoT/sensors), ai (ML/LLMs/analytics), deeptech (high-risk R&D). Omit physics/chem/math."
    )

class ProgramList(BaseModel):
    programs: List[ProgramItem]

# -------------------------------------------------------------------
# 2. SQLite Database Engine (Persistent Tag Storage)
# -------------------------------------------------------------------
class ProgramDatabase:
    def __init__(self, db_path="german_startup_programs.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS programs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    provider TEXT,
                    url TEXT UNIQUE,
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

    def insert_program(self, prog: ProgramItem) -> bool:
        """Inserts a program into SQLite. Returns False if duplicate URL is found."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO programs (
                        name, provider, url, description,
                        type_tag, funding_tag, stage_tag, eligibility_tag,
                        backing_tag, bmwk_tag, exist_tag, focus_tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prog.name, prog.provider, prog.url, prog.description,
                    prog.type_tag, prog.funding_tag, prog.stage_tag, prog.eligibility_tag,
                    prog.backing_tag, prog.bmwk_tag, prog.exist_tag,
                    ",".join(prog.focus_tags)
                ))
            return True
        except sqlite3.IntegrityError:
            # Duplicate URL caught by UNIQUE constraint
            return False

    def list_all_programs(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, type_tag, funding_tag, stage_tag, eligibility_tag, 
                       backing_tag, bmwk_tag, exist_tag, focus_tags, url 
                FROM programs ORDER BY discovered_at DESC
            """)
            return cursor.fetchall()

# -------------------------------------------------------------------
# 3. Discovery & Taxonomy Categorization Core Logic
# -------------------------------------------------------------------
class GermanEcosystemAgent:
    def __init__(self):
        self.db = ProgramDatabase()
        self.ai = genai.Client()  # Uses GEMINI_API_KEY
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def search_and_categorize(self, search_query: str):
        print(f"\n🔍 Searching web: '{search_query}'...")

        # Step 1: Live web search targeting the German startup ecosystem
        search_response = self.tavily.search(query=search_query, max_results=7)
        web_context = "\n\n".join([
            f"Source URL: {res['url']}\nContent: {res['content']}"
            for res in search_response.get("results", [])
        ])

        # Step 2: Parse and categorize using Gemini 2.5
        prompt = f"""
        Analyze the web content below to discover grants, accelerators, incubators, and VC programs in Germany.
        For each valid opportunity, extract its details and categorize it strictly using the provided schema.

        Taxonomy Rules:
        - type: grant, accelerator, incubator, or vc
        - funding: equity-free, zero-equity, equity, or subsidy
        - stage: pre-seed, seed, or scale
        - eligibility: de-academic, de-entity, or international
        - backing: public, ppp, or private
        - bmwk: direct (e.g. EXIST, Digital Hub), mandated (e.g. HTGF), or none
        - exist: grant (actual stipend/grant), partner (university incubator sponsor), or none
        - focus: array subset of ["software", "electronics", "ai", "deeptech"]

        Web Context:
        {web_context}
        """

        response = self.ai.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProgramList,
                temperature=0.1
            )
        )

        extracted_data: ProgramList = response.parsed
        
        # Step 3: Append new programs / Skip existing list duplicates
        added, duplicates = 0, 0
        for program in extracted_data.programs:
            if self.db.insert_program(program):
                tags_str = f"type:{program.type_tag} | funding:{program.funding_tag} | stage:{program.stage_tag} | eligibility:{program.eligibility_tag} | bmwk:{program.bmwk_tag} | exist:{program.exist_tag}"
                print(f"  ✅ Appended: {program.name} ({program.provider})")
                print(f"     Tags -> [{tags_str}]")
                added += 1
            else:
                print(f"  ⏭️ Skipped (Already in DB): {program.name}")
                duplicates += 1

        print(f"Query Finished. Added: {added} | Skipped Duplicates: {duplicates}")

    def run(self):
        # Target queries covering the German innovation pipeline
        target_queries = [
            "German startup grants EXIST Gründungsstipendium Forschungstransfer 2026",
            "BMWK Digital Hub Initiative accelerators incubators Germany",
            "High-Tech Gründerfonds seed capital deeptech electronics AI Germany",
            "German university startup incubator EXIST partner network",
            "Federal state startup subsidies public VC programs Germany 2026"
        ]

        for query in target_queries:
            self.search_and_categorize(query)

# -------------------------------------------------------------------
# Execution & Summary Display
# -------------------------------------------------------------------
if __name__ == "__main__":
    agent = GermanEcosystemAgent()
    agent.run()

    print("\n" + "="*80)
    print("Compiled Database Summary (With 8 Taxonomy Tags)")
    print("="*80)
    
    all_programs = agent.db.list_all_programs()
    for row in all_programs:
        print(f"\n• {row[0]}")
        print(f"  Tags: type:{row[1]} | funding:{row[2]} | stage:{row[3]} | eligibility:{row[4]} | backing:{row[5]} | bmwk:{row[6]} | exist:{row[7]} | focus:{row[8]}")
        print(f"  URL:  {row[9]}")
