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

        search_response = self.tavily.search(query=search_query, max_results=5)
        web_context = "\n\n".join([
            f"Source URL: {res['url']}\nContent: {res['content']}"
            for res in search_response.get("results", [])
        ])

        prompt = f"""
        Analyze the web content below to discover grants, accelerators, incubators, and VC programs in Germany.
        Extract details and categorize strictly using the provided schema.

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
        
        added, duplicates = 0, 0
        for program in extracted_data.programs:
            if self.db.insert_program(program):
                tags_str = f"type:{program.type_tag} | funding:{program.funding_tag} | stage:{program.stage_tag} | bmwk:{program.bmwk_tag} | exist:{program.exist_tag}"
                print(f"  ✅ Appended: {program.name} ({program.provider})")
                print(f"     Tags -> [{tags_str}]")
                added += 1
            else:
                print(f"  ⏭️ Skipped (Already in DB): {program.name}")
                duplicates += 1

        print(f"Query Finished. Added: {added} | Skipped Duplicates: {duplicates}")

    def run(self):
        target_queries = [
            "German startup grants EXIST Gründungsstipendium Forschungstransfer 2026",
            "BMWK Digital Hub Initiative accelerators incubators Germany",
            "High-Tech Gründerfonds seed capital deeptech electronics AI Germany",
            "German university startup incubator EXIST partner network",
            "Federal state startup subsidies public VC programs Germany 2026"
        ]

        for query in target_queries:
            self.search_and_categorize(query)

    def export_to_json(self, json_path="grants.json"):
        """Exports SQLite records to grants.json for GitHub Pages."""
        programs = self.db.list_all_programs()
        data = []
        for row in programs:
            data.append({
                "name": row[0],
                "type": row[1],
                "funding": row[2],
                "stage": row[3],
                "eligibility": row[4],
                "backing": row[5],
                "bmwk": row[6],
                "exist": row[7],
                "focus": row[8].split(",") if row[8] else [],
                "url": row[9]
            })
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n🌐 Exported {len(data)} programs to '{json_path}' for web dashboard.")

# -------------------------------------------------------------------
# Execution & Export
# -------------------------------------------------------------------
if __name__ == "__main__":
    agent = GermanEcosystemAgent()
    agent.run()
    agent.export_to_json()
