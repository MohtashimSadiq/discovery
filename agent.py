import os
import sqlite3
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tavily import TavilyClient

# -------------------------------------------------------------------
# 1. Pydantic Schema for Structured Output
# -------------------------------------------------------------------
class GrantItem(BaseModel):
    grant_name: str = Field(description="Official name of the grant program")
    provider: str = Field(description="Organization, foundation, or government body offering the grant")
    stage: str = Field(description="Target startup stage: Pre-Seed, Seed, Series A, Series B+, or All Stages")
    max_amount: str = Field(description="Maximum funding amount or estimated financial range")
    eligibility: str = Field(description="Key eligibility criteria or geographic constraints")
    url: str = Field(description="Direct URL to the grant application or provider page")

class GrantList(BaseModel):
    grants: List[GrantItem]

# -------------------------------------------------------------------
# 2. SQLite Persistence & Deduplication Engine
# -------------------------------------------------------------------
class GrantDatabase:
    def __init__(self, db_path="startup_grants.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS grants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    grant_name TEXT NOT NULL,
                    provider TEXT,
                    stage TEXT NOT NULL,
                    max_amount TEXT,
                    eligibility TEXT,
                    url TEXT UNIQUE,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def insert_grant(self, grant: GrantItem) -> bool:
        """Inserts a grant if it doesn't already exist in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO grants (grant_name, provider, stage, max_amount, eligibility, url)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (grant.grant_name, grant.provider, grant.stage, grant.max_amount, grant.eligibility, grant.url))
            return True
        except sqlite3.IntegrityError:
            # Duplicate URL caught by UNIQUE constraint
            return False

    def list_all_grants(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT stage, grant_name, provider, max_amount, url FROM grants ORDER BY stage")
            return cursor.fetchall()

# -------------------------------------------------------------------
# 3. Discovery Agent Core Logic
# -------------------------------------------------------------------
class GrantDiscoveryAgent:
    def __init__(self):
        self.db = GrantDatabase()
        self.ai = genai.Client()  # Uses GEMINI_API_KEY environment variable
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def search_stage_grants(self, stage: str):
        query = f"active non-dilutive grants funding application for {stage} stage startups 2026"
        print(f"\n🔍 Searching for {stage} stage grants...")

        # Step 1: Live web search
        search_response = self.tavily.search(query=query, max_results=5)
        web_context = "\n\n".join([
            f"Source: {res['url']}\nContent: {res['content']}"
            for res in search_response.get("results", [])
        ])

        # Step 2: Extract structured records via Gemini 2.5
        prompt = f"""
        Extract all active startup grant programs from the web content below.
        Extract the official grant name, provider, stage ({stage}), max funding amount, key eligibility, and source URL.
        Ignore general news articles unless a specific grant opportunity is detailed.

        Web Results:
        {web_context}
        """

        response = self.ai.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GrantList,
                temperature=0.1
            )
        )

        extracted_data: GrantList = response.parsed
        
        # Step 3: Deduplicate & Append
        added, duplicates = 0, 0
        for grant in extracted_data.grants:
            if self.db.insert_grant(grant):
                print(f"  ✅ Appended: [{grant.stage}] {grant.grant_name} ({grant.max_amount})")
                added += 1
            else:
                print(f"  ⏭️ Skipped Duplicate: {grant.grant_name}")
                duplicates += 1

        print(f"Finished {stage}. Added: {added} | Duplicates Skipped: {duplicates}")

    def run(self, stages: List[str]):
        for stage in stages:
            self.search_stage_grants(stage)

# -------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    agent = GrantDiscoveryAgent()
    
    # Target startup stages
    target_stages = ["Pre-Seed", "Seed", "Series A"]
    agent.run(target_stages)

    # Print summary of compiled grants
    print("\n--- Compiled Database Summary ---")
    all_grants = agent.db.list_all_grants()
    for row in all_grants:
        print(f"• [{row[0]}] {row[1]} by {row[2]} | Max: {row[3]} | URL: {row[4]}")
