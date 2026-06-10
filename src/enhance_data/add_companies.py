import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List, Set, Tuple

import polars as pl
from diskcache import Cache
from tldextract import tldextract
from tqdm import tqdm

from dataloader import (
    TABLE_SCHEMAS,
    IndividualProjectContext,
)

PUBLIC_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "icloud.com",
    "aol.com",
    "zoho.com",
    "proton.me",
    "protonmail.com",
    "mail.com",
}

# Simple, global disk cache instance
cache = Cache("data/github_cache")

# ==========================================
# GITHUB API DATA FETCHER LAYER
# ==========================================


def _extract_email_domain_company(email: str) -> str | None:
    """Extracts a capitalized candidate organization from non-public corporate email domains."""
    if "@" not in email:
        return None
    domain_part = email.split("@")[-1].lower()
    if domain_part in PUBLIC_DOMAINS:
        return None

    extracted = tldextract.extract(domain_part)
    if extracted.domain:
        return extracted.domain


@cache.memoize(tag="_execute_graphql_request")
def _execute_graphql_request(query: str, variables: dict | None = None) -> dict:
    """Handles network transmission and rate-limiting for GitHub GraphQL API."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GitHub API token not found in environment variables (GITHUB_TOKEN)"
        )

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "Polars-Context-Enricher",
        },
    )

    while True:
        try:
            with urllib.request.urlopen(req) as response:
                response_data = json.loads(response.read().decode())
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                reset_time = e.headers.get("X-RateLimit-Reset")
                sleep_duration = (
                    max(float(reset_time) - time.time(), 0) + 2 if reset_time else 60
                )
                print(
                    f"\n ⚠️ Rate limit hit. Cooling down for {int(sleep_duration)} seconds..."
                )
                time.sleep(sleep_duration)
                continue
            raise

    if "errors" in response_data:
        for error in response_data["errors"]:
            if error.get("type") == "NOT_FOUND":
                print(f"⚠️ No results found: {error['message']}")
                continue
            else:
                raise RuntimeError(
                    f"GraphQL Query execution failed: {response_data['errors']}"
                )

    return response_data.get("data", {})


def _extract_companies_from_github_nodes(
    nodes: list[dict], search_term: str
) -> set[str]:
    """Extracts and cleans company and organization names from GitHub user nodes."""
    if not nodes:
        return set()

    companies = set()
    for node in nodes:
        if node.get("username") == "Unknown":
            continue
        companies.update(_extract_companies_from_github_node(node))

    return companies


def _extract_companies_from_github_node(node: dict) -> set[str]:
    """Extracts company profile text and explicit GitHub organization list memberships."""
    if node is None:
        return set()

    companies = set()
    pattern = r"[,@|]|\bpreviously\b|previous\b|\bprev\b|\band\b"

    # 1. Parse corporate/company string field from profile
    company_field = node.get("company")
    if company_field:
        raw_companies = re.split(pattern, company_field, flags=re.IGNORECASE)
        for comp in raw_companies:
            cleaned = comp.strip()
            if cleaned:
                companies.add(cleaned)

    # 2. Extract structured user organization memberships
    org_nodes = (
        node.get("organizations", {}).get("nodes", [])
        if node.get("organizations")
        else []
    )
    for org in org_nodes:
        if org and org.get("name"):
            cleaned_org = org["name"].strip()
            if cleaned_org:
                companies.add(cleaned_org)

    return companies


# ==========================================
# MEMOIZED BATCH FUNCTIONS
# ==========================================


def _fetch_companies_graphql_batch(
    pending_emails: tuple[str, ...],
) -> dict[str, list[str]]:
    """Queries the GitHub GraphQL API for a fully packed batch of missing emails."""
    results = {}
    query_fragments = []
    variables = {}
    var_definitions = []

    for idx, email in enumerate(pending_emails):
        var_name = f"email_query_{idx}"
        var_definitions.append(f"${var_name}: String!")
        variables[var_name] = f"{email} in:email"

        query_fragments.append(
            f"email_{idx}: search(type: USER, query: ${var_name}, first: 1) {{ "
            f"  nodes {{ ... on User {{ company organizations(first: 10) {{ nodes {{ name }} }} }} }} "
            f"}}"
        )

    graphql_query = (
        f"query({', '.join(var_definitions)}) {{ {' '.join(query_fragments)} }}"
    )
    data_map = _execute_graphql_request(graphql_query, variables)

    for idx, email in enumerate(pending_emails):
        alias_key = f"email_{idx}"
        nodes = data_map.get(alias_key, {}).get("nodes", [])
        results[email] = list(_extract_companies_from_github_nodes(nodes, email))

    return results


def _fetch_companies_by_username_batch(
    pending_usernames: tuple[str, ...],
) -> dict[str, list[str]]:
    """Queries the GitHub GraphQL API for companies matching specific usernames directly."""
    results = {}
    query_fragments = []
    for idx, username in enumerate(pending_usernames):
        query_fragments.append(
            f'user_{idx}: user(login: "{username}") {{ company organizations(first: 10) {{ nodes {{ name }} }} }}'
        )

    graphql_query = f"query {{ {' '.join(query_fragments)} }}"
    data_map = _execute_graphql_request(graphql_query)

    for idx, username in enumerate(pending_usernames):
        alias_key = f"user_{idx}"
        user_node = data_map.get(alias_key)
        companies = _extract_companies_from_github_node(user_node)

        results[username] = list(companies)

    return results


def _fetch_companies_by_fullname_batch(
    pending_names: tuple[str, ...],
) -> dict[str, set[str]]:
    """Queries the GitHub GraphQL API for companies by full name."""
    results = {}
    query_fragments = []
    variables = {}
    var_definitions = []

    for idx, name in enumerate(pending_names):
        var_name = f"name_query_{idx}"
        var_definitions.append(f"${var_name}: String!")
        variables[var_name] = f"{name} in:name"

        query_fragments.append(
            f"name_{idx}: search(type: USER, query: ${var_name}, first: 2) {{ "
            f"  nodes {{ ... on User {{ company organizations(first: 10) {{ nodes {{ name }} }} }} }} "
            f"}}"
        )

    graphql_query = (
        f"query({', '.join(var_definitions)}) {{ {' '.join(query_fragments)} }}"
    )
    data_map = _execute_graphql_request(graphql_query, variables)

    for idx, name in enumerate(pending_names):
        alias_key = f"name_{idx}"
        nodes = data_map.get(alias_key, {}).get("nodes", [])
        results[name] = _extract_companies_from_github_nodes(nodes, name)

    return results


# ==========================================
# STATE AND RECORD SYNCHRONIZATION
# ==========================================


def _update_context_records(
    person_id: int,
    discovered_companies: set[str],
    existing_orgs: dict[str, int],
    existing_affils_set: set[tuple[int, int]],
    max_org_id: int,
    new_orgs_records: list[dict],
    new_affils_records: list[dict],
) -> int:
    """Updates internal mapping states and queues up unique data records for dataframes."""
    for comp_name in discovered_companies:
        if not comp_name:
            continue

        if comp_name not in existing_orgs:
            max_org_id += 1
            existing_orgs[comp_name] = max_org_id
            new_orgs_records.append(
                {"organisation_id": max_org_id, "organisation_name": comp_name}
            )

        target_org_id = existing_orgs[comp_name]

        if (person_id, target_org_id) not in existing_affils_set:
            new_affils_records.append(
                {"organisation_id": target_org_id, "person_id": person_id}
            )
            existing_affils_set.add((person_id, target_org_id))

    return max_org_id


# ==========================================
# CORE PIPELINE ENGINE
# ==========================================


def enrich_project_contexts_with_companies(
    contexts: list[IndividualProjectContext], batch_size: int = 50
) -> list[IndividualProjectContext]:
    """Processes contexts sequentially executing cleanly decoupled discovery tactics."""
    enriched_contexts = []
    print("\n🚀 Starting Enrichment Pipeline...")

    for ctx in contexts:
        orgs_df = ctx.organisations.clone()
        affils_df = ctx.affiliations.clone()
        idents_df = ctx.person_identifiers
        persons_df = ctx.persons

        existing_affils_set = set(
            affils_df.select(["person_id", "organisation_id"]).iter_rows()
        )
        max_value = orgs_df["organisation_id"].max()
        max_org_id = int(max_value) if max_value is not None else 0
        existing_orgs = dict(
            orgs_df.select(["organisation_name", "organisation_id"]).iter_rows()
        )

        initial_org_count, initial_affils_count = (
            len(existing_orgs),
            len(existing_affils_set),
        )
        new_orgs_records, new_affils_records = [], []

        rows = idents_df.to_dicts() if not idents_df.is_empty() else []
        resolved_companies: Dict[Tuple[int, str], Set[str]] = {
            (row["person_id"], str(row["identifier"]).strip()): set() for row in rows
        }

        person_rows = persons_df.to_dicts() if not persons_df.is_empty() else []
        for p_row in person_rows:
            p_id = p_row["person_id"]
            fallback_key = (p_id, "full_name_lookup")
            if not any(k[0] == p_id for k in resolved_companies):
                resolved_companies[fallback_key] = set()

        # ------------------------------------------------------------------
        # TACTIC 1: Email Heuristic Domain Profiling (Local Only)
        # ------------------------------------------------------------------
        for row in rows:
            ident = str(row["identifier"]).strip()
            if "@" in ident and len(ident.split("@")) == 2:
                domain_company = _extract_email_domain_company(ident)
                if domain_company:
                    resolved_companies[(row["person_id"], ident)].add(domain_company)

        # ------------------------------------------------------------------
        # TACTIC 2: GitHub Graph API Lookups via Email Strings
        # ------------------------------------------------------------------
        emails = [
            str(r["identifier"]).strip()
            for r in rows
            if "@" in str(r["identifier"]) and len(str(r["identifier"]).split("@")) == 2
        ]
        if emails:
            unique_emails = sorted(list(set(emails)))
            for i in tqdm(
                range(0, len(unique_emails), batch_size),
                desc=f"📦 Fetching Email Batches ({ctx.project_name})",
            ):
                batch = tuple(unique_emails[i : i + batch_size])
                api_results = _fetch_companies_graphql_batch(batch)

                for row_key, discovered in resolved_companies.items():
                    _, ident = row_key
                    if ident in api_results and api_results[ident]:
                        discovered.update(api_results[ident])

        # ------------------------------------------------------------------
        # TACTIC 3: GitHub Graph API Lookups via Username Attributes
        # ------------------------------------------------------------------
        usernames = [
            str(r["identifier"]).strip()
            for r in rows
            if str(r.get("domain", "")).lower() == "github.com"
            and str(r["identifier_type"]).lower() == "username"
        ]
        if usernames:
            unique_usernames = sorted(list(set(usernames)))
            for i in tqdm(
                range(0, len(unique_usernames), batch_size),
                desc=f"👤 Fetching Username Batches ({ctx.project_name})",
            ):
                batch = tuple(unique_usernames[i : i + batch_size])
                api_results = _fetch_companies_by_username_batch(batch)

                for row_key, discovered in resolved_companies.items():
                    _, ident = row_key
                    if ident in api_results and api_results[ident]:
                        discovered.update(api_results[ident])

        # ------------------------------------------------------------------
        # TACTIC 4: GitHub Graph API Lookups via Full Name
        # ------------------------------------------------------------------
        names = [
            str(p_row["full_name"]).strip()
            for p_row in person_rows
            if p_row.get("full_name")
        ]
        if names:
            unique_names = sorted(list(set(names)))
            name_to_row_keys_map: Dict[str, List[Tuple[int, str]]] = {}

            for p_row in person_rows:
                p_id = p_row["person_id"]
                fn = str(p_row.get("full_name", "")).strip()
                if not fn:
                    continue
                target_keys = [k for k in resolved_companies if k[0] == p_id] or [
                    (p_id, "full_name_lookup")
                ]
                name_to_row_keys_map.setdefault(fn, []).extend(target_keys)

            for i in tqdm(
                range(0, len(unique_names), batch_size),
                desc=f"📝 Fetching Full Name Batches ({ctx.project_name})",
            ):
                batch = tuple(unique_names[i : i + batch_size])
                api_results = _fetch_companies_by_fullname_batch(batch)

                for name_item, company_found in api_results.items():
                    if company_found and name_item in name_to_row_keys_map:
                        for target_key in name_to_row_keys_map[name_item]:
                            resolved_companies[target_key].update(company_found)

        # ------------------------------------------------------------------
        # PHASE 5: State Pipeline Sync & DataFrame Materialization
        # ------------------------------------------------------------------
        for (person_id, _), companies in resolved_companies.items():
            if companies:
                max_org_id = _update_context_records(
                    person_id=person_id,
                    discovered_companies=companies,
                    existing_orgs=existing_orgs,
                    existing_affils_set=existing_affils_set,
                    max_org_id=max_org_id,
                    new_orgs_records=new_orgs_records,
                    new_affils_records=new_affils_records,
                )

        if new_orgs_records:
            orgs_df = pl.concat(
                [
                    orgs_df,
                    pl.DataFrame(
                        new_orgs_records, schema=TABLE_SCHEMAS["Organisation"]
                    ),
                ]
            ).unique(subset=["organisation_name"])

        if new_affils_records:
            affils_df = pl.concat(
                [
                    affils_df,
                    pl.DataFrame(
                        new_affils_records, schema=TABLE_SCHEMAS["Affiliation"]
                    ),
                ]
            ).unique()

        ctx.organisations = orgs_df
        ctx.affiliations = affils_df
        enriched_contexts.append(ctx)

        print(
            f"✅ Completed Project {ctx.project_name}: +{len(existing_orgs) - initial_org_count} companies, +{len(existing_affils_set) - initial_affils_count} affiliations"
        )

    return enriched_contexts
