"""
DataCleaner — normalize, deduplicate, and structure raw parsed PubMed records.
Fully query-agnostic: operates on whatever papers were retrieved without
any domain-specific assumptions.
"""

import html
import logging
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from thefuzz import fuzz

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_AUTHOR_SIMILARITY_THRESHOLD = 90
_TITLE_SIMILARITY_THRESHOLD = 95

# Common boilerplate phrases found at the end of PubMed abstracts
_BOILERPLATE_PATTERNS = [
    r"copyright\s+©.*?$",
    r"©\s*\d{4}.*?$",
    r"all rights reserved\.?$",
    r"published by elsevier.*?$",
    r"this article is protected by copyright.*?$",
    r"for personal use only.*?$",
    r"no commercial use.*?$",
    r"this is an open.access article.*?$",
]
_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE | re.MULTILINE
)

# HTML/XML entity and tag patterns
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# PubMed month name → zero-padded number
_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    "january": "01", "february": "02", "march": "03", "april": "04",
    "june": "06", "july": "07", "august": "08", "september": "09",
    "october": "10", "november": "11", "december": "12",
}


class DataCleaner:
    """
    Applies normalization, deduplication, and structuring steps to raw
    parsed PubMed records. Works on any biomedical domain without modification.
    """

    # ── Author normalization ──────────────────────────────────────────────────

    def normalize_author_name(self, name: str) -> str:
        """
        Standardize author name to 'Lastname, Firstname' format.
        Handles 'Firstname Lastname', 'Lastname FM' (initials), and
        'Lastname, FM' variants.
        """
        if not name or not name.strip():
            return ""
        name = name.strip()

        # Already in 'Lastname, Firstname/Initials' format
        if "," in name:
            parts = [p.strip() for p in name.split(",", 1)]
            last = self._title_case(parts[0])
            first = parts[1].strip() if len(parts) > 1 else ""
            return f"{last}, {first}" if first else last

        # 'Lastname FM' — last token is all uppercase initials
        tokens = name.split()
        if len(tokens) >= 2:
            possible_initials = tokens[-1]
            if possible_initials.isupper() and len(possible_initials) <= 3:
                last = self._title_case(" ".join(tokens[:-1]))
                return f"{last}, {possible_initials}"
            # 'Firstname Lastname' — reverse
            last = self._title_case(tokens[-1])
            first = " ".join(self._title_case(t) for t in tokens[:-1])
            return f"{last}, {first}"

        return self._title_case(name)

    @staticmethod
    def _title_case(text: str) -> str:
        """Title-case with Unicode, hyphen, and apostrophe support (e.g. O’Brien, García)."""
        # Capitalize the first Unicode letter after any word boundary
        return re.sub(
            r"\b(\w)",
            lambda m: m.group(1).upper(),
            text.lower(),
            flags=re.UNICODE,
        )

    def deduplicate_authors(self, author_list: List[str]) -> List[str]:
        """
        Cluster similar author names using fuzzy matching (threshold=90).
        Returns list with canonical names (the most common form in each cluster).
        """
        if not author_list:
            return []

        normalized = [self.normalize_author_name(a) for a in author_list]
        clusters: List[List[int]] = []
        assigned = [False] * len(normalized)

        for i, name_i in enumerate(normalized):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, len(normalized)):
                if not assigned[j]:
                    score = fuzz.token_sort_ratio(name_i, normalized[j])
                    if score >= _AUTHOR_SIMILARITY_THRESHOLD:
                        cluster.append(j)
                        assigned[j] = True
            clusters.append(cluster)

        canonical = []
        for cluster in clusters:
            # Pick the longest (most complete) name as canonical
            best = max(cluster, key=lambda idx: len(normalized[idx]))
            canonical.append(normalized[best])

        return canonical

    # ── Keyword normalization ─────────────────────────────────────────────────

    @staticmethod
    def normalize_keyword(keyword: str) -> str:
        """
        Lowercase, strip, and remove punctuation artifacts from a keyword.
        No MeSH lookup — stays domain-agnostic.
        """
        if not keyword:
            return ""
        kw = keyword.lower().strip()
        # Remove leading/trailing punctuation and internal double spaces
        kw = re.sub(r"[^\w\s\-\/]", " ", kw)
        kw = _WHITESPACE_RE.sub(" ", kw).strip()
        return kw

    # ── Abstract cleaning ─────────────────────────────────────────────────────

    def clean_abstract(self, text: str) -> str:
        """
        Return clean plain text for an abstract.
        Removes HTML, normalizes unicode, strips boilerplate.
        """
        if not text:
            return ""

        # Decode HTML entities (&amp; &lt; etc.)
        text = html.unescape(text)

        # Strip HTML tags
        text = _HTML_TAG_RE.sub(" ", text)

        # Normalize unicode (e.g. ligatures, non-breaking spaces)
        text = unicodedata.normalize("NFKC", text)

        # Remove boilerplate copyright/publisher notices
        text = _BOILERPLATE_RE.sub("", text)

        # Collapse whitespace
        text = _WHITESPACE_RE.sub(" ", text).strip()

        return text

    # ── Duplicate detection ───────────────────────────────────────────────────

    def detect_duplicates(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact PMID duplicates and near-duplicate titles (similarity ≥ 95).
        Returns a deduplicated DataFrame.
        """
        if papers_df.empty:
            return papers_df

        before = len(papers_df)

        # 1. Exact PMID deduplication
        papers_df = papers_df.drop_duplicates(subset=["pmid"], keep="first")

        # 2. Near-duplicate title detection
        titles = papers_df["title"].fillna("").tolist()
        pmids = papers_df["pmid"].tolist()
        drop_indices = set()

        for i in range(len(titles)):
            if i in drop_indices:
                continue
            for j in range(i + 1, len(titles)):
                if j in drop_indices:
                    continue
                if not titles[i] or not titles[j]:
                    continue
                score = fuzz.token_sort_ratio(titles[i], titles[j])
                if score >= _TITLE_SIMILARITY_THRESHOLD:
                    logger.debug(
                        "Near-duplicate titles (score=%d): '%s' vs '%s'",
                        score,
                        pmids[i],
                        pmids[j],
                    )
                    drop_indices.add(j)

        if drop_indices:
            keep_mask = [i not in drop_indices for i in range(len(papers_df))]
            papers_df = papers_df[keep_mask].reset_index(drop=True)

        after = len(papers_df)
        if before != after:
            logger.info("Deduplication: %d → %d papers (removed %d)", before, after, before - after)

        return papers_df.reset_index(drop=True)

    # ── Date normalization ────────────────────────────────────────────────────

    def normalize_date(self, date_string: Optional[str]) -> Optional[datetime]:
        """
        Parse the many PubMed date formats and return a datetime object.
        Returns None if the string cannot be parsed.
        """
        if not date_string:
            return None

        s = date_string.strip()

        # Try ISO-like formats: "2023-Jan-15", "2023-01-15", "2023-Jan", "2023"
        patterns = [
            ("%Y-%m-%d", r"^\d{4}-\d{2}-\d{2}$"),
            ("%Y-%m", r"^\d{4}-\d{2}$"),
            ("%Y", r"^\d{4}$"),
        ]
        for fmt, pattern in patterns:
            if re.match(pattern, s):
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    pass

        # Month name variants: "2023 Jan 15", "2023 Jan"
        month_name_re = re.compile(
            r"^(\d{4})\s+([A-Za-z]+)(?:\s+(\d{1,2}))?$"
        )
        m = month_name_re.match(s)
        if m:
            year, month_str, day = m.group(1), m.group(2).lower(), m.group(3)
            month_num = _MONTH_MAP.get(month_str[:3])
            if month_num:
                date_str = f"{year}-{month_num}"
                if day:
                    date_str += f"-{day.zfill(2)}"
                    try:
                        return datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        pass
                try:
                    return datetime.strptime(date_str, "%Y-%m")
                except ValueError:
                    pass

        # Hyphen month name: "2023-Jan-15"
        hyphen_re = re.compile(r"^(\d{4})-([A-Za-z]+)(?:-(\d{1,2}))?$")
        m = hyphen_re.match(s)
        if m:
            year, month_str, day = m.group(1), m.group(2).lower(), m.group(3)
            month_num = _MONTH_MAP.get(month_str[:3])
            if month_num:
                date_str = f"{year}-{month_num}"
                if day:
                    date_str += f"-{day.zfill(2)}"
                    try:
                        return datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        pass
                try:
                    return datetime.strptime(date_str, "%Y-%m")
                except ValueError:
                    pass

        # MedlineDate ranges like "2023 Jan-Feb" — use the start
        medline_re = re.compile(r"^(\d{4})\s+([A-Za-z]+)")
        m = medline_re.match(s)
        if m:
            year, month_str = m.group(1), m.group(2).lower()
            month_num = _MONTH_MAP.get(month_str[:3])
            if month_num:
                try:
                    return datetime.strptime(f"{year}-{month_num}", "%Y-%m")
                except ValueError:
                    pass

        # Last resort: extract 4-digit year
        year_match = re.search(r"\b(\d{4})\b", s)
        if year_match:
            try:
                return datetime.strptime(year_match.group(1), "%Y")
            except ValueError:
                pass

        logger.warning("Could not parse date string: %r", date_string)
        return None

    def normalize_date_str(self, date_string: Optional[str]) -> Optional[str]:
        """Return ISO date string (YYYY-MM-DD or YYYY-MM or YYYY) from raw PubMed date."""
        dt = self.normalize_date(date_string)
        if dt is None:
            return None
        return dt.strftime("%Y-%m-%d")

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run_full_pipeline(
        self, raw_papers: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Apply all cleaning steps in sequence to a list of raw parsed paper dicts.

        Steps:
          1. Convert to DataFrame
          2. Clean abstracts
          3. Normalize author names within each paper
          4. Normalize author-assigned keywords
          5. Normalize publication dates
          6. Deduplicate
          7. Add derived columns (pub_year, pub_month)

        Returns a clean pandas DataFrame ready for NLP processing.
        This method is fully query-agnostic — it processes whatever
        papers are in *raw_papers* without any domain assumptions.
        """
        if not raw_papers:
            logger.warning("run_full_pipeline received empty paper list")
            return pd.DataFrame()

        logger.info("Cleaning pipeline: starting with %d raw papers", len(raw_papers))

        # ── Step 1: Build DataFrame from core scalar fields ───────────────────
        scalar_fields = [
            "pmid", "title", "abstract", "journal", "volume", "issue",
            "pages", "pub_date", "doi", "language", "query_used",
        ]
        rows = []
        for p in raw_papers:
            row = {f: p.get(f) for f in scalar_fields}
            # Store list fields as-is for later pipeline stages
            row["authors"]           = p.get("authors", [])
            row["author_keywords"]   = p.get("author_keywords", [])
            row["mesh_terms"]        = p.get("mesh_terms", [])
            row["chemicals"]         = p.get("chemicals", [])
            row["publication_types"] = p.get("publication_types", [])
            row["grants"]            = p.get("grants", [])
            rows.append(row)

        df = pd.DataFrame(rows)

        # ── Step 2: Clean abstracts ───────────────────────────────────────────
        df["abstract"] = df["abstract"].apply(
            lambda x: self.clean_abstract(x) if isinstance(x, str) else x
        )

        # ── Step 3: Normalize author names ────────────────────────────────────
        def _normalize_authors(author_list):
            if not isinstance(author_list, list):
                return []
            return [
                {
                    "name": self.normalize_author_name(a.get("name", "")),
                    "affiliation": a.get("affiliation"),
                }
                for a in author_list
                if a.get("name")
            ]

        df["authors"] = df["authors"].apply(_normalize_authors)

        # ── Step 4: Normalize author-assigned keywords ────────────────────────
        df["author_keywords"] = df["author_keywords"].apply(
            lambda kws: [self.normalize_keyword(k) for k in (kws or []) if k]
        )

        # ── Step 5: Normalize publication dates ───────────────────────────────
        df["pub_date_normalized"] = df["pub_date"].apply(self.normalize_date_str)

        def _safe_year(date_str):
            if not date_str:
                return None
            try:
                return int(date_str[:4])
            except (ValueError, TypeError):
                return None

        def _safe_month(date_str):
            if not date_str or len(date_str) < 7:
                return None
            try:
                return int(date_str[5:7])
            except (ValueError, TypeError):
                return None

        df["pub_year"]  = df["pub_date_normalized"].apply(_safe_year)
        df["pub_month"] = df["pub_date_normalized"].apply(_safe_month)

        # ── Step 6: Deduplicate ───────────────────────────────────────────────
        df = self.detect_duplicates(df)

        # ── Step 7: Drop rows with neither title nor abstract ─────────────────
        before = len(df)
        mask = df["title"].notna() | df["abstract"].notna()
        df = df[mask].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.warning(
                "Dropped %d records with no title and no abstract", dropped
            )

        logger.info("Cleaning pipeline complete: %d papers ready for NLP", len(df))
        return df
