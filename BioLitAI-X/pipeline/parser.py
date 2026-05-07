"""
XMLParser — parse PubMed XML records into structured Python dicts.
Handles all required fields including MeSH, chemicals, publication types, grants.
"""

import logging
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


def _text(element: Optional[ET.Element], default: Optional[str] = None) -> Optional[str]:
    if element is None:
        return default
    return (element.text or "").strip() or default


def _iter_text(root: ET.Element, xpath: str) -> List[str]:
    return [
        el.text.strip()
        for el in root.findall(xpath)
        if el.text and el.text.strip()
    ]


class XMLParser:
    """Parse raw PubMed XML into structured dicts ready for database storage."""

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_abstract(article: ET.Element) -> Optional[str]:
        """
        Extract abstract text, preserving structured section labels
        (Background, Methods, Results, Conclusion) when present.
        """
        abstract_el = article.find(".//Abstract")
        if abstract_el is None:
            return None

        parts: List[str] = []
        for text_el in abstract_el.findall("AbstractText"):
            label = text_el.get("Label")
            content = "".join(text_el.itertext()).strip()
            if not content:
                continue
            if label:
                parts.append(f"{label.upper()}: {content}")
            else:
                parts.append(content)

        return " ".join(parts) if parts else None

    @staticmethod
    def _extract_authors(article: ET.Element) -> List[Dict[str, str]]:
        authors = []
        for author_el in article.findall(".//AuthorList/Author"):
            last = _text(author_el.find("LastName"))
            fore = _text(author_el.find("ForeName"))
            initials = _text(author_el.find("Initials"))
            collective = _text(author_el.find("CollectiveName"))

            if collective:
                name = collective
            elif last and fore:
                name = f"{last}, {fore}"
            elif last and initials:
                name = f"{last}, {initials}"
            elif last:
                name = last
            else:
                continue

            affiliation_parts = []
            for aff_el in author_el.findall(".//AffiliationInfo/Affiliation"):
                txt = "".join(aff_el.itertext()).strip()
                if txt:
                    affiliation_parts.append(txt)
            affiliation = "; ".join(affiliation_parts) if affiliation_parts else None

            authors.append({"name": name, "affiliation": affiliation})
        return authors

    @staticmethod
    def _extract_author_keywords(medline: ET.Element) -> List[str]:
        return _iter_text(medline, ".//KeywordList/Keyword")

    @staticmethod
    def _extract_mesh(medline: ET.Element) -> List[Dict[str, Any]]:
        """
        Extract MeSH headings.
        Returns list of dicts with keys: descriptor, qualifier, is_major_topic.
        MeSH qualifier terms attached to a descriptor are returned as separate rows.
        """
        mesh_entries = []
        for heading_el in medline.findall(".//MeshHeadingList/MeshHeading"):
            descriptor_el = heading_el.find("DescriptorName")
            if descriptor_el is None:
                continue
            descriptor = (descriptor_el.text or "").strip()
            desc_major = descriptor_el.get("MajorTopicYN", "N").upper() == "Y"

            qualifiers = heading_el.findall("QualifierName")
            if qualifiers:
                for qual_el in qualifiers:
                    qualifier = (qual_el.text or "").strip()
                    qual_major = qual_el.get("MajorTopicYN", "N").upper() == "Y"
                    mesh_entries.append(
                        {
                            "descriptor": descriptor,
                            "qualifier": qualifier,
                            "is_major_topic": desc_major or qual_major,
                        }
                    )
            else:
                mesh_entries.append(
                    {
                        "descriptor": descriptor,
                        "qualifier": None,
                        "is_major_topic": desc_major,
                    }
                )
        return mesh_entries

    @staticmethod
    def _extract_chemicals(medline: ET.Element) -> List[Dict[str, str]]:
        chemicals = []
        for chem_el in medline.findall(".//ChemicalList/Chemical"):
            name_el = chem_el.find("NameOfSubstance")
            reg_el = chem_el.find("RegistryNumber")
            if name_el is not None and (name_el.text or "").strip():
                chemicals.append(
                    {
                        "name": name_el.text.strip(),
                        "registry_number": (
                            reg_el.text.strip()
                            if reg_el is not None and reg_el.text
                            else None
                        ),
                    }
                )
        return chemicals

    @staticmethod
    def _extract_publication_types(article: ET.Element) -> List[str]:
        return _iter_text(article, ".//PublicationTypeList/PublicationType")

    @staticmethod
    def _extract_grants(article: ET.Element) -> List[Dict[str, str]]:
        grants = []
        for grant_el in article.findall(".//GrantList/Grant"):
            grants.append(
                {
                    "grant_id": _text(grant_el.find("GrantID")),
                    "acronym": _text(grant_el.find("Acronym")),
                    "agency": _text(grant_el.find("Agency")),
                    "country": _text(grant_el.find("Country")),
                }
            )
        return grants

    @staticmethod
    def _extract_pub_date(article: ET.Element) -> Optional[str]:
        for date_path in (
            ".//Journal/JournalIssue/PubDate",
            ".//PubMedPubDate[@PubStatus='pubmed']",
            ".//ArticleDate",
        ):
            date_el = article.find(date_path)
            if date_el is None:
                continue
            year = _text(date_el.find("Year"))
            month = _text(date_el.find("Month"))
            day = _text(date_el.find("Day"))
            medline_date = _text(date_el.find("MedlineDate"))
            if year:
                parts = [year]
                if month:
                    parts.append(month.zfill(2) if month.isdigit() else month[:3])
                if day:
                    parts.append(day.zfill(2))
                return "-".join(parts)
            if medline_date:
                return medline_date
        return None

    @staticmethod
    def _extract_doi(article: ET.Element) -> Optional[str]:
        for eid_el in article.findall(".//ELocationID"):
            if eid_el.get("EIdType") == "doi":
                return (eid_el.text or "").strip() or None
        for aid_el in article.findall(".//ArticleId"):
            if aid_el.get("IdType") == "doi":
                return (aid_el.text or "").strip() or None
        return None

    # ── Public interface ──────────────────────────────────────────────────────

    def parse_record(
        self, xml_record: ET.Element, query_used: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single <PubmedArticle> element.
        Returns a structured dict, or None if the element is malformed.
        """
        try:
            medline = xml_record.find("MedlineCitation")
            if medline is None:
                return None

            pmid_el = medline.find("PMID")
            pmid = (pmid_el.text or "").strip() if pmid_el is not None else None
            if not pmid:
                logger.warning("Skipping record with missing PMID")
                return None

            article = medline.find("Article")
            if article is None:
                logger.warning("PMID %s: no <Article> element", pmid)
                return None

            title = _text(article.find("ArticleTitle")) or _text(
                article.find(".//ArticleTitle")
            )
            if not title:
                logger.debug("PMID %s: missing title", pmid)

            abstract = self._extract_abstract(article)
            if not abstract:
                logger.debug("PMID %s: missing abstract", pmid)

            journal_el = article.find(".//Journal/Title")
            journal = _text(journal_el)

            volume = _text(article.find(".//JournalIssue/Volume"))
            issue = _text(article.find(".//JournalIssue/Issue"))

            pagination_el = article.find(".//Pagination/MedlinePgn")
            pages = _text(pagination_el)

            pub_date = self._extract_pub_date(article)

            doi = self._extract_doi(article) or self._extract_doi(
                xml_record.find("PubmedData") or ET.Element("_")
            )

            language_els = article.findall(".//Language")
            language = language_els[0].text.strip() if language_els else None

            authors = self._extract_authors(article)
            author_keywords = self._extract_author_keywords(medline)
            mesh = self._extract_mesh(medline)
            chemicals = self._extract_chemicals(medline)
            publication_types = self._extract_publication_types(article)
            grants = self._extract_grants(article)

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "author_keywords": author_keywords,
                "mesh_terms": mesh,
                "chemicals": chemicals,
                "publication_types": publication_types,
                "grants": grants,
                "journal": journal,
                "volume": volume,
                "issue": issue,
                "pages": pages,
                "pub_date": pub_date,
                "doi": doi,
                "language": language,
                "query_used": query_used,
            }
        except Exception as exc:
            logger.error("parse_record failed: %s", exc, exc_info=True)
            return None

    def parse_batch(
        self, xml_batch: str, query_used: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Parse a raw XML string containing one or more <PubmedArticle> elements.
        Returns a list of successfully parsed record dicts.
        """
        records: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_batch)
        except ET.ParseError as exc:
            logger.error("XML parse error on batch: %s", exc)
            return records

        articles = root.findall("PubmedArticle")
        if not articles:
            # Sometimes the root IS a PubmedArticleSet
            articles = root.findall(".//PubmedArticle")

        for article_el in articles:
            parsed = self.parse_record(article_el, query_used=query_used)
            if parsed:
                records.append(parsed)

        logger.debug(
            "parse_batch: %d articles found, %d parsed successfully",
            len(articles),
            len(records),
        )
        return records
