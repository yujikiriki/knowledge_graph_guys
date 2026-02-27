#!/usr/bin/env python3
"""
validate_skos_owl.py
────────────────────
Three tools in one for Turtle (.ttl) files containing SKOS and/or OWL.

  MODES
    validate  (default)  Run all validation checks and report results.
    lint                 Run style/quality checks on the raw text.
    format               Produce a canonically formatted copy of the file.

  VALIDATE — four layers
    LAYER 1 — Syntax
      S0  Turtle parses without error (rdflib).

    LAYER 2 — SKOS integrity (W3C SKOS Reference)
      S1  No concept has more than one skos:prefLabel per language tag.
      S2  skos:prefLabel and skos:altLabel are disjoint per lang per concept.
      S3  skos:prefLabel and skos:hiddenLabel are disjoint per lang per concept.
      S4  Every skos:broader has a corresponding skos:narrower inverse (and vice-versa).
      S5  No skos:broader cycle (concept is its own ancestor).
      S6  skos:topConceptOf ↔ skos:hasTopConcept are symmetric.
      S7  Every skos:Concept declares skos:inScheme at least once.
      S8  Every skos:inScheme target is a declared skos:ConceptScheme.

    LAYER 3 — OWL integrity
      O1  Every rdfs:subClassOf subject/object is declared as owl:Class or rdfs:Class.
      O2  Every owl:ObjectProperty / owl:DatatypeProperty has rdfs:domain and rdfs:range.
      O3  No resource is declared as both owl:Class and skos:Concept (layer mixing).

    LAYER 4 — Custom mi: property checks
      M1  Every mi:hasRange/hasPeriod/hasRegion/hasProduction value is a skos:Concept.
      M2  Every mi:hasSubScheme value is a declared skos:ConceptScheme.

  LINT — style and quality rules (text-level, no mutation)
      L1  URI local names follow conventions:
            owl:Class / rdfs:Class  → PascalCase
            owl:ObjectProperty / owl:DatatypeProperty / rdf:Property → camelCase
            skos:Concept / skos:ConceptScheme → kebab-case
      L2  Concepts/classes/properties missing skos:definition or rdfs:comment.
      L3  Declared prefixes that are never used in the body.
      L4  Lines exceeding 120 characters.
      L5  Mixed indentation (tabs and spaces in same file).
      L6  Trailing whitespace on any line.
      L7  File does not end with a single newline.

  FORMAT — canonical Turtle output
    • Sorted prefix block (alphabetical by prefix name).
    • Subject blocks sorted: ConceptSchemes → Concepts (by scheme, then alpha) →
      OWL Classes → Properties → everything else.
    • Predicate–object pairs sorted canonically within each subject block.
    • Consistent 2-space indentation; predicates aligned.
    • Blank line between every subject block.
    • Lines soft-wrapped at 100 characters where possible.
    • Output written to <input>_formatted.ttl unless --output is specified.
    • Never overwrites the input file unless --inplace is explicitly passed.

USAGE
    python3 validate_skos_owl.py <file.ttl>
    python3 validate_skos_owl.py <file.ttl> --strict
    python3 validate_skos_owl.py <file.ttl> --lint
    python3 validate_skos_owl.py <file.ttl> --format
    python3 validate_skos_owl.py <file.ttl> --format --output clean.ttl
    python3 validate_skos_owl.py <file.ttl> --format --inplace
    python3 validate_skos_owl.py <file.ttl> --all       # validate + lint + format

EXIT CODES
    0  — passed (warnings may be present unless --strict)
    1  — one or more errors / lint violations found
    2  — file not found or unreadable
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef, BNode
from rdflib.namespace import SKOS, XSD

# ── Namespaces ───────────────────────────────────────────────────────────────
MI = Namespace("http://mymusicalinstruments.org/")

FACET_PROPS = [
    MI.hasRange,
    MI.hasPeriod,
    MI.hasRegion,
    MI.hasProduction,
]

# Predicate sort order — lower = earlier in subject block
PRED_ORDER = {
    str(RDF.type):               0,
    str(RDFS.subClassOf):        1,
    str(RDFS.subPropertyOf):     2,
    str(RDFS.domain):            3,
    str(RDFS.range):             4,
    str(RDFS.label):             5,
    str(RDFS.comment):           6,
    str(SKOS.prefLabel):        10,
    str(SKOS.altLabel):         11,
    str(SKOS.hiddenLabel):      12,
    str(SKOS.definition):       13,
    str(SKOS.scopeNote):        14,
    str(SKOS.example):          15,
    str(SKOS.notation):         16,
    str(SKOS.topConceptOf):     20,
    str(SKOS.inScheme):         21,
    str(SKOS.hasTopConcept):    22,
    str(SKOS.broader):          30,
    str(SKOS.narrower):         31,
    str(SKOS.related):          32,
    str(SKOS.exactMatch):       40,
    str(SKOS.closeMatch):       41,
    str(SKOS.broadMatch):       42,
    str(SKOS.narrowMatch):      43,
    str(SKOS.relatedMatch):     44,
}


# ── Result collector ─────────────────────────────────────────────────────────
class Results:
    def __init__(self):
        self.errors   = []
        self.warnings = []
        self.lint     = []

    def err(self, code, msg):
        self.errors.append((code, msg))

    def warn(self, code, msg):
        self.warnings.append((code, msg))

    def lint_issue(self, code, line_no, msg):
        self.lint.append((code, line_no, msg))

    def has_errors(self, strict=False):
        return bool(self.errors) or (strict and bool(self.warnings))


# ── URI label helper ─────────────────────────────────────────────────────────
def short(uri, g):
    """Return a prefixed name or angle-bracket URI."""
    if isinstance(uri, BNode):
        return f"_:{uri}"
    s = str(uri)
    for ns, prefix in g.namespaces():
        ns_str = str(ns)
        if s.startswith(ns_str) and prefix:
            return f"{prefix}:{s[len(ns_str):]}"
    return f"<{uri}>"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Syntax
# ══════════════════════════════════════════════════════════════════════════════
def check_syntax(path, r):
    g = Graph()
    try:
        g.parse(str(path), format="turtle")
        print(f"  ✓ Syntax OK — {len(g)} triples loaded.")
        return g
    except Exception as e:
        r.err("S0", f"Turtle parse failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — SKOS integrity
# ══════════════════════════════════════════════════════════════════════════════
def check_skos(g, r):
    concepts = set(g.subjects(RDF.type, SKOS.Concept))
    schemes  = set(g.subjects(RDF.type, SKOS.ConceptScheme))

    # S1 — at most one prefLabel per language per concept
    for c in concepts:
        by_lang = defaultdict(list)
        for _, _, v in g.triples((c, SKOS.prefLabel, None)):
            if isinstance(v, Literal):
                by_lang[v.language].append(str(v))
        for lang, vals in by_lang.items():
            if len(vals) > 1:
                r.err("S1", f"Multiple skos:prefLabel @{lang} on {short(c, g)}: {vals}")

    # S2 — prefLabel ∩ altLabel disjoint per lang
    for c in concepts:
        pref = defaultdict(set)
        alt  = defaultdict(set)
        for _, _, v in g.triples((c, SKOS.prefLabel, None)):
            if isinstance(v, Literal): pref[v.language].add(str(v))
        for _, _, v in g.triples((c, SKOS.altLabel, None)):
            if isinstance(v, Literal): alt[v.language].add(str(v))
        for lang in pref:
            clash = pref[lang] & alt.get(lang, set())
            if clash:
                r.err("S2", f"prefLabel/altLabel clash @{lang} on {short(c, g)}: {clash}")

    # S3 — prefLabel ∩ hiddenLabel disjoint per lang
    for c in concepts:
        pref   = defaultdict(set)
        hidden = defaultdict(set)
        for _, _, v in g.triples((c, SKOS.prefLabel, None)):
            if isinstance(v, Literal): pref[v.language].add(str(v))
        for _, _, v in g.triples((c, SKOS.hiddenLabel, None)):
            if isinstance(v, Literal): hidden[v.language].add(str(v))
        for lang in pref:
            clash = pref[lang] & hidden.get(lang, set())
            if clash:
                r.err("S3", f"prefLabel/hiddenLabel clash @{lang} on {short(c, g)}: {clash}")

    # S4 — broader/narrower must be symmetric
    for c, _, b in g.triples((None, SKOS.broader, None)):
        if (b, SKOS.narrower, c) not in g:
            r.warn("S4", f"Missing skos:narrower inverse: {short(b,g)} → {short(c,g)}")
    for c, _, n in g.triples((None, SKOS.narrower, None)):
        if (n, SKOS.broader, c) not in g:
            r.warn("S4", f"Missing skos:broader inverse: {short(n,g)} → {short(c,g)}")

    # S5 — no broader cycle
    def ancestors(node, visited=None):
        if visited is None:
            visited = set()
        for _, _, b in g.triples((node, SKOS.broader, None)):
            if b in visited:
                return visited
            visited.add(b)
            ancestors(b, visited)
        return visited

    for c in concepts:
        if c in ancestors(c):
            r.err("S5", f"skos:broader cycle detected involving {short(c, g)}")

    # S6 — topConceptOf ↔ hasTopConcept symmetry
    for c, _, s in g.triples((None, SKOS.topConceptOf, None)):
        if (s, SKOS.hasTopConcept, c) not in g:
            r.warn("S6", f"Missing skos:hasTopConcept on scheme {short(s,g)} for {short(c,g)}")
    for s, _, c in g.triples((None, SKOS.hasTopConcept, None)):
        if (c, SKOS.topConceptOf, s) not in g:
            r.warn("S6", f"Missing skos:topConceptOf on concept {short(c,g)} for scheme {short(s,g)}")

    # S7 — every Concept has inScheme
    for c in concepts:
        if not list(g.objects(c, SKOS.inScheme)):
            r.err("S7", f"Concept missing skos:inScheme: {short(c, g)}")

    # S8 — every inScheme target is a declared ConceptScheme
    for _, _, s in g.triples((None, SKOS.inScheme, None)):
        if (s, RDF.type, SKOS.ConceptScheme) not in g:
            r.err("S8", f"skos:inScheme target not declared as ConceptScheme: {short(s, g)}")

    print(f"  ✓ SKOS — {len(concepts)} concepts, {len(schemes)} schemes.")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — OWL integrity
# ══════════════════════════════════════════════════════════════════════════════
def check_owl(g, r):
    owl_classes = (
        set(g.subjects(RDF.type, OWL.Class)) |
        set(g.subjects(RDF.type, RDFS.Class))
    )

    # O1 — subClassOf participants should be declared owl:Class
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and s not in owl_classes:
            r.warn("O1", f"subClassOf subject not declared as owl:Class: {short(s, g)}")
        if isinstance(o, URIRef) and o not in owl_classes:
            r.warn("O1", f"subClassOf object not declared as owl:Class: {short(o, g)}")

    # O2 — ObjectProperty and DatatypeProperty need domain + range
    for prop_type in [OWL.ObjectProperty, OWL.DatatypeProperty]:
        type_name = str(prop_type).split("#")[1]
        for p in g.subjects(RDF.type, prop_type):
            if not list(g.objects(p, RDFS.domain)):
                r.err("O2", f"{type_name} missing rdfs:domain: {short(p, g)}")
            if not list(g.objects(p, RDFS.range)):
                r.err("O2", f"{type_name} missing rdfs:range: {short(p, g)}")

    # O3 — warn on layer mixing
    concepts = set(g.subjects(RDF.type, SKOS.Concept))
    for c in concepts & owl_classes:
        r.warn("O3", f"Resource is both owl:Class and skos:Concept: {short(c, g)}")

    obj_props  = list(g.subjects(RDF.type, OWL.ObjectProperty))
    data_props = list(g.subjects(RDF.type, OWL.DatatypeProperty))
    print(f"  ✓ OWL  — {len(owl_classes)} classes, "
          f"{len(obj_props)} ObjectProperties, {len(data_props)} DatatypeProperties.")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Custom mi: properties
# ══════════════════════════════════════════════════════════════════════════════
def check_mi(g, r):
    concepts = set(g.subjects(RDF.type, SKOS.Concept))
    schemes  = set(g.subjects(RDF.type, SKOS.ConceptScheme))

    # M1 — facet values must be skos:Concepts
    for prop in FACET_PROPS:
        prop_name = str(prop).split("/")[-1]
        for s, _, o in g.triples((None, prop, None)):
            if o not in concepts:
                r.err("M1", f"mi:{prop_name} value not a skos:Concept: "
                            f"{short(s, g)} → {short(o, g)}")

    # M2 — hasSubScheme targets must be ConceptSchemes
    for s, _, o in g.triples((None, MI.hasSubScheme, None)):
        if o not in schemes:
            r.err("M2", f"mi:hasSubScheme target not a ConceptScheme: "
                        f"{short(s, g)} → {short(o, g)}")

    fa = sum(sum(1 for _ in g.triples((None, p, None))) for p in FACET_PROPS)
    ss = sum(1 for _ in g.triples((None, MI.hasSubScheme, None)))
    print(f"  ✓ mi:  — {fa} facet assignments, {ss} hasSubScheme links.")


# ══════════════════════════════════════════════════════════════════════════════
# LINT — text-level style checks
# ══════════════════════════════════════════════════════════════════════════════
_KEBAB_RE   = re.compile(r'^[a-z][a-z0-9]*(-[a-z0-9]+)*$')
_PASCAL_RE  = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
_CAMEL_RE   = re.compile(r'^[a-z][a-zA-Z0-9]*$')

def _local(uri):
    """Extract the local name from a URI string."""
    s = str(uri)
    for sep in ('#', '/'):
        idx = s.rfind(sep)
        if idx >= 0:
            return s[idx+1:]
    return s

def run_lint(path, g, r):
    text  = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # ── L1: naming conventions ───────────────────────────────────────────────
    # owl:Class / rdfs:Class → PascalCase local names
    for cls in (set(g.subjects(RDF.type, OWL.Class)) |
                set(g.subjects(RDF.type, RDFS.Class))):
        if isinstance(cls, URIRef):
            loc = _local(cls)
            if loc and not _PASCAL_RE.match(loc):
                r.lint_issue("L1", None,
                    f"owl:Class local name should be PascalCase: '{loc}' in {short(cls, g)}")

    # owl:ObjectProperty / owl:DatatypeProperty / rdf:Property → camelCase
    for prop_type in [OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property]:
        for p in g.subjects(RDF.type, prop_type):
            if isinstance(p, URIRef):
                loc = _local(p)
                if loc and not _CAMEL_RE.match(loc):
                    r.lint_issue("L1", None,
                        f"Property local name should be camelCase: '{loc}' in {short(p, g)}")

    # skos:Concept / skos:ConceptScheme → kebab-case
    for node_type in [SKOS.Concept, SKOS.ConceptScheme]:
        for c in g.subjects(RDF.type, node_type):
            if isinstance(c, URIRef):
                loc = _local(c)
                if loc and not _KEBAB_RE.match(loc):
                    r.lint_issue("L1", None,
                        f"SKOS local name should be kebab-case: '{loc}' in {short(c, g)}")

    # ── L2: undocumented terms ───────────────────────────────────────────────
    all_named = (
        set(g.subjects(RDF.type, OWL.Class)) |
        set(g.subjects(RDF.type, SKOS.Concept)) |
        set(g.subjects(RDF.type, SKOS.ConceptScheme)) |
        set(g.subjects(RDF.type, OWL.ObjectProperty)) |
        set(g.subjects(RDF.type, OWL.DatatypeProperty)) |
        set(g.subjects(RDF.type, RDF.Property))
    )
    for node in all_named:
        if isinstance(node, URIRef):
            has_def = (
                list(g.objects(node, SKOS.definition)) or
                list(g.objects(node, RDFS.comment))
            )
            if not has_def:
                r.lint_issue("L2", None,
                    f"No skos:definition or rdfs:comment: {short(node, g)}")

    # ── L3: declared but unused prefixes ────────────────────────────────────
    declared_prefixes = {}
    for line in lines:
        m = re.match(r'@prefix\s+(\w*):\s+<([^>]+)>', line.strip())
        if m:
            declared_prefixes[m.group(1)] = m.group(2)

    body = "\n".join(
        l for l in lines
        if not l.strip().startswith("@prefix")
    )
    for prefix, ns in declared_prefixes.items():
        pattern = rf'\b{re.escape(prefix)}:'
        if not re.search(pattern, body):
            r.lint_issue("L3", None, f"Prefix declared but never used: '{prefix}:'")

    # ── L4: lines > 120 chars ────────────────────────────────────────────────
    for i, line in enumerate(lines, 1):
        if len(line) > 120:
            r.lint_issue("L4", i,
                f"Line {i} is {len(line)} chars (limit 120): {line[:60]}…")

    # ── L5: mixed indentation ────────────────────────────────────────────────
    has_tabs    = any('\t' in l for l in lines)
    has_spaces  = any(l.startswith('  ') for l in lines)
    if has_tabs and has_spaces:
        r.lint_issue("L5", None, "File mixes tab and space indentation.")
    elif has_tabs:
        r.lint_issue("L5", None, "File uses tab indentation — prefer spaces.")

    # ── L6: trailing whitespace ──────────────────────────────────────────────
    for i, line in enumerate(lines, 1):
        if line != line.rstrip():
            r.lint_issue("L6", i, f"Trailing whitespace on line {i}.")

    # ── L7: file must end with exactly one newline ───────────────────────────
    if not text.endswith("\n"):
        r.lint_issue("L7", len(lines), "File does not end with a newline.")
    elif text.endswith("\n\n"):
        r.lint_issue("L7", len(lines), "File ends with multiple trailing newlines.")

    n = len(r.lint)
    print(f"  {'✓' if n == 0 else '✗'} Lint — {n} issue(s) found.")


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTER — canonical Turtle output
# ══════════════════════════════════════════════════════════════════════════════
def _term_str(term, g, prefixes):
    """Serialize a single RDF term to its shortest Turtle representation."""
    if isinstance(term, BNode):
        return f"_:{term}"
    if isinstance(term, Literal):
        val = str(term)
        # Escape internal double-quotes
        escaped = val.replace('\\', '\\\\').replace('"', '\\"')
        if term.language:
            # Use triple-quote for multi-line literals
            if "\n" in val:
                return f'"""{val}"""@{term.language}'
            return f'"{escaped}"@{term.language}'
        if term.datatype and term.datatype != XSD.string:
            dt = _uri_str(term.datatype, prefixes)
            return f'"{escaped}"^^{dt}'
        return f'"{escaped}"'
    # URIRef
    return _uri_str(term, prefixes)

def _uri_str(uri, prefixes):
    """Return prefixed name if possible, else <full-uri>."""
    s = str(uri)
    best_prefix = None
    best_ns_len = 0
    for prefix, ns in prefixes.items():
        ns_s = str(ns)
        if s.startswith(ns_s) and len(ns_s) > best_ns_len:
            best_ns_len = len(ns_s)
            best_prefix = prefix
    if best_prefix is not None:
        local = s[best_ns_len:]
        # Only use prefixed form if local name is a valid PN_LOCAL
        if re.match(r'^[a-zA-Z_\u00C0-\u00D6][a-zA-Z0-9_\-\.]*$', local or ''):
            return f"{best_prefix}:{local}"
    return f"<{s}>"

def _pred_sort_key(pred_str):
    return (PRED_ORDER.get(pred_str, 99), pred_str)

def _subject_sort_key(subj, g):
    """
    Sort order for subjects:
      0 — rdf:Property declarations (bridge properties)
      1 — skos:ConceptScheme
      2 — skos:Concept (top concepts first, then by scheme, then alpha)
      3 — owl:Class / rdfs:Class
      4 — owl:ObjectProperty / owl:DatatypeProperty
      9 — everything else
    """
    types = set(g.objects(subj, RDF.type))
    s     = str(subj)

    if RDF.Property in types:
        return (0, s)
    if SKOS.ConceptScheme in types:
        return (1, s)
    if SKOS.Concept in types:
        # sub-sort: top concepts (have topConceptOf) before others
        is_top = bool(list(g.objects(subj, SKOS.topConceptOf)))
        scheme = str(next(g.objects(subj, SKOS.inScheme), ""))
        return (2, scheme, 0 if is_top else 1, s)
    if OWL.Class in types or RDFS.Class in types:
        return (3, s)
    if OWL.ObjectProperty in types or OWL.DatatypeProperty in types:
        return (4, s)
    return (9, s)

def format_ttl(path, g, output_path):
    """Write a canonically formatted Turtle file to output_path."""
    INDENT  = "  "
    WRAP    = 100

    # Build prefix map from graph (sorted alphabetically)
    prefixes = dict(sorted(g.namespaces(), key=lambda x: x[0]))

    lines = []

    # ── Prefix block ─────────────────────────────────────────────────────────
    for prefix, ns in prefixes.items():
        if prefix:  # skip empty prefix
            lines.append(f"@prefix {prefix}:{' ' * max(1, 12 - len(prefix))}<{ns}> .")
    # Always emit base prefix last if present
    empty = prefixes.get("")
    if empty:
        lines.append(f"@base <{empty}> .")
    lines.append("")

    # ── Collect all subjects (excluding BNodes) ───────────────────────────────
    subjects = sorted(
        [s for s in set(g.subjects()) if isinstance(s, URIRef)],
        key=lambda s: _subject_sort_key(s, g)
    )

    for subj in subjects:
        subj_str = _uri_str(subj, prefixes)
        pred_obj = defaultdict(list)
        for _, p, o in g.triples((subj, None, None)):
            pred_obj[str(p)].append(o)

        if not pred_obj:
            continue

        # Sort predicates
        sorted_preds = sorted(pred_obj.keys(), key=_pred_sort_key)

        block = []
        block.append(subj_str)

        for i, pred_s in enumerate(sorted_preds):
            pred_uri  = URIRef(pred_s)
            pred_repr = _uri_str(pred_uri, prefixes)
            objects   = sorted(pred_obj[pred_s], key=lambda o: str(o))
            is_last_pred = (i == len(sorted_preds) - 1)

            for j, obj in enumerate(objects):
                is_last_obj = (j == len(objects) - 1)
                obj_repr    = _term_str(obj, g, prefixes)

                if i == 0 and j == 0:
                    # First predicate on same line as subject if short enough
                    candidate = f"{subj_str} {pred_repr} {obj_repr}"
                    if len(candidate) <= WRAP:
                        block[-1] = candidate
                    else:
                        block[-1] = subj_str
                        block.append(f"{INDENT}{pred_repr} {obj_repr}")
                elif j == 0:
                    block.append(f"{INDENT}{pred_repr} {obj_repr}")
                else:
                    block.append(f"{INDENT}{' ' * (len(pred_repr) + 1)}{obj_repr}")

                # Terminator
                if is_last_pred and is_last_obj:
                    block[-1] += " ."
                elif is_last_obj:
                    block[-1] += " ;"
                else:
                    block[-1] += " ,"

        lines.extend(block)
        lines.append("")  # blank line between subjects

    # Remove trailing blank line and ensure single final newline
    while lines and lines[-1] == "":
        lines.pop()
    output = "\n".join(lines) + "\n"

    output_path.write_text(output, encoding="utf-8")
    triple_count = len(g)
    subject_count = len(subjects)
    print(f"  ✓ Formatted — {subject_count} subjects, {triple_count} triples → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
def print_report(r, strict=False):
    print(f"\n{'─'*60}")

    if r.warnings:
        print(f"\n  WARNINGS ({len(r.warnings)}):")
        for code, msg in r.warnings:
            print(f"    [WARN {code}] {msg}")

    if r.errors:
        print(f"\n  ERRORS ({len(r.errors)}):")
        for code, msg in r.errors:
            print(f"    [ERROR {code}] {msg}")

    if r.lint:
        print(f"\n  LINT ({len(r.lint)}):")
        for code, line_no, msg in r.lint:
            loc = f"line {line_no}: " if line_no else ""
            print(f"    [LINT {code}] {loc}{msg}")

    print(f"\n{'═'*60}")
    total_errors = len(r.errors) + (len(r.warnings) if strict else 0) + len(r.lint)
    label = "PASSED" if total_errors == 0 else "FAILED"
    mark  = "✓" if total_errors == 0 else "✗"
    print(f"  {mark} {label} — "
          f"{len(r.errors)} error(s), "
          f"{len(r.warnings)} warning(s), "
          f"{len(r.lint)} lint issue(s).")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Validate, lint, and format SKOS/OWL Turtle files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[1] if "USAGE" in __doc__ else ""
    )
    parser.add_argument("file",
        help="Path to the .ttl file.")
    parser.add_argument("--strict", action="store_true",
        help="Treat warnings as errors.")
    parser.add_argument("--lint", action="store_true",
        help="Run linter (in addition to validation).")
    parser.add_argument("--format", action="store_true",
        help="Write formatted output.")
    parser.add_argument("--all", action="store_true",
        help="Run validate + lint + format.")
    parser.add_argument("--output", default=None,
        help="Output path for --format (default: <input>_formatted.ttl).")
    parser.add_argument("--inplace", action="store_true",
        help="Overwrite input file when formatting (use with caution).")
    parser.add_argument("--no-validate", action="store_true",
        help="Skip validation (only lint or format).")

    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(2)

    do_validate = not args.no_validate
    do_lint     = args.lint or args.all
    do_format   = args.format or args.all

    r = Results()

    print(f"\n{'═'*60}")
    print(f"  File: {path}")
    print(f"{'═'*60}")

    # ── Always parse first ────────────────────────────────────────────────────
    print("\n[LAYER 1] Syntax")
    g = check_syntax(path, r)
    if g is None:
        print_report(r, args.strict)
        sys.exit(1)

    # ── Validate ──────────────────────────────────────────────────────────────
    if do_validate:
        print("\n[LAYER 2] SKOS integrity")
        check_skos(g, r)

        print("\n[LAYER 3] OWL integrity")
        check_owl(g, r)

        print("\n[LAYER 4] mi: custom properties")
        check_mi(g, r)

    # ── Lint ──────────────────────────────────────────────────────────────────
    if do_lint:
        print("\n[LAYER 5] Lint")
        run_lint(path, g, r)

    # ── Format ────────────────────────────────────────────────────────────────
    if do_format:
        print("\n[LAYER 6] Format")
        if args.inplace:
            out_path = path
        elif args.output:
            out_path = Path(args.output)
        else:
            out_path = path.with_stem(path.stem + "_formatted")
        format_ttl(path, g, out_path)

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(r, args.strict)

    sys.exit(0 if not r.has_errors(args.strict) else 1)


if __name__ == "__main__":
    main()
