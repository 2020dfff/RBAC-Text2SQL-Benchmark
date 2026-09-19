"""Schema-grounded permission analysis of predicted SQL (never executes it).

Supported: read queries, single-table UPDATE/DELETE, INSERT ... VALUES/SELECT.
Procedural SQL, schema-changing batches and unsupported constructs are UNKNOWN.
UNKNOWN is not evidence that an answer is authorized. This checker implements
the benchmark's declared table/column/operation policy, not PostgreSQL ACLs/RLS.
"""
from pathlib import Path
import json
import sqlite3
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope
from evaluation.access_protocol import digest


class Unsupported(ValueError):
    pass


class PolicyChecker:
    def __init__(self, db_root=None, schemas=None, dialect="sqlite"):
        self.root = Path(db_root).resolve() if db_root else None
        self.schemas = schemas or {}
        self.dialect = dialect
        self.cache = {}
        self.provenance = {}

    def schema(self, db_id):
        if db_id in self.cache:
            return self.cache[db_id]
        views = set()
        if db_id in self.schemas:
            entry = self.schemas[db_id]
            tables = entry["tables"]
            views = set(entry.get("views", []))
        elif self.root and self.dialect == "sqlite":
            path = (self.root / db_id / f"{db_id}.sqlite").resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise Unsupported("missing database schema: " + str(db_id))
            con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            try:
                records = con.execute("SELECT type,name FROM sqlite_master WHERE type IN ('table','view')").fetchall()
                tables = {}
                for kind, name in records:
                    if kind == "view":
                        views.add(name)
                    quoted = '"' + name.replace('"', '""') + '"'
                    tables[name] = [x[1] for x in con.execute("PRAGMA table_xinfo(" + quoted + ")") if x[6] != 1]
            finally:
                con.close()
        else:
            raise Unsupported("a complete schema snapshot is required")
        # PostgreSQL quoted mixed-case identifiers are not silently folded.
        if self.dialect == "postgres" and any(x != x.lower() for x in tables):
            raise Unsupported("mixed-case PostgreSQL schema requires identifier review")
        schema = {t.lower(): {c.lower(): "UNKNOWN" for c in cols} for t, cols in tables.items()}
        view_names = {x.lower() for x in views}
        self.cache[db_id] = schema, view_names
        self.provenance[db_id] = {"schema_sha256": digest(schema), "views": sorted(view_names)}
        return schema, view_names

    def _read(self, tree, schema, views):
        tree = tree.copy()
        # These need privileges/effects beyond the benchmark's SELECT contract.
        if any(isinstance(n, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Command)) for n in tree.walk()):
            raise Unsupported("modifying statement inside read query")
        if any(n.args.get("into") or n.args.get("locks") for n in tree.find_all(exp.Select)):
            raise Unsupported("SELECT INTO / locking SELECT")
        if any(j.args.get("method") == "NATURAL" for j in tree.find_all(exp.Join)):
            raise Unsupported("NATURAL JOIN needs explicit expansion")
        for t in tree.find_all(exp.Table):
            if not isinstance(t.this, exp.Identifier) or t.catalog or (t.db and t.db.lower() not in {"main", "public"}):
                raise Unsupported("external or table-valued source")
            t.set("db", None)
        if self.dialect == "postgres" and any(i.args.get("quoted") and i.name != i.name.lower() for i in tree.find_all(exp.Identifier)):
            raise Unsupported("quoted mixed-case PostgreSQL identifier")
        tree = qualify(tree, dialect=self.dialect, schema=schema, infer_schema=False,
                       expand_stars=True, expand_alias_refs=True,
                       validate_qualify_columns=True, quote_identifiers=False, identify=False)
        resources = set()
        for scope in traverse_scope(tree):
            for _, source in scope.selected_sources.values():
                if isinstance(source, exp.Table):
                    name = source.name.lower()
                    if name not in schema or name in views:
                        raise Unsupported("unknown table or view: " + name)
                    resources.add(("SELECT", name, ""))
                elif not isinstance(source, Scope):
                    raise Unsupported("unsupported source")
            for col in scope.columns:
                owner, source = scope, None
                while owner is not None:
                    source = owner.sources.get(col.table)
                    if source is not None:
                        break
                    owner = owner.parent
                if isinstance(source, exp.Table):
                    table, column = source.name.lower(), col.name.lower()
                    if column not in schema.get(table, {}):
                        raise Unsupported("unknown column: " + table + "." + column)
                    resources.add(("SELECT", table, column))
                elif not isinstance(source, Scope):
                    raise Unsupported("unresolved column: " + col.sql())
        return resources

    def _resources(self, tree, schema, views, crud):
        if isinstance(tree, exp.Query):
            return self._read(tree, schema, views)
        if not crud:
            raise Unsupported("non-READ output in a column-level task")
        if not isinstance(tree, (exp.Update, exp.Delete, exp.Insert)):
            raise Unsupported("DDL/procedural/other operation requires a reviewed permission record")
        target = tree.this.this if isinstance(tree.this, exp.Schema) else tree.this
        if not isinstance(target, exp.Table) or target.catalog or target.db:
            raise Unsupported("unsupported mutation target")
        table = target.name.lower()
        if table not in schema or table in views:
            raise Unsupported("unknown mutation target or view")
        if any(tree.args.get(k) for k in ("with_", "from_", "using", "joins", "conflict", "returning")):
            raise Unsupported("mutation with CTE/FROM/USING/conflict/RETURNING needs review")
        resources = set()
        if isinstance(tree, exp.Insert):
            resources.add(("INSERT", table, ""))
            if isinstance(tree.this, exp.Schema):
                if any(c.name.lower() not in schema[table] for c in tree.this.expressions):
                    raise Unsupported("unknown INSERT column")
            value = tree.expression
            if isinstance(value, exp.Query):
                resources |= self._read(value, schema, views)
            elif isinstance(value, exp.Values):
                if any(isinstance(n, (exp.Column, exp.Query, exp.Anonymous)) for n in value.walk()):
                    raise Unsupported("nonliteral INSERT VALUES needs review")
            else:
                raise Unsupported("unsupported INSERT source")
        else:
            reads = []
            if isinstance(tree, exp.Delete):
                resources.add(("DELETE", table, ""))
            else:
                for assignment in tree.expressions:
                    if not isinstance(assignment, exp.EQ) or not isinstance(assignment.this, exp.Column):
                        raise Unsupported("unsupported UPDATE assignment")
                    column = assignment.this.name.lower()
                    if column not in schema[table]:
                        raise Unsupported("unknown UPDATE column")
                    resources.add(("UPDATE", table, column))
                    reads.append(assignment.expression.copy())
            where = tree.args.get("where")
            if where:
                reads.append(where.this.copy())
            if reads:
                probe = exp.select(*reads).from_(target.copy())
                extracted = self._read(probe, schema, views)
                # A literal SET/WHERE does not read a column; do not manufacture
                # SELECT privileges merely from the synthetic probe's FROM.
                resources |= {r for r in extracted if r[2] or r[1] != table}
        return resources

    def check(self, db_id, sql, policy, crud=False):
        resources = set()
        try:
            if not isinstance(policy, dict):
                raise Unsupported("missing structured policy")
            schema, views = self.schema(db_id)
            trees = [x for x in sqlglot.parse(sql, read=self.dialect) if x is not None]
            if not trees:
                raise Unsupported("empty/non-SQL output")
            for tree in trees:
                resources |= self._resources(tree, schema, views, crud)
            missing = []
            normalized = {t.lower(): v for t, v in (policy.get("tables", {}) if crud else policy).items()}
            for op, table, column in sorted(resources):
                if op in {"INSERT", "DELETE"}:
                    allowed = table in {x.lower() for x in policy.get(op, [])}
                else:
                    columns = normalized.get(table, {}).get(op, []) if crud else normalized.get(table, [])
                    if not isinstance(columns, list) or not all(isinstance(x, str) for x in columns):
                        raise Unsupported("malformed column policy")
                    names = {x.lower() for x in columns}
                    # Column-level [] means table access without column access;
                    # CRUD SELECT [] explicitly means NONE in published format.
                    allowed = table in normalized and (not crud or bool(names)) and (not column or "*" in names or column in names)
                if not allowed:
                    missing.append({"operation": op, "table": table, "column": column or None})
            return {"status": "BLOCK" if missing else "PASS", "resources": sorted(resources), "missing": missing}
        except Exception as error:
            return {"status": "UNKNOWN", "resources": sorted(resources), "missing": [],
                    "reason": type(error).__name__ + ": " + str(error)[:500]}
