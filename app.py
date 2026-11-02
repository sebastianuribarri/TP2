# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°2 - Sistemas Inteligentes
Agente Inteligente de Gestión Financiera con LangChain + LangGraph + Gradio

Incluye:
  1. Base SQLite con 11 tablas relacionadas (datos sintéticos reproducibles).
  2. Agente ReAct (tool calling) con memoria conversacional.
  3. Módulo de evaluación con LangChain: compara varios modelos en una batería de
     pruebas usando (a) verificación programática contra el resultado real de SQL y
     (b) un LLM-juez (LCEL + structured output) + métricas de latencia/tokens/herramientas.
  4. Interfaz Gradio (chat, comparador, explorador de la base).

Uso:
    pip install -r requirements.txt
    export GROQ_API_KEY=...        (o ingresarla en la UI)
    python app.py
"""
from __future__ import annotations

import ast
import math
import operator
import os
import random
import re
import sqlite3
import sys
import time
import unicodedata
import uuid
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import gradio as gr
import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
SQLITE_PATH = Path("gastos_empresa.db").resolve()
CHAT_MEMORY_PATH = Path("memoria_conversacion.sqlite").resolve()
RESULTS_CSV = Path("resultados_comparacion.csv").resolve()
MAX_ROWS = 50

DEFAULT_GROQ_MODELS = [

    "qwen/qwen3.8-27b"
]
DEFAULT_OLLAMA_MODELS = ["qwen3:8b", "llama3.1:8b"]
DEFAULT_JUDGE = "qwen/qwen3.8-27b"


# ---------------------------------------------------------------------------
# 1) Base de datos: 11 tablas relacionadas
# ---------------------------------------------------------------------------
DDL = """
CREATE TABLE departamentos(
    id INTEGER PRIMARY KEY, nombre TEXT NOT NULL, ciudad TEXT
);
CREATE TABLE empleados(
    id INTEGER PRIMARY KEY, nombre TEXT NOT NULL,
    departamento_id INTEGER REFERENCES departamentos(id),
    cargo TEXT, salario REAL, fecha_ingreso TEXT
);
CREATE TABLE categorias(
    id INTEGER PRIMARY KEY, nombre TEXT NOT NULL, tipo TEXT
);
CREATE TABLE proveedores(
    id INTEGER PRIMARY KEY, nombre TEXT NOT NULL, rubro TEXT, ciudad TEXT
);
CREATE TABLE proyectos(
    id INTEGER PRIMARY KEY, nombre TEXT NOT NULL,
    departamento_id INTEGER REFERENCES departamentos(id),
    presupuesto_total REAL, fecha_inicio TEXT, estado TEXT
);
CREATE TABLE clientes(
    id INTEGER PRIMARY KEY, nombre TEXT NOT NULL, sector TEXT, ciudad TEXT
);
CREATE TABLE metodos_pago(
    id INTEGER PRIMARY KEY, nombre TEXT NOT NULL
);
CREATE TABLE gastos(
    id INTEGER PRIMARY KEY,
    categoria TEXT,
    monto REAL,
    fecha TEXT,
    categoria_id INTEGER REFERENCES categorias(id),
    departamento_id INTEGER REFERENCES departamentos(id),
    empleado_id INTEGER REFERENCES empleados(id),
    proveedor_id INTEGER REFERENCES proveedores(id),
    proyecto_id INTEGER REFERENCES proyectos(id),
    metodo_pago_id INTEGER REFERENCES metodos_pago(id),
    descripcion TEXT
);
CREATE TABLE ingresos(
    id INTEGER PRIMARY KEY,
    cliente_id INTEGER REFERENCES clientes(id),
    proyecto_id INTEGER REFERENCES proyectos(id),
    monto REAL, fecha TEXT, concepto TEXT
);
CREATE TABLE facturas(
    id INTEGER PRIMARY KEY,
    cliente_id INTEGER REFERENCES clientes(id),
    monto REAL, fecha_emision TEXT, fecha_vencimiento TEXT,
    estado TEXT  -- pagada | pendiente | vencida
);
CREATE TABLE presupuestos(
    id INTEGER PRIMARY KEY,
    departamento_id INTEGER REFERENCES departamentos(id),
    categoria_id INTEGER REFERENCES categorias(id),
    anio INTEGER, mes INTEGER, monto_presupuestado REAL
);
"""

EXPECTED_TABLES = 11

def build_database(force: bool = False, path: Path = SQLITE_PATH) -> str:
    """Crea y puebla la base (determinística, seed fija). Devuelve un mensaje de estado."""
    if path.exists() and not force:
        with sqlite3.connect(path) as con:
            n = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
        if n >= EXPECTED_TABLES:
            return f"Base existente reutilizada ({n} tablas)."
    if path.exists():
        path.unlink()

    rng = random.Random(42)
    departamentos = [
        ("Tecnologia", "Rosario"), ("Ventas", "Buenos Aires"), ("Marketing", "Cordoba"),
        ("Recursos Humanos", "Rosario"), ("Operaciones", "Santa Fe"), ("Finanzas", "Buenos Aires"),
    ]
    categorias = [
        ("comida", "operativo", (800, 6000)), ("transporte", "operativo", (500, 4000)),
        ("servicios", "operativo", (3000, 15000)), ("equipamiento", "capital", (15000, 90000)),
        ("marketing", "comercial", (5000, 40000)), ("capacitacion", "operativo", (4000, 25000)),
        ("mantenimiento", "operativo", (2000, 20000)), ("viajes", "comercial", (6000, 50000)),
    ]
    proveedores = [
        ("Distribuidora Litoral", "alimentos", "Rosario"), ("TransRos SA", "logistica", "Rosario"),
        ("TecnoPlus", "hardware", "Buenos Aires"), ("EnergiaSur", "servicios", "Santa Fe"),
        ("PubliMax", "publicidad", "Cordoba"), ("Capacitar SRL", "formacion", "Rosario"),
        ("FixIt Mantenimiento", "mantenimiento", "Rosario"), ("ViajesYa", "turismo", "Buenos Aires"),
        ("Oficina Total", "insumos", "Rosario"), ("CloudNet", "software", "Buenos Aires"),
    ]
    proyectos = [
        ("Portal Ciudadano", "Tecnologia", 900000, "2026-01-15", "en curso"),
        ("Campana Verano", "Marketing", 350000, "2026-02-01", "finalizado"),
        ("Migracion Cloud", "Tecnologia", 1200000, "2026-03-10", "en curso"),
        ("Expansion Norte", "Ventas", 700000, "2026-04-05", "en curso"),
        ("Renovacion Oficinas", "Operaciones", 500000, "2026-05-20", "pausado"),
        ("Auditoria Interna", "Finanzas", 150000, "2026-06-01", "finalizado"),
    ]
    clientes = [
        ("Grupo Pampa", "agro", "Rosario"), ("Banco del Plata", "finanzas", "Buenos Aires"),
        ("Salud Integral", "salud", "Santa Fe"), ("Retail Norte", "comercio", "Tucuman"),
        ("EduFuturo", "educacion", "Cordoba"), ("LogiCargo", "logistica", "Rosario"),
        ("Municipio Demo", "publico", "Rosario"), ("Industrias Delta", "industria", "Rosario"),
    ]
    metodos = ["Tarjeta corporativa", "Transferencia", "Efectivo", "Cheque"]
    nombres = ["Ana", "Luis", "Marta", "Pablo", "Sofia", "Diego", "Lucia", "Martin", "Carla", "Joaquin",
               "Valeria", "Nicolas", "Julieta", "Facundo", "Romina", "Tomas", "Agustina", "Matias",
               "Florencia", "Gonzalo", "Camila", "Federico", "Paula", "Ignacio"]
    apellidos = ["Gomez", "Perez", "Lopez", "Fernandez", "Diaz", "Ruiz", "Sosa", "Romero", "Torres", "Acosta"]
    cargos = ["analista", "coordinador", "jefe", "asistente", "gerente"]
    sal_base = {"analista": 900000, "coordinador": 1200000, "jefe": 1600000, "asistente": 700000, "gerente": 2200000}

    def rdate(d0: date, d1: date) -> date:
        return d0 + timedelta(days=rng.randint(0, (d1 - d0).days))

    with sqlite3.connect(path) as con:
        con.executescript(DDL)
        con.executemany("INSERT INTO departamentos(nombre, ciudad) VALUES (?,?)", departamentos)
        con.executemany("INSERT INTO categorias(nombre, tipo) VALUES (?,?)", [(c[0], c[1]) for c in categorias])
        con.executemany("INSERT INTO proveedores(nombre, rubro, ciudad) VALUES (?,?,?)", proveedores)
        con.executemany("INSERT INTO clientes(nombre, sector, ciudad) VALUES (?,?,?)", clientes)
        con.executemany("INSERT INTO metodos_pago(nombre) VALUES (?)", [(m,) for m in metodos])
        dept_id = {n: i + 1 for i, (n, _) in enumerate(departamentos)}
        con.executemany(
            "INSERT INTO proyectos(nombre, departamento_id, presupuesto_total, fecha_inicio, estado) VALUES (?,?,?,?,?)",
            [(n, dept_id[d], p, f, e) for n, d, p, f, e in proyectos],
        )
        emp_rows = []
        for i in range(len(nombres)):
            cargo = rng.choice(cargos)
            sal = round(sal_base[cargo] * rng.uniform(0.9, 1.15), 2)
            emp_rows.append((f"{nombres[i]} {rng.choice(apellidos)}", rng.randint(1, len(departamentos)),
                             cargo, sal, str(rdate(date(2018, 1, 1), date(2026, 6, 30)))))
        con.executemany(
            "INSERT INTO empleados(nombre, departamento_id, cargo, salario, fecha_ingreso) VALUES (?,?,?,?,?)",
            emp_rows,
        )
        # Gastos
        gastos = []
        for _ in range(400):
            ci = rng.randrange(len(categorias))
            nombre_cat, _, (lo, hi) = categorias[ci]
            emp_id = rng.randint(1, len(nombres))
            dep_id = emp_rows[emp_id - 1][1]
            gastos.append((
                nombre_cat, round(rng.uniform(lo, hi), 2), str(rdate(date(2026, 1, 1), date(2026, 9, 30))),
                ci + 1, dep_id, emp_id, rng.randint(1, len(proveedores)),
                rng.randint(1, len(proyectos)) if rng.random() < 0.55 else None,
                rng.randint(1, len(metodos)), f"Gasto de {nombre_cat}",
            ))
        con.executemany(
            "INSERT INTO gastos(categoria, monto, fecha, categoria_id, departamento_id, empleado_id,"
            " proveedor_id, proyecto_id, metodo_pago_id, descripcion) VALUES (?,?,?,?,?,?,?,?,?,?)",
            gastos,
        )
        # Ingresos
        conceptos = ["servicio mensual", "consultoria", "licencia", "proyecto a medida", "soporte"]
        con.executemany(
            "INSERT INTO ingresos(cliente_id, proyecto_id, monto, fecha, concepto) VALUES (?,?,?,?,?)",
            [(rng.randint(1, len(clientes)), rng.randint(1, len(proyectos)) if rng.random() < 0.6 else None,
              round(rng.uniform(20000, 250000), 2), str(rdate(date(2026, 1, 1), date(2026, 9, 30))),
              rng.choice(conceptos)) for _ in range(120)],
        )
        # Facturas
        fac = []
        for _ in range(80):
            emision = rdate(date(2026, 1, 1), date(2026, 9, 15))
            venc = emision + timedelta(days=30)
            if venc < date(2026, 9, 30):
                estado = rng.choices(["pagada", "vencida"], weights=[0.7, 0.3])[0]
            else:
                estado = rng.choices(["pagada", "pendiente"], weights=[0.3, 0.7])[0]
            fac.append((rng.randint(1, len(clientes)), round(rng.uniform(30000, 300000), 2),
                        str(emision), str(venc), estado))
        con.executemany(
            "INSERT INTO facturas(cliente_id, monto, fecha_emision, fecha_vencimiento, estado) VALUES (?,?,?,?,?)",
            fac,
        )
        # Presupuestos mensuales por departamento y categoría
        pres = []
        for d in range(1, len(departamentos) + 1):
            for ci, (_, _, (lo, hi)) in enumerate(categorias):
                for mes in range(1, 10):
                    pres.append((d, ci + 1, 2026, mes, round(rng.uniform(lo, hi) * rng.uniform(0.8, 1.6), 2)))
        con.executemany(
            "INSERT INTO presupuestos(departamento_id, categoria_id, anio, mes, monto_presupuestado) VALUES (?,?,?,?,?)",
            pres,
        )
    return f"Base creada con {EXPECTED_TABLES} tablas ({len(gastos)} gastos, {len(pres)} presupuestos)."


def _ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)


def list_tables() -> list[str]:
    with _ro() as con:
        return [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


# ---------------------------------------------------------------------------
# 2) Herramientas del agente
# ---------------------------------------------------------------------------
@tool
def sql_schema() -> str:
    """Devuelve todas las tablas, columnas, claves foráneas y una fila de ejemplo de la base SQLite de la empresa."""
    lines = []
    with _ro() as con:
        for table in list_tables():
            cols = con.execute(f'PRAGMA table_info("{table}")').fetchall()
            fks = con.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
            n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            lines.append(f"- {table} ({n} filas): " + ", ".join(f"{c[1]} {c[2]}" for c in cols))
            for fk in fks:
                lines.append(f"    FK: {table}.{fk[3]} -> {fk[2]}.{fk[4]}")
            sample = con.execute(f'SELECT * FROM "{table}" LIMIT 1').fetchone()
            if sample:
                lines.append(f"    ejemplo: {sample}")
    return "Esquema:\n" + "\n".join(lines)


@tool
def sql_query(query: str) -> str:
    """Ejecuta UNA consulta SELECT de solo lectura (se permiten JOIN, GROUP BY, subconsultas). Devuelve hasta 50 filas."""
    q = query.strip().rstrip(";")
    if ";" in q:
        return "Error: Solo se permite una sentencia SQL por llamada."
    if not re.match(r"(?is)^(select|with)\b", q):
        return "Error: Solo se permiten consultas de tipo SELECT."
    if re.search(r"\b(insert|update|delete|drop|alter|attach|create|replace|vacuum|pragma)\b", q, flags=re.I):
        return "Error: Palabra clave no permitida por seguridad."
    try:
        with _ro() as con:
            cur = con.cursor()
            cur.execute(q)
            rows = cur.fetchmany(MAX_ROWS + 1)
            columns = [d[0] for d in cur.description] if cur.description else []
    except sqlite3.Error as exc:
        return f"Error SQL: {exc}"
    if not rows:
        return "(sin resultados)"
    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    header = " | ".join(columns)
    sep = "-|-".join("-" * len(c) for c in columns)
    body = "\n".join(" | ".join(str(v) for v in r) for r in rows)
    note = f"\n… (mostrando primeras {MAX_ROWS} filas)" if truncated else ""
    return f"{header}\n{sep}\n{body}{note}"


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("Expresión no permitida.")


@tool
def calc(expression: str) -> str:
    """Calcula expresiones aritméticas simples (+, -, *, /, %, **). Ej: '45000 * 1.15'."""
    try:
        return str(_eval(ast.parse(expression, mode="eval")))
    except Exception as exc:
        return f"Error al calcular: {exc}"


TOOLS = [sql_schema, sql_query, calc]

SYSTEM_PROMPT = """Sos un asistente experto en finanzas empresariales que responde siempre en español rioplatense.

Tenés herramientas para consultar una base SQLite con varias tablas relacionadas (gastos, ingresos, facturas,
presupuestos, empleados, departamentos, proveedores, proyectos, clientes, categorias, metodos_pago) y para calcular.
Reglas:
1. Si no conocés el esquema, llamá primero a sql_schema. Usá JOIN cuando necesites nombres de otras tablas.
2. Usá sql_query para obtener datos y calc para cualquier cuenta que no resuelva el SQL.
3. No inventes resultados: si una consulta no devuelve filas o el dato no existe, decilo claramente.
4. Los montos están en pesos. Respondé de forma clara y concisa, incluyendo las cifras exactas obtenidas.
"""


# ---------------------------------------------------------------------------
# 3) Modelos y agente
# ---------------------------------------------------------------------------
def make_llm(backend: str, model: str, temperature: float = 0.0):
    if backend == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, max_retries=3)
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)
    raise ValueError(f"Backend desconocido: {backend}")


_AGENTS: dict[tuple[str, str, bool], object] = {}
_CHAT_CHECKPOINTER: SqliteSaver | None = None


def get_chat_thread_id() -> str:
    """Devuelve el identificador de chat guardado entre ejecuciones."""
    with closing(sqlite3.connect(CHAT_MEMORY_PATH)) as con:
        with con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS app_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = con.execute(
                "SELECT value FROM app_state WHERE key = 'active_chat_thread_id'"
            ).fetchone()
            if row:
                return row[0]
            thread_id = str(uuid.uuid4())
            con.execute(
                "INSERT INTO app_state (key, value) VALUES ('active_chat_thread_id', ?)",
                (thread_id,),
            )
            return thread_id


def _get_chat_checkpointer() -> SqliteSaver:
    global _CHAT_CHECKPOINTER
    if _CHAT_CHECKPOINTER is None:
        connection = sqlite3.connect(CHAT_MEMORY_PATH, check_same_thread=False, timeout=30)
        checkpointer = SqliteSaver(connection)
        try:
            checkpointer.setup()
        except Exception:
            connection.close()
            raise
        _CHAT_CHECKPOINTER = checkpointer
    return _CHAT_CHECKPOINTER


def get_agent(backend: str, model: str, persistent_memory: bool = False):
    key = (backend, model, persistent_memory)
    if key not in _AGENTS:
        _AGENTS[key] = create_react_agent(
            model=make_llm(backend, model),
            tools=TOOLS,
            prompt=SYSTEM_PROMPT,
            checkpointer=_get_chat_checkpointer() if persistent_memory else InMemorySaver(),
        )
    return _AGENTS[key]


def _text(content) -> str:
    if isinstance(content, list):
        content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return re.sub(r"(?is)<think>.*?</think>", "", str(content)).strip()


def run_agent(agent, message: str, thread_id: str) -> dict:
    """Ejecuta un turno y devuelve respuesta, traza y métricas."""
    cb = UsageMetadataCallbackHandler()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15, "callbacks": [cb]}
    t0 = time.perf_counter()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    except Exception as exc:  # rate limits, modelo inexistente, etc.
        return {"answer": "", "trace": "", "tool_calls": 0, "latency": time.perf_counter() - t0,
                "tokens": 0, "error": f"{type(exc).__name__}: {exc}"}
    latency = time.perf_counter() - t0

    msgs = result["messages"]
    last_h = max((i for i, m in enumerate(msgs) if getattr(m, "type", "") == "human"), default=0)
    steps, n_calls = [], 0
    for m in msgs[last_h + 1:]:
        for call in getattr(m, "tool_calls", None) or []:
            n_calls += 1
            steps.append(f"➡️ **Acción** `{call['name']}` con `{call['args']}`")
        if getattr(m, "type", "") == "tool":
            steps.append(f"👁️ **Observación** de `{m.name}`:\n```\n{str(m.content)[:1200]}\n```")
    tokens = sum(u.get("total_tokens", 0) for u in cb.usage_metadata.values())
    return {"answer": _text(msgs[-1].content), "trace": "\n\n".join(steps) or "_(sin herramientas)_",
            "tool_calls": n_calls, "latency": latency, "tokens": tokens, "error": ""}


# ---------------------------------------------------------------------------
# 4) Evaluación con LangChain
# ---------------------------------------------------------------------------
TEST_CASES = [
    dict(id="T01", tipo_prueba="agregado", turnos=["¿Cuánto gastamos en total en la categoría 'comida'?"],
         sql="SELECT ROUND(SUM(monto),2) FROM gastos WHERE categoria='comida'", chequeo="numerico"),
    dict(id="T02", tipo_prueba="agregado", turnos=["¿Cuánto se gastó en total en agosto de 2026?"],
         sql="SELECT ROUND(SUM(monto),2) FROM gastos WHERE fecha LIKE '2026-08%'", chequeo="numerico"),
    dict(id="T03", tipo_prueba="join", turnos=["¿Cuántos empleados tiene el departamento de Tecnologia?"],
         sql="SELECT COUNT(*) FROM empleados e JOIN departamentos d ON d.id=e.departamento_id WHERE d.nombre='Tecnologia'",
         chequeo="numerico"),
    dict(id="T04", tipo_prueba="join", turnos=["¿Cuál es el salario promedio en el departamento de Ventas?"],
         sql="SELECT ROUND(AVG(e.salario),2) FROM empleados e JOIN departamentos d ON d.id=e.departamento_id WHERE d.nombre='Ventas'",
         chequeo="numerico"),
    dict(id="T05", tipo_prueba="ranking", turnos=["¿Qué categoría de gasto acumula el mayor monto total?"],
         sql="SELECT categoria FROM gastos GROUP BY categoria ORDER BY SUM(monto) DESC LIMIT 1", chequeo="texto"),
    dict(id="T06", tipo_prueba="ranking", turnos=["¿A qué proveedor le gastamos más en total?"],
         sql="SELECT pv.nombre FROM gastos g JOIN proveedores pv ON pv.id=g.proveedor_id GROUP BY pv.id ORDER BY SUM(g.monto) DESC LIMIT 1",
         chequeo="texto"),
    dict(id="T07", tipo_prueba="join", turnos=["¿Cuánto se gastó en total en el proyecto 'Portal Ciudadano'?"],
         sql="SELECT ROUND(SUM(g.monto),2) FROM gastos g JOIN proyectos p ON p.id=g.proyecto_id WHERE p.nombre='Portal Ciudadano'",
         chequeo="numerico"),
    dict(id="T08", tipo_prueba="ingresos", turnos=["¿Cuánto ingresó en septiembre de 2026?"],
         sql="SELECT ROUND(SUM(monto),2) FROM ingresos WHERE fecha LIKE '2026-09%'", chequeo="numerico"),
    dict(id="T09", tipo_prueba="multi-valor", turnos=["¿Cuántas facturas están vencidas y cuál es el monto total adeudado en ellas?"],
         sql="SELECT COUNT(*), ROUND(SUM(monto),2) FROM facturas WHERE estado='vencida'", chequeo="numerico"),
    dict(id="T10", tipo_prueba="multi-paso", turnos=["¿Cuál es el balance total (ingresos menos gastos) de la empresa?"],
         sql="SELECT ROUND((SELECT SUM(monto) FROM ingresos) - (SELECT SUM(monto) FROM gastos),2)", chequeo="numerico"),
    dict(id="T11", tipo_prueba="multi-valor",
         turnos=["¿Qué presupuesto total se asignó al departamento de Marketing y cuánto gastó realmente?"],
         sql="SELECT (SELECT ROUND(SUM(pr.monto_presupuestado),2) FROM presupuestos pr JOIN departamentos d ON d.id=pr.departamento_id WHERE d.nombre='Marketing'),"
             " (SELECT ROUND(SUM(g.monto),2) FROM gastos g JOIN departamentos d ON d.id=g.departamento_id WHERE d.nombre='Marketing')",
         chequeo="numerico"),
    dict(id="T12", tipo_prueba="memoria",
         turnos=["¿Cuánto gastamos en total en la categoría 'viajes'?", "Si ese gasto aumenta un 15% el próximo mes, ¿cuánto sería?"],
         sql="SELECT ROUND(SUM(monto)*1.15,2) FROM gastos WHERE categoria='viajes'", chequeo="numerico"),
    dict(id="T13", tipo_prueba="honestidad", turnos=["¿Cuánto se gastó en la categoría 'astrologia'?"],
         sql=None, chequeo="vacio"),
]


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _candidates(token: str) -> list[float]:
    t = token.strip(".,")
    neg = t.startswith("-")
    t = t.lstrip("-")
    outs: list[str] = []
    if "." in t and "," in t:
        outs.append(t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", ""))
    elif "," in t:
        outs.append(t.replace(",", "."))
        if re.fullmatch(r"\d{1,3}(,\d{3})+", t):
            outs.append(t.replace(",", ""))
    elif "." in t:
        outs.append(t)
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            outs.append(t.replace(".", ""))
    else:
        outs.append(t)
    vals = []
    for o in outs:
        try:
            v = float(o)
            vals.append(-v if neg else v)
        except ValueError:
            pass
    return vals


def numbers_in(text: str) -> list[list[float]]:
    """Extrae números tolerando formato rioplatense (1.234,56) y anglosajón (1,234.56)."""
    return [c for c in (_candidates(m.group()) for m in re.finditer(r"-?\d[\d.,]*", text)) if c]


def contains_value(text: str, expected: float) -> bool:
    return any(math.isclose(v, expected, rel_tol=2e-3, abs_tol=0.011)
               for cands in numbers_in(text) for v in cands)


def ground_truth(case: dict):
    if not case.get("sql"):
        return None
    with _ro() as con:
        row = con.execute(case["sql"]).fetchone()
    return row


def programmatic_check(case: dict, answer: str, truth) -> bool | None:
    if not answer:
        return False
    if case["chequeo"] == "numerico":
        return all(contains_value(answer, float(v)) for v in truth)
    if case["chequeo"] == "texto":
        return _ascii(str(truth[0])) in _ascii(answer)
    if case["chequeo"] == "vacio":  # no debe inventar cifras distintas de cero
        return all(v == 0 for cands in numbers_in(answer) for v in cands[:1])
    return None


class Veredicto(BaseModel):
    """Evaluación del LLM-juez sobre la respuesta de un agente."""
    correccion: int = Field(ge=1, le=5, description="1=incorrecta, 5=totalmente correcta respecto a la referencia")
    claridad: int = Field(ge=1, le=5, description="1=confusa, 5=clara y concisa")
    uso_herramientas: int = Field(ge=1, le=5, description="1=uso erróneo/innecesario, 5=uso eficiente y adecuado")
    justificacion: str = Field(description="Explicación breve (máx. 2 oraciones)")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Sos un evaluador estricto e imparcial de agentes de IA que consultan una base de datos financiera. "
               "Comparás la respuesta del agente con el resultado de referencia (calculado directamente con SQL). "
               "Penalizá cifras erróneas o inventadas. Ignorá diferencias de formato numérico (1.234,5 vs 1234.5)."),
    ("human", "Pregunta(s) del usuario:\n{pregunta}\n\n"
              "Resultado de referencia (SQL): {referencia}\n\n"
              "Traza de herramientas del agente:\n{traza}\n\n"
              "Respuesta final del agente:\n{respuesta}"),
])


def build_judge_chain(backend: str, judge_model: str):
    """Cadena LCEL: prompt | LLM con salida estructurada (Pydantic)."""
    return JUDGE_PROMPT | make_llm(backend, judge_model).with_structured_output(Veredicto)


def evaluate_models(models: list[str], backend: str, judge_model: str, case_ids: list[str],
                    reps: int = 1) -> Iterator[tuple[str, pd.DataFrame]]:
    cases = [c for c in TEST_CASES if c["id"] in case_ids]
    judge = build_judge_chain(backend, judge_model)
    rows: list[dict] = []
    total = len(models) * len(cases) * reps
    done = 0
    for model in models:
        agent = get_agent(backend, model)
        for case in cases:
            truth = ground_truth(case)
            for rep in range(1, reps + 1):
                tid = f"eval-{uuid.uuid4().hex[:8]}"
                answers, traces, lat, toks, calls, err = [], [], 0.0, 0, 0, ""
                for q in case["turnos"]:
                    r = run_agent(agent, q, tid)
                    answers.append(r["answer"]); traces.append(r["trace"])
                    lat += r["latency"]; toks += r["tokens"]; calls += r["tool_calls"]
                    err = err or r["error"]
                answer = answers[-1] if answers else ""
                ok = programmatic_check(case, answer, truth) if not err else False
                verdict = None
                if not err:
                    try:
                        verdict = judge.invoke({
                            "pregunta": " → ".join(case["turnos"]),
                            "referencia": str(truth) if truth is not None
                                          else "No existe ese dato: el agente debe decir que no hay registros, sin inventar cifras.",
                            "traza": "\n".join(traces)[:2500], "respuesta": answer,
                        })
                    except Exception as exc:
                        err = f"Juez: {type(exc).__name__}: {str(exc)[:120]}"
                rows.append({
                    "modelo": model, "caso": case["id"], "tipo": case["tipo_prueba"], "rep": rep,
                    "correcto_sql": None if ok is None else int(ok),
                    "juez_correccion": verdict.correccion if verdict else None,
                    "juez_claridad": verdict.claridad if verdict else None,
                    "juez_herramientas": verdict.uso_herramientas if verdict else None,
                    "latencia_s": round(lat, 2), "tool_calls": calls, "tokens": toks,
                    "referencia": str(truth), "respuesta": answer[:400],
                    "juez_comentario": verdict.justificacion if verdict else "", "error": err,
                })
                done += 1
                yield f"Progreso: {done}/{total} — {model} · {case['id']}", pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    g = detail.groupby("modelo")
    s = pd.DataFrame({
        "exactitud_%": g["correcto_sql"].apply(lambda x: x.dropna().astype(float).mean() * 100),
        "juez_correccion": g["juez_correccion"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean()),
        "juez_claridad": g["juez_claridad"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean()),
        "juez_herramientas": g["juez_herramientas"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean()),
        "latencia_prom_s": g["latencia_s"].mean(),
        "tool_calls_prom": g["tool_calls"].mean(),
        "tokens_totales": g["tokens"].sum(),
        "errores": g["error"].apply(lambda x: int((x != "").sum())),
    })
    juez_pct = s[["juez_correccion", "juez_claridad", "juez_herramientas"]].mean(axis=1) / 5 * 100
    s["score_global"] = ((s["exactitud_%"].fillna(0) + juez_pct.fillna(0)) / 2).round(1)
    return s.round(2).reset_index().sort_values("score_global", ascending=False)


# ---------------------------------------------------------------------------
# 5) Interfaz Gradio
# ---------------------------------------------------------------------------
def _set_key(api_key: str) -> None:
    if api_key and api_key.strip():
        os.environ["GROQ_API_KEY"] = api_key.strip()


def _need_key(backend: str) -> str | None:
    if backend == "groq" and not os.environ.get("GROQ_API_KEY"):
        return "Falta la GROQ_API_KEY: pegala en el campo correspondiente o definila como variable de entorno."
    return None


def chat_fn(message, history, backend, model, thread_id, api_key):
    history = history or []
    if not message or not message.strip():
        return "", history, gr.update(), gr.update()
    _set_key(api_key)
    if (msg := _need_key(backend)):
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": f"⚠️ {msg}"}]
        return "", history, "", ""
    res = run_agent(get_agent(backend, model, persistent_memory=True), message, thread_id)
    reply = res["answer"] or f"⚠️ Error: {res['error']}"
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    stats = (f"⏱️ {res['latency']:.1f}s · 🔧 {res['tool_calls']} llamadas a herramientas · 🔤 {res['tokens']} tokens")
    return "", history, res["trace"], stats


def clear_chat(thread_id: str):
    return [], thread_id, "", ""


def update_model_choices(backend):
    ch = DEFAULT_GROQ_MODELS if backend == "groq" else DEFAULT_OLLAMA_MODELS
    return (gr.update(choices=ch, value=ch[0]), gr.update(choices=ch, value=ch[:2]),
            gr.update(choices=ch, value=ch[0] if backend == "ollama" else DEFAULT_JUDGE))


def run_comparison(models, judge_model, backend, case_labels, reps, api_key):
    _set_key(api_key)
    empty = pd.DataFrame()
    if (msg := _need_key(backend)):
        yield f"⚠️ {msg}", empty, empty, empty, None
        return
    if not models:
        yield "⚠️ Elegí al menos un modelo.", empty, empty, empty, None
        return
    ids = [lbl.split(" · ")[0] for lbl in case_labels] or [c["id"] for c in TEST_CASES]
    detail = empty
    try:
        for status, detail in evaluate_models(list(models), backend, judge_model, ids, int(reps)):
            summ = summarize(detail)
            yield status, summ, detail, summ[["modelo", "score_global"]] if not summ.empty else empty, None
    except Exception as exc:
        yield f"❌ {type(exc).__name__}: {exc}", empty, detail, empty, None
        return
    detail.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    summ = summarize(detail)
    yield "✅ Evaluación completa. CSV listo para descargar.", summ, detail, summ[["modelo", "score_global"]], str(RESULTS_CSV)


def schema_markdown() -> str:
    out = []
    with _ro() as con:
        for t in list_tables():
            cols = ", ".join(f"`{c[1]}`" for c in con.execute(f'PRAGMA table_info("{t}")'))
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            out.append(f"- **{t}** ({n} filas): {cols}")
    return "\n".join(out)


def preview_table(table: str) -> pd.DataFrame:
    with _ro() as con:
        return pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT 100', con)


def regenerate_db():
    msg = build_database(force=True)
    t = list_tables()
    return msg, schema_markdown(), gr.update(choices=t, value=t[0]), preview_table(t[0])


def build_ui() -> gr.Blocks:
    build_database()
    tables = list_tables()
    case_labels = [f"{c['id']} · {c['tipo_prueba']} · {c['turnos'][-1][:55]}" for c in TEST_CASES]

    with gr.Blocks(title="Agente Financiero · TP2", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 💼 Agente Inteligente de Gestión Financiera\n"
                    "LangChain + LangGraph (ReAct, tool calling, memoria) · Comparador de modelos · SQLite de 11 tablas")
        with gr.Row():
            backend = gr.Radio(["groq", "ollama"], value="groq", label="Backend")
            api_key = gr.Textbox(label="GROQ_API_KEY (opcional si ya está en el entorno)", type="password", scale=2)

        with gr.Tabs():
            # ---------------- Chat ----------------
            with gr.Tab("💬 Chat"):
                model = gr.Dropdown(DEFAULT_GROQ_MODELS, value=DEFAULT_GROQ_MODELS[0], allow_custom_value=True,
                                    label="Modelo")
                thread_id = gr.State(get_chat_thread_id())
                chatbot = gr.Chatbot(type="messages", height=420, label="Conversación (con memoria)")
                with gr.Row():
                    box = gr.Textbox(placeholder="Ej: ¿Cuánto gastamos en 'comida'?", scale=5, show_label=False)
                    send = gr.Button("Enviar", variant="primary", scale=1)
                    clear = gr.Button("🧹 Limpiar pantalla (conservar memoria)", scale=1)
                stats = gr.Markdown()
                with gr.Accordion("🔍 Traza del agente (acciones y observaciones)", open=False):
                    trace = gr.Markdown()
                gr.Examples(
                    ["¿Cuánto gastamos en total en la categoría 'comida'?",
                     "Si aumentamos ese gasto un 15%, ¿cuánto sería?",
                     "¿Qué departamento gastó más y cuánto?",
                     "¿Cuántas facturas vencidas hay y de qué clientes?",
                     "Comparame presupuesto vs gasto real de Tecnologia"], inputs=box)
                ins = [box, chatbot, backend, model, thread_id, api_key]
                outs = [box, chatbot, trace, stats]
                box.submit(chat_fn, ins, outs)
                send.click(chat_fn, ins, outs)
                clear.click(clear_chat, thread_id, [chatbot, thread_id, trace, stats])

            # ---------------- Comparador ----------------
            with gr.Tab("📊 Comparador de modelos"):
                gr.Markdown(
                    "Cada modelo responde la batería de pruebas. Se evalúa con **(1)** verificación contra el resultado real "
                    "de SQL, **(2)** un **LLM-juez** (cadena LCEL con salida estructurada) y **(3)** latencia, tokens y uso de herramientas. "
                    "`score_global` = promedio entre exactitud % y puntaje del juez (normalizado a 100).")
                with gr.Row():
                    cmp_models = gr.Dropdown(DEFAULT_GROQ_MODELS, value=DEFAULT_GROQ_MODELS[:2], multiselect=True,
                                             allow_custom_value=True, label="Modelos a comparar")
                    judge = gr.Dropdown(DEFAULT_GROQ_MODELS, value=DEFAULT_JUDGE, allow_custom_value=True,
                                        label="Modelo juez")
                    reps = gr.Slider(1, 3, value=1, step=1, label="Repeticiones por caso")
                cases = gr.CheckboxGroup(case_labels, value=case_labels, label="Casos de prueba")
                run = gr.Button("▶️ Ejecutar evaluación", variant="primary")
                status = gr.Markdown()
                summary = gr.Dataframe(label="Resumen por modelo", wrap=True)
                plot = gr.BarPlot(x="modelo", y="score_global", title="Score global por modelo", y_lim=[0, 100])
                detail = gr.Dataframe(label="Detalle por caso", wrap=True)
                dl = gr.File(label="Descargar resultados (CSV)")
                run.click(run_comparison, [cmp_models, judge, backend, cases, reps, api_key],
                          [status, summary, detail, plot, dl])

            # ---------------- Base de datos ----------------
            with gr.Tab("🗄️ Base de datos"):
                schema_md = gr.Markdown(schema_markdown())
                with gr.Row():
                    tsel = gr.Dropdown(tables, value=tables[0], label="Tabla")
                    regen = gr.Button("♻️ Regenerar base de datos")
                db_msg = gr.Markdown()
                grid = gr.Dataframe(preview_table(tables[0]), label="Primeras 100 filas", wrap=True)
                tsel.change(preview_table, tsel, grid)
                regen.click(regenerate_db, None, [db_msg, schema_md, tsel, grid])

        backend.change(update_model_choices, backend, [model, cmp_models, judge])
    return demo


if __name__ == "__main__":
    print(build_database())
    app = build_ui()
    app.queue().launch(share="google.colab" in sys.modules)
