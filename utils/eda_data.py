"""utils/eda_data.py — Datos extraidos de los 4 EDAs ejecutados.
Cada dict/DataFrame replica exactamente los outputs de los notebooks.
"""
from __future__ import annotations
import pandas as pd

# ═══════════════════════════════════════════════════════════
# 1. PROMISE
# ═══════════════════════════════════════════════════════════
PROMISE_BALANCE = {"Limpio (0)": 10047, "Defectuoso (1)": 5728}

PROMISE_CK_CORR = pd.DataFrame({
    "metrica": ["rfc","npm","wmc","loc","ce","cbm","moa","amc","max_cc","cbo",
                "avg_cc","ic","lcom","ca","dam","noc","dit","mfa","lcom3","cam"],
    "correlacion": [0.202,0.172,0.166,0.164,0.147,0.141,0.136,0.126,0.115,0.112,
                    0.090,0.085,0.069,0.067,0.038,0.032,0.027,0.018,-0.071,-0.162],
})

PROMISE_DEFECT_DENSITY = pd.DataFrame({
    "proyecto": ["xalan-2.7","log4j-1.2","velocity-1.4","xerces-1.4.4","velocity-1.5",
                 "poi-3.0","jedit-4.3","xalan-2.6","lucene-2.4","camel-1.6",
                 "xalan-2.5","xalan-2.4","ant-1.7","xerces-1.3","camel-1.4",
                 "poi-2.5","xerces-1.2","camel-1.2","jedit-4.2","ivy-1.2",
                 "ant-1.6","ant-1.5","camel-1.0","ivy-1.4","ant-1.4",
                 "ant-1.3","jedit-3.2","poi-2.0","jedit-4.0","forrest-0.7",
                 "ivy-2.0","lucene-2.0","log4j-1.0","jedit-4.1","pbeans-2",
                 "log4j-1.1","arc","synapse-1.0","synapse-1.1","synapse-1.2","forrest-0.6"],
    "densidad": [0.988,0.922,0.750,0.743,0.664,0.642,0.561,0.458,0.396,0.195,
                 0.385,0.362,0.223,0.152,0.166,0.642,0.152,0.363,0.130,0.068,
                 0.263,0.109,0.036,0.068,0.225,0.160,0.333,0.117,0.248,0.024,
                 0.115,0.462,0.341,0.249,0.087,0.338,0.116,0.101,0.271,0.337,0.049],
})

PROMISE_CK_STATS = pd.DataFrame({
    "metrica": ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","loc",
                "dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc","lcom3"],
    "media": [13.087,1.456,0.533,7.875,26.923,434.577,4.625,5.455,8.185,117.618,
              0.315,0.199,0.424,0.416,0.548,0.585,28.494,4.144,1.317,0.517],
    "mediana": [8.0,1.0,0.0,5.0,17.0,5.0,2.0,3.0,5.0,52.0,
                0.0,0.0,0.429,0.385,0.0,0.0,16.4,2.0,1.0,0.571],
    "max": [540,8,134,172,508,99999,66,98,218,12290,
            1,8,1,1,5,6,1197,236,28.667,2],
})

# ═══════════════════════════════════════════════════════════
# 2. APACHE JIRA
# ═══════════════════════════════════════════════════════════
APACHE_COLECCIONES = pd.DataFrame({
    "coleccion": ["comments","events","issues","projects","users","worklogs"],
    "filas": [1875,5120,1875,656,5120,5120],
    "columnas": [12,7,112,43,13,14],
    "cols_completas": [10,7,65,42,13,14],
})

APACHE_ISSUES_POR_ANO = pd.DataFrame({
    "ano": [2003,2004,2005,2006,2009,2010,2011,2014,2015,2020],
    "issues": [11,286,80,373,1,374,163,236,139,212],
})

APACHE_CYCLE_TIME = {
    "con_resolucion": 1560, "total": 1875,
    "media": 205.88, "mediana": 16.0, "p25": 1.0, "p75": 103.0, "max": 3516.0,
}

APACHE_TRANSICIONES = pd.DataFrame({
    "estado": ["Resolved","Patch Available","Closed","Open","In Progress"],
    "conteo": [232,202,97,89,63],
})

# ═══════════════════════════════════════════════════════════
# 3. RED HAT JIRA
# ═══════════════════════════════════════════════════════════
REDHAT_ISSUE_TYPES = pd.DataFrame({
    "tipo": ["Bug","Task","Story","Feature Request","Enhancement","Epic",
             "Component Upgrade","Feature","Spike","Release"],
    "conteo": [223922,112917,71406,33654,24826,13293,9650,6656,3013,1104],
})

REDHAT_STATUSES = pd.DataFrame({
    "estado": ["Closed","Resolved","New","Open","To Do","Done","Backlog",
               "Verified","In Progress","Release Pending","Planning",
               "Rejected","PR Sent","Dev Complete","Accepted"],
    "conteo": [362387,61483,21621,13427,8453,6696,5720,5656,4315,2668,
               2268,917,912,895,893],
})

REDHAT_CYCLE_TIME = {
    "total": 505096, "con_resolucion": 436475, "cobertura_pct": 86.4,
    "p50": 28.0, "p75": 127.0, "p90": 400.0, "media": 156.5, "max": 7030.0,
}

REDHAT_META = {
    "filas": 505096, "columnas": 14, "proyectos": 251,
    "rango_created": ("2001-04-06", "2024-11-27"),
    "rango_resolved": ("2004-11-23", "2024-11-27"),
}

# ═══════════════════════════════════════════════════════════
# 4. GHARCHIVE
# ═══════════════════════════════════════════════════════════
GHARCHIVE_EVENT_TYPES = pd.DataFrame({
    "tipo": ["IssueCommentEvent","PullRequestReviewEvent","PullRequestReviewCommentEvent",
             "PullRequestEvent","PushEvent","WatchEvent","IssuesEvent","ForkEvent",
             "CreateEvent","DeleteEvent","GollumEvent","ReleaseEvent","MemberEvent"],
    "conteo": [1094,1085,796,656,589,304,182,162,72,29,15,2,8],
})

GHARCHIVE_DEPLOY_FREQ = pd.DataFrame({
    "repo": ["apache/camel","apache/airflow","apache/incubator-nuttx","apache/myfaces-tobago",
             "apache/beam","apache/hbase","apache/arrow","apache/spark","apache/tinkerpop",
             "apache/shardingsphere","apache/flink","apache/trafficserver","apache/maven-dist-tool"],
    "pushes_main": [11,10,9,8,7,6,6,6,6,6,6,5,5],
})

GHARCHIVE_TOP_REPOS = pd.DataFrame({
    "repo": ["apache/airflow","apache/spark","apache/arrow","apache/pulsar","apache/beam",
             "apache/hudi","apache/flink","apache/tvm","apache/camel","apache/kafka",
             "apache/shardingsphere","apache/dubbo","apache/trafficserver",
             "apache/incubator-nuttx","apache/superset"],
    "eventos": [247,211,208,176,163,135,125,120,117,115,95,91,85,80,72],
})

GHARCHIVE_TEMPORAL = pd.DataFrame({
    "hora": [0,6,12,18],
    "eventos": [1033,1409,1490,1062],
})

GHARCHIVE_CFR = {
    "prs_merged": 255, "prs_cerrados_sin_merge": 90,
    "cfr_proxy": 0.261, "total_prs": 656,
}

GHARCHIVE_META = {
    "archivos_json": 24, "tamano_gib": 1.69,
    "eventos_muestra": 4994, "tipos_unicos": 13,
    "rango": ("2022-01-01", "2022-03-15"),
    "repos_apache": 118,
}
