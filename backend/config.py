"""
配置模組 - 包含系統配置和常量定義
"""

# --- 使用者可修改參數 ---
TARGET_SCORE_Q = [1, 2, 3, 4, 11]
BATCH_SIZE = 6
DO_SIMILARITY_CHECK = True

# --- 相似度檢測閾值 ---
SIMILARITY_THRESHOLDS = {
    'high': 0.85,
    'medium': 0.70,
    'min_length': 50
}

# --- 評分標準 (Rubrics) ---
RUBRICS = {
    1: '''Rubric (10 pts total)

Section A – CZ vs FZ single‑crystal Si comparison (0‑4 pts)
  • Clearly list ≥ 2 advantages and ≥ 2 disadvantages for **each** process  
    – e.g. CZ: large diameter, cheaper / higher O contamination, lower resistivity  
    – FZ: crucible‑less, ultra‑high purity, high resistivity / smaller boule, costly  
  • 0.5 pt per correct advantage/disadvantage, up to 4 pts

Section B – Channeling effect definition & impact (0‑3 pts)
  • Defines channeling as ions travelling along low‑index crystal axes/planes (1 pt)  
  • Describes deeper projected range / dose loss / tail in dopant profile (1 pt)  
  • Mentions dependence on crystal orientation/energy (1 pt)

Section C – Mitigation methods (0‑3 pts)
  Any three of the following, 1 pt each (max 3 pts):  
    – Tilt/rotate wafer during implantation  
    – Use amorphizing pre‑implant (e.g. Si, Ge)  
    – Grow/retain surface oxide or SiN mask  
    – Implant through amorphous layer (screen oxide)  
    – Use random beam incidence or beam wobbling
''',
    2: '''Rubric (10 pts total) — Student **answers any TWO** sub‑problems  
Score each answered sub‑problem 0‑5 pts, keep the best two (max 10)

Sub‑problem 1  Mass‑transport‑ vs. surface‑reaction‑limited CVD (0‑5)
  • Correct definition of each regime & rate‑determining step (2 pts)  
  • Describes dependence on temperature, pressure, boundary layer (1 pt)  
  • Mentions impact on thickness uniformity or step coverage (1 pt)  
  • Gives practical example or sketch of concentration profile (1 pt)

Sub‑problem 2  MBE working principle (0‑5)
  • UHV environment & effusion cells produce atom/molecule beams (2 pts)  
  • Ballistic arrival / adsorption–surface diffusion–incorporation process (1 pt)  
  • In‑situ monitoring (e.g. RHEED) & precise flux control (1 pt)  
  • Typical growth rate (~1 µm h⁻¹) & ultra‑high purity advantage (1 pt)

Sub‑problem 3  Exceeding critical thickness in heteroepitaxy (0‑5)
  • Introduces misfit strain & Matthews–Blakeslee criterion (1 pt)  
  • Formation of misfit dislocations / strain relaxation (2 pts)  
  • Possible 3‑D islanding (S–K), surface roughening or cracks (1 pt)  
  • Electrical/optical degradation consequence (1 pt)
''',
    3: '''Rubric (10 pts total) — Student **answers any TWO** sub‑problems  
Score each answered sub‑problem 0‑5 pts, keep the best two (max 10)

Sub‑problem 1  Si vs GaAs band structure differences (0‑5)
  • Indirect (Si) vs direct (GaAs) bandgap nature & values (2 pts)  
  • Conduction‑band valley positions / density‑of‑states (1 pt)  
  • Carrier mobility & effective mass comparison (1 pt)  
  • Consequence for optoelectronic efficiency (1 pt)

Sub‑problem 2  "Straddling gap" heterojunction (Type‑I) (0‑5)
  • Correctly picks Type‑I (1 pt)  
  • Conduction & valence bands of wider‑gap material both higher/lower enclosing narrow‑gap (2 pts)  
  • Draws / verbally describes band diagram & carrier confinement (2 pts)

Sub‑problem 3  Multijunction solar‑cell spectral utilization (0‑5)
  • Bandgap stacking / current matching concept (2 pts)  
  • Use of tunnel junctions / graded buffers (1 pt)  
  • Spectrum splitting or lattice‑matched material selection (1 pt)  
  • Anti‑reflection or light‑management strategy (1 pt)
''',
    4: '''Rubric (10 pts total) — Student **answers any TWO** sub‑problems  
Score each answered sub‑problem 0‑5 pts, keep the best two (max 10)

Sub‑problem 1  Early challenges in GaN on sapphire (0‑5)
  • ~16 % lattice mismatch & thermal‑expansion mismatch (2 pts)  
  • High threading dislocation density / cracking (1 pt)  
  • Poor surface wetting, nucleation issues prior to LT‑buffer invention (1 pt)  
  • Impact on LED efficiency / reliability (1 pt)

Sub‑problem 2  Difficulty of achieving p‑type GaN (0‑5)
  • Deep acceptor level of Mg (≈200 meV) limits hole activation (2 pts)  
  • H passivation forming Mg–H complexes, need anneal (1 pt)  
  • Compensation by native donors / defects (1 pt)  
  • Historically low hole mobility / conductivity (1 pt)

Sub‑problem 3  (D) Lower manufacturing cost? — critical discussion (0‑5)
  • States that early GaN/Sapphire actually ↑ cost due to low yield (1 pt)  
  • Explains how subsequent mass‑production & cheap sapphire made cost reasonable (1 pt)  
  • Compares to SiC, bulk GaN or phosphor‑converted alternatives (1 pt)  
  • Evaluates epi reactor throughput, wafer price, device efficiency vs cost (2 pts)
''',
    11: '''Rubric (10 pts total)

Part 1  Quantum confinement in 0‑D nanomaterials (0‑5)
  • Defines confinement when particle size ≲ exciton Bohr radius (1 pt)  
  • Energy levels become discrete, bandgap widens with decreasing size (2 pts)  
  • Mentions size‑tunable optical emission / quantum‑size effect (1 pt)  
  • Gives formula E ∝ 1/L² or cites typical CdSe Q‑dot example (1 pt)

Part 2  Si nanowires as Li‑ion anode (0‑5)
  • Lists Si theoretical capacity ≈4200 mAh g⁻¹ (1 pt)  
  • bulk Si issues: >300 % volume expansion, pulverization, loss of contact (2 pts)  
  • SiNWs accommodate strain radially / maintain electrical pathway (1 pt)  
  • Large surface area for fast Li diffusion & facile SEI formation control (1 pt)
''',
}

# --- Prompt 模板 ---
PROMPT_TEMPLATE = """You are a meticulous graduate-level TA.
First, judge whether the following answer exhibits obvious LLM writing style (including but not limited to overly smooth transitions, template phrases like 'moreover', almost no typos).
Return an integer 'ai_risk' from 0‑100 (higher = more AI-like).
{grade_block}
Respond ONLY with a valid JSON object like {{"ai_risk": ..., "score": ...}}.

Question:
{question}

Answer:
{answer}
"""
