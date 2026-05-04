"""Build submission HTML files with embedded figures."""
import base64, pathlib, markdown as md

def b64img(path):
    p = pathlib.Path(path)
    if not p.exists():
        return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
    data = p.read_bytes()
    return 'data:image/png;base64,' + base64.b64encode(data).decode()

imgs = {k: b64img(f'figures/{f}') for k, f in [
    ('scatter',      'fig1_awi_vs_pqi_scatter.png'),
    ('heatmap',      'fig2_position_heatmap.png'),
    ('half_compare', 'fig3_half_comparison.png'),
    ('pqi_decomp',   'fig4_pqi_decomposition.png'),
]}

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; color: #1a1a2e; background: #fff; }
.slide {
  max-width: 860px; margin: 0 auto; padding: 36px 48px 32px 48px;
  border-bottom: 3px solid #e8e8f0;
  page-break-after: always;
}
.slide:last-child { border-bottom: none; page-break-after: avoid; }
.slide-header {
  display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: 18px; border-bottom: 2px solid #0d47a1; padding-bottom: 8px;
}
.slide-num { font-size: 8.5pt; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.slide-title { font-size: 16pt; font-weight: 700; color: #0d47a1; }
.slide-subtitle { font-size: 11pt; color: #444; margin-bottom: 16px; font-style: italic; }
h3 { font-size: 10.5pt; font-weight: 700; color: #1a1a2e; margin: 14px 0 6px 0; }
p { margin: 8px 0; line-height: 1.55; }
ul { padding-left: 20px; margin: 6px 0 10px 0; }
li { margin-bottom: 4px; line-height: 1.5; }
blockquote {
  border-left: 4px solid #0d47a1; margin: 14px 0; padding: 8px 16px;
  color: #444; font-style: italic; background: #f4f6fb; border-radius: 0 4px 4px 0;
}
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }
th { background: #0d47a1; color: #fff; padding: 7px 11px; text-align: left; font-weight: 600; }
td { border: 1px solid #dde; padding: 6px 11px; }
tr:nth-child(even) td { background: #f7f8fc; }
code { background: #f0f2f8; padding: 2px 5px; font-size: 9pt; font-family: Consolas, monospace; border-radius: 2px; }
pre { background: #f0f2f8; padding: 14px; border-left: 3px solid #0d47a1; font-size: 9pt;
      font-family: Consolas, monospace; line-height: 1.5; margin: 10px 0; overflow-x: auto; }
.fig { text-align: center; margin: 16px 0; }
.fig img { max-width: 100%; border: 1px solid #e0e0e8; border-radius: 4px; }
.fig-caption { font-size: 8.5pt; color: #777; margin-top: 4px; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 12px 0; }
.kpi-bar { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
.kpi-label { width: 160px; font-size: 9.5pt; flex-shrink: 0; }
.kpi-val { font-weight: 700; font-size: 10pt; color: #0d47a1; }
.tag { display: inline-block; background: #e8f0fe; color: #1a237e; padding: 1px 7px;
       border-radius: 10px; font-size: 8.5pt; font-weight: 600; margin: 0 2px; }
.cover { text-align: center; padding: 56px 48px 48px; }
.cover h1 { font-size: 28pt; color: #0d47a1; margin-bottom: 12px; }
.cover .sub { font-size: 13pt; color: #444; margin-bottom: 6px; }
.cover .meta { font-size: 9.5pt; color: #888; margin-top: 24px; line-height: 1.8; }
@media print {
  body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .slide { margin: 0; }
}
"""

SLIDES = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Football Body Intelligence Platform  -  Executive Summary</title>
<style>{CSS}</style>
</head>
<body>

<!-- COVER -->
<div class="slide cover">
  <h1>Football Body Intelligence Platform</h1>
  <p class="sub">AWI &middot; Awareness Index &nbsp;+&nbsp; PQI &middot; Pressure Quality Index</p>
  <p class="sub" style="font-size:11pt; color:#555;">AWS World Sports Innovation Cup 2026 &middot; Challenge 2</p>
  <p class="meta">
    Team: itzmore<br>
    5 Bundesliga matches &middot; 400 player-phase observations &middot; TRACAB TF15 skeleton data at 50 fps
  </p>
</div>

<!-- SLIDE 1: PROBLEM & KPIs -->
<div class="slide">
  <div class="slide-header">
    <span class="slide-num">Slide 1 of 6</span>
    <span class="slide-title">The Problem &amp; Our Two KPIs</span>
  </div>
  <p class="slide-subtitle">Standard tracking tells you <em>where</em> players move. TF15 tells you <em>how</em> they move and <em>where they look</em>.</p>

  <p>The Bundesliga's TRACAB TF15 system captures 141 million data points per match  -  full 3D skeleton at 50 fps. Today it powers only automated event detection. The body intelligence it contains goes unmeasured.</p>

  <div class="two-col" style="margin-top:14px;">
    <div>
      <h3>AWI &ndash; Awareness Index</h3>
      <p>Discrete head-scanning events per minute, derived from 3D nose/neck/ear keypoints.<br>
      <em>Impossible with 2D tracking. Directly measures where a player is looking.</em></p>
      <p>A scan = head rotation &ge;45&deg; within 0.5 s. One sustained turn = one event.</p>
    </div>
    <div>
      <h3>PQI &ndash; Pressure Quality Index</h3>
      <p>Pressing mechanics score [0&ndash;100] during genuine press frames (&ge;10 consecutive frames within 5 m of ball carrier).</p>
      <p>PQI = 0.40 &times; orientation + 0.30 &times; stance + 0.30 &times; proximity</p>
    </div>
  </div>

  <blockquote style="margin-top:16px;">A player can scan brilliantly but press with poor mechanics (high AWI, low PQI), or press perfectly without pre-scanning (low AWI, high PQI). Elite players score high on both &ndash; and the data shows exactly who they are.</blockquote>
</div>

<!-- SLIDE 2: TECHNICAL -->
<div class="slide">
  <div class="slide-header">
    <span class="slide-num">Slide 2 of 6</span>
    <span class="slide-title">Technical Approach</span>
  </div>
  <p class="slide-subtitle">S3 &rarr; Skeleton &rarr; Angles &rarr; AWI + PQI &rarr; Dashboard &rarr; AI Narratives</p>

<pre>TF15 Parquet (S3, ~4 GB/match)
  &rarr; pyarrow row-group pushdown        # stream, never full download
  &rarr; _extract_angles_vectorized()      # head yaw: nose/neck primary, ear fallback
                                          # body yaw: shoulder primary, hip fallback
  &rarr; detect_scans()                    # 11-frame circular smooth &rarr; 25-frame delta &rarr; &ge;45&deg; leading edge
  &rarr; compute_awi()                     # scan_count / phase_minutes
  &rarr; pqi_calculator.py                 # orientation + stance + proximity, vectorized
  &rarr; Streamlit dashboard               # Player Profile &middot; Match Overview &middot; Leaderboard &middot; Fan View
  &rarr; Amazon Bedrock (Nova Lite)        # natural-language scouting narratives</pre>

  <h3>Key engineering decisions</h3>
  <table>
    <thead><tr><th>Decision</th><th>Why it matters</th></tr></thead>
    <tbody>
      <tr><td>Circular smoothing (sin/cos decomposition)</td><td>Standard rolling mean breaks at &plusmn;180&deg;</td></tr>
      <tr><td>Leading-edge counting</td><td>1 sustained rotation = 1 event, not N frames</td></tr>
      <tr><td>45&deg; threshold (not 30&deg;)</td><td>XY-plane projection compresses 3D angles; tuned on Kimmich</td></tr>
      <tr><td>Press frame filter: &ge;10 consecutive frames within 5 m</td><td>Excludes incidental proximity; captures genuine pressing intent</td></tr>
    </tbody>
  </table>
  <p style="margin-top:10px;"><strong>Scale:</strong> 5 matches &times; ~40 players &times; 2 halves = <strong>400 rows</strong> &middot; 212 unit tests &middot; SageMaker Processing for parallel compute (~15 min)</p>
</div>

<!-- SLIDE 3: RESULTS -->
<div class="slide">
  <div class="slide-header">
    <span class="slide-num">Slide 3 of 6</span>
    <span class="slide-title">Results &ndash; AWI Leaderboard &amp; Position Patterns</span>
  </div>
  <p class="slide-subtitle">Defensive midfielders are the most cognitively active players on the pitch</p>

  <div class="two-col">
    <div>
      <h3>Top 8 player-phases by AWI</h3>
      <table>
        <thead><tr><th>#</th><th>Player</th><th>Pos</th><th>Match</th><th>AWI</th></tr></thead>
        <tbody>
          <tr><td>1</td><td>Oscar H&oslash;jlund</td><td><span class="tag">DMZ</span></td><td>SGE-FCB H1</td><td><strong>26.90</strong></td></tr>
          <tr><td>2</td><td>Hugo Larsson</td><td><span class="tag">DMZ</span></td><td>SGE-FCU H1</td><td>26.26</td></tr>
          <tr><td>3</td><td>Joshua Kimmich</td><td><span class="tag">DMR</span></td><td>FCU-FCB H1</td><td>23.38</td></tr>
          <tr><td>4</td><td>Rani Khedira</td><td><span class="tag">DMR</span></td><td>SGE-FCU H1</td><td>22.95</td></tr>
          <tr><td>5</td><td>Oscar H&oslash;jlund</td><td><span class="tag">DMZ</span></td><td>SGE-FCB H2</td><td>22.74</td></tr>
          <tr><td>6</td><td>Rani Khedira</td><td><span class="tag">DMR</span></td><td>SGE-FCU H2</td><td>22.27</td></tr>
          <tr><td>7</td><td>Aljoscha Kemlein</td><td><span class="tag">DMZ</span></td><td>FCU-FCB H2</td><td>21.89</td></tr>
          <tr><td>8</td><td>Joshua Kimmich</td><td><span class="tag">DMR</span></td><td>FCB-HSV H1</td><td>21.77</td></tr>
        </tbody>
      </table>
      <p style="margin-top:10px;"><strong>Kimmich validation:</strong> 21.77 (FCB-HSV H1) &rarr; 21.15 (H2) &rarr; 23.38 (FCU-FCB H1) &rarr; 11.29 (H2). The H2 drop of 52% is a fatigue signal no GPS metric would catch.</p>
      <p style="margin-top:6px;">Cross-half r = <strong>0.660</strong> (n = 79): AWI is a stable player trait.</p>
    </div>
    <div>
      <div class="fig">
        <img src="{imgs['heatmap']}" alt="Position AWI heatmap" />
        <p class="fig-caption">Position-level AWI heatmap across all 5 matches. DMZ leads at 15.6 scans/min; goalkeepers (TW) at 3.5.</p>
      </div>
    </div>
  </div>
</div>

<!-- SLIDE 4: THE KEY FINDING -->
<div class="slide">
  <div class="slide-header">
    <span class="slide-num">Slide 4 of 6</span>
    <span class="slide-title">The Key Finding &ndash; AWI and PQI Are Independent</span>
  </div>
  <p class="slide-subtitle">Scanning awareness and pressing mechanics are two orthogonal dimensions of player intelligence</p>

  <div class="two-col">
    <div>
      <h3>AWI vs PQI: r = &minus;0.11 (p = 0.12)</h3>
      <p>Knowing a player's scan rate tells you almost nothing about their pressing mechanics. <strong>Both metrics are needed.</strong></p>

      <h3 style="margin-top:14px;">The Body Intelligence quadrant</h3>
      <table>
        <thead><tr><th></th><th>High PQI (&ge;62.6)</th><th>Low PQI</th></tr></thead>
        <tbody>
          <tr><td><strong>High AWI (&ge;14.7)</strong></td><td style="background:#e8f5e9;">&#x1F7E2; Elite</td><td style="background:#fff9c4;">Cognitive strength</td></tr>
          <tr><td><strong>Low AWI</strong></td><td style="background:#fff9c4;">Physical strength</td><td style="background:#ffebee;">&#x1F534; Development</td></tr>
        </tbody>
      </table>
      <p style="margin-top:10px;"><strong>Elite quadrant (both &gt;75th pct):</strong> 10 unique players, 15 player-phases</p>
      <p>Top performers: H&oslash;jlund (DMZ, 26.90 / 63.9), Kimmich (DMR, 21.77 / 64.7), Kemlein (DMZ, 21.89 / 63.2), Remberg (DMR, 20.52 / 63.7), Vu&scaron;kovi&cacute; (IVZ, 18.33 / 63.7)</p>
      <p style="margin-top:8px; font-size:9pt; color:#555;">Vu&scaron;kovi&cacute; is a centre-back whose scanning profile matches a defensive midfielder &ndash; tactical versatility visible in the data.</p>
    </div>
    <div>
      <div class="fig">
        <img src="{imgs['scatter']}" alt="AWI vs PQI scatter  -  body intelligence quadrant" />
        <p class="fig-caption">AWI vs PQI scatter for all player-phases. Near-zero correlation confirms two independent dimensions. Elite quadrant (top-right) contains 10 unique players.</p>
      </div>
    </div>
  </div>
</div>

<!-- SLIDE 5: BUSINESS VALUE + CROSS-DOMAIN -->
<div class="slide">
  <div class="slide-header">
    <span class="slide-num">Slide 5 of 5</span>
    <span class="slide-title">Business Value &amp; Cross-Domain Validation</span>
  </div>
  <p class="slide-subtitle">From 141 million data points per match to two numbers every coach understands &ndash; validated across five domains</p>

  <div class="two-col">
    <div>
      <h3>For clubs</h3>
      <ul>
        <li><strong>Scouting:</strong> rank by AWI and PQI independently &ndash; find dual-elite players or target specific development gaps</li>
        <li><strong>Fatigue detection:</strong> Kimmich's AWI drops 52% in FCU-FCB 2nd half &ndash; invisible to GPS</li>
        <li><strong>Positional outliers:</strong> Vu&scaron;kovi&cacute; (CB) in the elite quadrant flags tactical versatility automatically</li>
      </ul>
      <h3>For broadcasters &amp; fans</h3>
      <ul>
        <li>Live overlay: <em>"H&oslash;jlund scanned 27 times before that interception"</em></li>
        <li>Fan View tab: broadcast-style top-3 counter, quadrant classification, Body Intelligence leaderboard</li>
      </ul>
      <h3>For the DFL</h3>
      <ul>
        <li>First matchday-grade metrics derived solely from TF15 skeleton data</li>
        <li>Differentiates the 3D product from any 2D competitor</li>
      </ul>
    </div>
    <div>
      <h3>Cross-domain validation &ndash; every concept has a proven ancestor</h3>
      <table style="font-size:9pt;">
        <thead><tr><th>External System</th><th>Maps to</th></tr></thead>
        <tbody>
          <tr><td><strong>NFL Next Gen Stats</strong> <span style="color:#666;font-weight:normal;">(Eager et al., 2023, MIT Sloan)</span></td><td>PQI proximity sub-score (distance-decay scoring)</td></tr>
          <tr><td><strong>NBA Second Spectrum</strong> <span style="color:#666;font-weight:normal;">(Cervone et al., 2016, JASA)</span></td><td>PQI orientation sub-score (EPV defensive positioning)</td></tr>
          <tr><td><strong>Tennis biomechanics</strong> <span style="color:#666;font-weight:normal;">(Elliott, 2006, BJSM)</span></td><td>PQI stance sub-score (optimal knee-flexion deviation)</td></tr>
          <tr><td><strong>Occupational biomechanics REBA</strong> <span style="color:#666;font-weight:normal;">(Hignett &amp; McAtamney, 2000, Appl. Ergon.)</span></td><td>PQI stance Gaussian penalty (joint-angle deviation structure)</td></tr>
          <tr><td><strong>Rugby 3D motion capture</strong> <span style="color:#666;font-weight:normal;">(Hendricks et al., 2021, Sports Med. Open)</span></td><td>PQI composite (three-factor tackle biomechanics)</td></tr>
          <tr><td><strong>Cockpit visual scanning</strong> <span style="color:#666;font-weight:normal;">(Lounis et al., 2021, PLOS ONE)</span></td><td>AWI (scan-transition rate as situational awareness proxy)</td></tr>
        </tbody>
      </table>
      <blockquote style="margin-top:10px; font-size:9pt;">The pre-pass AWI spike (~19.2 scans/min) reaches the aviation elite level (19 scans/min, Lounis et al., 2021). TF15 makes this measurement possible at matchday scale for the first time in team sports.</blockquote>
    </div>
  </div>
</div>

</body>
</html>"""

pathlib.Path('submission/executive_summary_slides.html').write_text(SLIDES, encoding='utf-8')
size = pathlib.Path('submission/executive_summary_slides.html').stat().st_size // 1024
print(f'executive_summary_slides.html written: {size} KB')

# ── PRFAQ ─────────────────────────────────────────────────────────────────────

def build_prfaq():
    PRFAQ_CSS = """
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; color: #1a1a1a;
       background: #ffffff; max-width: 860px; margin: 40px auto; padding: 0 40px; line-height: 1.6; }
h1 { font-size: 22pt; color: #e30613; border-bottom: 3px solid #e30613; padding-bottom: 10px; margin-bottom: 6px; }
h2 { font-size: 14pt; color: #e30613; margin-top: 32px; border-bottom: 1px solid #f5c0c3; padding-bottom: 6px; }
h3 { font-size: 11pt; font-weight: 700; margin-top: 20px; color: #1a1a1a; }
p { margin: 8px 0; }
ul, ol { padding-left: 22px; margin: 8px 0; }
li { margin-bottom: 5px; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 10pt; }
th { background: #e30613; color: #fff; padding: 7px 11px; text-align: left; }
td { border: 1px solid #ddd; padding: 6px 11px; color: #1a1a1a; }
tr:nth-child(even) td { background: #fdf5f5; }
blockquote { border-left: 4px solid #e30613; margin: 14px 0; padding: 8px 16px;
             color: #444; font-style: italic; background: #fff5f5; border-radius: 0 4px 4px 0; }
code { background: #f5f5f5; color: #c0000a; padding: 2px 5px; font-size: 9pt; font-family: Consolas, monospace; border-radius: 2px; }
hr { border: none; border-top: 2px solid #e30613; margin: 28px 0; opacity: 0.3; }
a { color: #e30613; }
.fig { text-align: center; margin: 20px 0; }
.fig img { width: 100%; max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; display: block; margin: 0 auto; }
.fig-caption { font-size: 8.5pt; color: #777; margin-top: 5px; }
img { width: 100%; max-width: 100%; height: auto; border: 1px solid #e0e0e0; border-radius: 4px; display: block; margin: 12px auto; }
@media print {
  body { -webkit-print-color-adjust: exact; print-color-adjust: exact; margin: 20px; }
  .screen-footer { display: none; }
}
@page {
  margin: 15mm 20mm 20mm 20mm;
  @bottom-center {
    content: "Football Body Intelligence Platform  ·  Page " counter(page) " of " counter(pages);
    font-size: 8pt;
    color: #777;
    font-family: 'Segoe UI', Arial, sans-serif;
  }
}
"""

    raw = pathlib.Path('submission/prfaq.md').read_text(encoding='utf-8')
    body = md.markdown(raw, extensions=['tables', 'fenced_code'])
    # Embed scatter figure as base64 so prfaq.html is self-contained (no relative path dependency)
    body = body.replace(
        'src="../figures/fig1_awi_vs_pqi_scatter.png"',
        f'src="{imgs["scatter"]}"',
    )

    # Inject consistency figure after the cross-half correlation paragraph
    consistency_tag = (
        '\n<div class="fig">'
        f'<img src="{imgs["half_compare"]}" alt="Cross-half AWI consistency">'
        '<p class="fig-caption">1st half vs 2nd half AWI &ndash; r = 0.660, p &lt; 0.001 across all 5 matches</p>'
        '</div>\n'
    )
    # Insert after the sentence containing "r = 0.660"
    body = body.replace(
        'We also observe cross-match stability for players who appear in multiple games (Kimmich, Goretzka, Kimmich). The metric captures individual cognitive style, not just tactical instructions.</p>',
        'We also observe cross-match stability for players who appear in multiple games (Kimmich, Goretzka, Kimmich). The metric captures individual cognitive style, not just tactical instructions.</p>'
        + consistency_tag
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>AWI PRFAQ</title><style>{PRFAQ_CSS}</style></head>
<body>{body}
<footer class="screen-footer" style="margin-top:48px; padding:12px 0; border-top:2px solid #e30613;
  text-align:center; font-size:8.5pt; color:#888; font-family:'Segoe UI',Arial,sans-serif;">
  Football Body Intelligence Platform &nbsp;&middot;&nbsp; AWS World Sports Innovation Cup 2026 &nbsp;&middot;&nbsp; Team: itzmore
</footer>
</body></html>"""

    pathlib.Path('submission/prfaq.html').write_text(html, encoding='utf-8')
    size = pathlib.Path('submission/prfaq.html').stat().st_size // 1024
    print(f'prfaq.html written: {size} KB')

build_prfaq()
