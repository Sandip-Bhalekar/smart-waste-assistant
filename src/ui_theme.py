APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --sw-bg: #f3f7f4;
  --sw-card: #ffffff;
  --sw-ink: #1c2b24;
  --sw-muted: #5b6b63;
  --sw-green: #1f7a4d;
  --sw-green-soft: #e7f5ec;
  --sw-amber: #8a5a12;
  --sw-amber-soft: #fff4e0;
  --sw-red: #9b2c2c;
  --sw-red-soft: #fdecec;
  --sw-line: #d7e3db;
}

html, body, [class*="css"] {
  font-family: 'Manrope', sans-serif;
}

.stApp {
  background: radial-gradient(circle at top right, #e8f4ec 0%, #f3f7f4 38%, #f7f4ef 100%);
}

.block-container {
  padding-top: 1.2rem;
  max-width: 1180px;
}

header[data-testid="stHeader"] {
  background: transparent;
}

.sw-hero {
  background: linear-gradient(135deg, #16382a 0%, #1f7a4d 62%, #3d9b66 100%);
  color: white;
  border-radius: 24px;
  padding: 28px 32px;
  box-shadow: 0 18px 40px rgba(22, 56, 42, 0.18);
  margin-bottom: 1.4rem;
}

.sw-hero h1 {
  margin: 0;
  font-size: 2.1rem;
  letter-spacing: -0.03em;
}

.sw-hero p {
  margin: 0.45rem 0 0;
  color: rgba(255,255,255,0.88);
  max-width: 640px;
}

.sw-chip {
  display: inline-block;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.78rem;
  margin-bottom: 12px;
}

.sw-card {
  background: var(--sw-card);
  border: 1px solid var(--sw-line);
  border-radius: 18px;
  padding: 18px 18px 14px;
  box-shadow: 0 8px 24px rgba(28, 43, 36, 0.05);
  margin-bottom: 1rem;
}

.sw-kicker {
  color: var(--sw-green);
  font-weight: 700;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.35rem;
}

.sw-metric {
  background: var(--sw-green-soft);
  border-radius: 16px;
  padding: 16px;
  border: 1px solid #cfe6d7;
}

.sw-metric b {
  display: block;
  font-size: 1.6rem;
  color: var(--sw-ink);
}

.sw-metric span {
  color: var(--sw-muted);
  font-size: 0.86rem;
}

.sw-warn {
  background: var(--sw-amber-soft);
  border: 1px solid #f0d7a4;
  color: var(--sw-amber);
  border-radius: 14px;
  padding: 12px 14px;
}

.sw-empty {
  background: #f8faf8;
  border: 1px dashed var(--sw-line);
  border-radius: 16px;
  padding: 22px;
  color: var(--sw-muted);
}

.sw-footer {
  margin-top: 1.5rem;
  padding: 18px;
  color: var(--sw-muted);
  font-size: 0.86rem;
  border-top: 1px solid var(--sw-line);
}

div.stButton > button {
  border-radius: 12px;
  font-weight: 700;
}

div.stButton > button[kind="primary"] {
  background: #1f7a4d;
  border-color: #1f7a4d;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
}
</style>
"""
