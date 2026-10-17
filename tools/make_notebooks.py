"""Generate the weekly demo notebooks (Hebrew RTL markdown + English code), then execute them
so that outputs are stored in the .ipynb (the Jupyter Book site shows them without re-running).
usage: python tools/make_notebooks.py

The notebooks keep an RTL <div> wrapper in every markdown cell (right-to-left in Colab / local Jupyter).
The site build strips it first (tools/prepare_site.py), because MyST flattens math inside raw HTML.
"""
import json, pathlib
import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
H = '<div dir="rtl" style="text-align:right">\n\n'
F = '\n\n</div>'

REPO = "yaminbens/SolidState"

def header(week, title, kicker):
    """First cell: title (becomes the page title), kicker line, Colab badge and download link."""
    nm = f"week{week:02d}.ipynb"
    return f"""
# {title}

*{kicker}*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{nm}) &nbsp;&nbsp; [⬇ הורדת המחברת](https://github.com/{REPO}/raw/main/{nm})

> 🧭 **איך עובדים עם המחברת?** ראשית, לפני כל גרף, נסו לנחש מה תראו. שנית, הריצו (*Run all*, או תא אחר תא); כל המחברת רצה בתוך דקה, ב-Colab או מקומית (`numpy`, `scipy`, `matplotlib`). לבסוף, ענו לעצמכם על שאלות הדיון שבסוף. אין צורך לכתוב קוד: הסעיף החישובי בתרגיל הבית נשען על ההדגמה הזו.
"""

def nb_from(cells):
    nb = nbformat.v4.new_notebook()
    for kind, src in cells:
        src = src.strip("\n")
        nb.cells.append(nbformat.v4.new_markdown_cell(H + src + F) if kind == "md"
                        else nbformat.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb

# ------------------------------------------------------------------ week 1
w1 = [
("md", header(1, "שבוע 1 · פוטנציאל כימי כתלות בטמפרטורה", "הדגמה לתרגול 1 · מבוא למצב מוצק תשפ״ז · אוניברסיטת תל אביב") + r"""
## מה עושים כאן?
בטמפרטורה סופית האכלוס נקבע על ידי $f_{FD}$, והפוטנציאל הכימי $\mu$ נקבע מהדרישה שמספר החלקיקים נשאר קבוע:
$$n=\int_0^\infty g(\varepsilon)\,f_{FD}(\varepsilon;\mu,T)\,d\varepsilon .$$
זו משוואה לא-ליניארית עבור $\mu(T)$, והמחברת פותרת אותה מספרית (מציאת שורש) בכל טמפרטורה.

**יחידות:** $\varepsilon_F=1$, $k_B=1$ (כלומר $T$ נמדד ב-$T_F$) ו-$n=1$.
"""),
("code", """
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit   # expit(x) = 1/(1+exp(-x)), numerically stable
"""),
("md", r"""
### ⚙️ פרמטרים
- `d` – המימד של הגז. צפיפות המצבים היא $g(\varepsilon)=c\,\varepsilon^{d/2-1}$, כאשר הקבוע $c$ נקבע כך שב-$T=0$ מתקבל $n=\int_0^{\varepsilon_F}g\,d\varepsilon=1$ (כלומר $c=d/2$).
- `T_max` – הטמפרטורה הגבוהה ביותר בגרפים, ביחידות של $T_F$.
"""),
("code", """
# ---- Parameters ----
d = 2          # dimension of the gas
T_max = 2.0    # highest temperature shown (units of T_F)
"""),
("md", r"""
### 1. התפלגות פרמי–דיראק וצפיפות המצבים
"""),
("code", """
def f_FD(e, mu, T):
    # Fermi-Dirac occupation, written with expit to avoid overflow
    return expit(-(e - mu) / T)

def g(e, d):
    # density of states ~ e^(d/2 - 1); the constant d/2 makes n = 1 at T = 0 (eps_F = 1)
    return (d / 2) * e ** (d / 2 - 1)
"""),
("md", r"""
### 2. פתרון מספרי של $\mu(T)$
בכל טמפרטורה מחפשים את ה-$\mu$ שעבורו האינטגרל שווה ל-$n=1$. האינטגרל עולה עם $\mu$, ולכן מציאת השורש אמינה.
"""),
("code", """
def n_of_mu(mu, T, d):
    return quad(lambda e: g(e, d) * f_FD(e, mu, T), 0, np.inf, limit=200)[0]

def solve_mu(T, d, n=1.0):
    # root of n(mu) - n = 0; n(mu) is increasing in mu, so brentq is safe
    return brentq(lambda m: n_of_mu(m, T, d) - n, -60 * max(T, 0.02), 3.0, xtol=1e-12)

def mu_curve(Ts, d):
    return np.array([solve_mu(T, d) for T in Ts])
"""),
("md", r"""
### 3. ההתפלגות בשלוש טמפרטורות
שימו לב: ב-$T\ll T_F$ ההתפלגות כמעט מדרגה, ו-$\mu$ קרוב ל-$\varepsilon_F$.
"""),
("code", """
x = np.linspace(0, 2, 600)
for T, ls in [(0.01, "-"), (0.1, "--"), (0.5, ":")]:
    mu = solve_mu(T, d)
    plt.plot(x, f_FD(x, mu, T), ls, label=f"T/T_F = {T},  mu = {mu:.4f}")
plt.axvline(1, color="0.7", lw=0.8)
plt.xlabel("epsilon / epsilon_F"); plt.ylabel("f_FD"); plt.title(f"d = {d}")
plt.legend(frameon=False); plt.show()
"""),
("md", r"""
### 4. $\mu(T)$
> 🔮 **נחשו:** כשמעלים את הטמפרטורה, האם $\mu$ עולה, יורד או נשאר קבוע? נמקו במשפט אחד לפני שמסתכלים על הגרף.
"""),
("code", """
Ts = np.linspace(0.02, T_max, 100)
plt.plot(Ts, mu_curve(Ts, d), label=f"d = {d}")
plt.axhline(1, color="0.7", lw=0.8); plt.axhline(0, color="0.7", lw=0.8)
plt.xlabel("T / T_F"); plt.ylabel("mu / epsilon_F"); plt.legend(frameon=False)
plt.show()
"""),
("md", r"""
### 💬 שאלות לדיון
1. מהי ההתנהגות של $\mu$ ב-$T\ll T_F$? האם השינוי ביחס ל-$\varepsilon_F$ גדול או קטן?
2. מה קורה ל-$\mu$ כש-$T\gg T_F$? מה הסימן שלו, ולמה? (רמז: הגבול הקלאסי.)
3. מה מייחד את צפיפות המצבים בסביבת $\varepsilon_F$, ואיך זה קשור להתנהגות שראיתם?

> ✅ **סיימתם?** תשובות לשאלות הדיון אינן להגשה. הסעיף החישובי בתרגיל הבית נשען על המחברת הזו. בהצלחה!
"""),
]

# ------------------------------------------------------------------ week 2
w2 = [
("md", header(2, "שבוע 2 · סימולציית דרודה", "הדגמה לתרגול 2 · מבוא למצב מוצק תשפ״ז · אוניברסיטת תל אביב") + r"""
## מה עושים כאן?
מדמים נושאי מטען (למשל אלקטרונים) בשני ממדים. בין פיזורים הם נעים תחת כוח לורנץ, ובכל פיזור, שמתרחש בהסתברות $dt/\tau$ בכל צעד, הם "שוכחים" את מהירותם ומקבלים מהירות חדשה מהתפלגות בולצמן. זה בדיוק המודל מהתרגול.

**יחידות:** $m=\tau=k_BT=c=1$, ומטען נמדד ביחידות של $|e|$. השדה המגנטי בכיוון $\hat z$, והשדה החשמלי במישור.
"""),
("code", """
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(1)
"""),
("md", r"""
### ⚙️ פרמטרים
- `charge` – מטען נושא המטען ביחידות של $|e|$ (אלקטרון: `-1`).
- `E0` – גודל השדה החשמלי.
- `N` – מספר נושאי המטען בסימולציה (יותר – פחות רעש, אבל איטי יותר).
"""),
("code", """
# ---- Parameters ----
charge = -1     # carrier charge in units of |e|
E0 = 0.3        # electric-field amplitude
N = 8000        # number of simulated carriers
"""),
("md", r"""
### הסימולציה
בכל צעד זמן $dt$: (1) המהירות מסתובבת בזווית הציקלוטרון ומתקבלת תוספת מהשדה החשמלי; (2) כל נושא מטען מתפזר בהסתברות $dt/\tau$ ומקבל מהירות אקראית חדשה.
"""),
("code", """
def simulate(E, B, q=charge, N=N, dt=0.02, T=12.0):
    \"\"\"Steady-state mean velocity of N carriers (charge q) in constant fields E=(Ex,Ey), B along z.\"\"\"
    v = rng.normal(size=(N, 2))                       # Maxwell-Boltzmann at t = 0 (k_B T = 1)
    c, s = np.cos(-q * B * dt), np.sin(-q * B * dt)   # rotation angle per step
    acc, cnt = np.zeros(2), 0
    for step in range(int(T / dt)):
        vx = c * v[:, 0] - s * v[:, 1] + q * E[0] * dt
        vy = s * v[:, 0] + c * v[:, 1] + q * E[1] * dt
        v = np.stack([vx, vy], axis=1)
        hit = rng.random(N) < dt                      # scattering probability dt/tau (tau = 1)
        v[hit] = rng.normal(size=(hit.sum(), 2))      # forget the old velocity
        if step * dt > T / 2:                         # average over the second half (steady state)
            acc += v.mean(axis=0); cnt += 1
    return acc / cnt
"""),
("md", r"""
### 1. מהירות סחיפה כפונקציה של השדה
> 🔮 **נחשו:** איזו מהירות סחיפה תצפו לקבל עבור $E=0.3$ (ביחידות המחברת), ואיך היא תשתנה כשמכפילים את השדה?

זרם החלקיקים הממוצע נקבע לפי $\langle v\rangle=q\,E\,\tau/m$. האם התלות בשדה לינארית (חוק אוהם)?
"""),
("code", """
Es = np.linspace(0, 0.6, 7)
v_sim = [simulate((E, 0.0), 0.0, N=N // 2)[0] for E in Es]
plt.plot(Es, v_sim, 'o', label='simulation')
plt.plot(Es, charge * Es, '--', label='q E tau / m')
plt.xlabel('E'); plt.ylabel('<v_x>'); plt.legend(frameon=False); plt.show()
"""),
("md", r"""
### 2. זווית הול
> 🔮 **נחשו:** כשהשדה המגנטי גדל, לאיזה כיוון ובאיזו מידה ייטה הזרם ביחס לשדה החשמלי?

מפעילים שדה חשמלי בכיוון $x$ ושדה מגנטי $B$ (כאן $B=\omega_c\tau$, כי $m=q=\tau=1$). הזרם $\mathbf J=q n\langle\mathbf v\rangle$ אינו בהכרח בכיוון $\mathbf E$. מודדים את הזווית בין $\mathbf J$ ל-$\mathbf E$ ומשווים לתחזית של דרודה מהתרגול.
"""),
("code", """
Bs = np.linspace(0, 3, 7)
angle = []
for B in Bs:
    v = simulate((E0, 0.0), B)
    J = charge * v                                     # current ~ q <v>
    angle.append(np.degrees(np.arctan2(J[1], J[0])))   # angle between J and E (E along x)
Bgrid = np.linspace(0, 3, 100)
plt.plot(Bs, angle, 'o', label='simulation')
plt.plot(Bgrid, np.degrees(-charge * np.arctan(Bgrid)), '--', label='Drude')
plt.xlabel('omega_c tau'); plt.ylabel('angle between J and E [deg]')
plt.axhline(0, color='0.7', lw=0.8); plt.legend(frameon=False); plt.show()
"""),
("md", r"""
### 3. תגובה לשדה חשמלי תלוי בזמן
> 🔮 **נחשו:** מה יקרה לגודל התגובה ולפאזה שלה ביחס לשדה כשהתדר עולה?

מפעילים שדה $E(t)=E_0\cos\omega t$ (ללא שדה מגנטי), ועוקבים אחרי המהירות הממוצעת $\langle v_x\rangle$ כפונקציה של הזמן, עבור שלושה תדרים. הקו המקווקו הוא הפרופיל של השדה עצמו (בקנה מידה). כשהתדר נמוך ($\omega\tau\ll1$) המהירות עוקבת אחרי השדה; מה קורה כשהוא עולה?
"""),
("code", """
def ac_response(omega, T=40.0, dt=0.02, N=N):
    v = rng.normal(size=N)
    t = np.arange(int(T / dt)) * dt
    out = np.empty_like(t)
    for k, tk in enumerate(t):
        v = v + charge * E0 * np.cos(omega * tk) * dt
        hit = rng.random(N) < dt
        v[hit] = rng.normal(size=hit.sum())
        out[k] = v.mean()
    return t, out

fig, axs = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
for ax, omega in zip(axs, [0.3, 1.0, 3.0]):
    t, vx = ac_response(omega)
    ax.plot(t, vx, label='<v_x>')
    ax.plot(t, charge * E0 * np.cos(omega * t), 'k--', lw=0.8, label='field (scaled)')
    ax.set_title(f'omega tau = {omega}'); ax.set_xlabel('t / tau')
axs[0].legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()
"""),
("md", r"""
### 💬 שאלות לדיון
1. מה קובע את שיפוע הגרף בחלק 1? מה היה משתנה אילו הזמן $\tau$ בין פיזורים היה ארוך יותר?
2. בחלק 2: מה קורה לזווית כש-$\omega_c\tau$ גדל, ולמה היא לא עוברת $90^\circ$?
3. בחלק 3: מה קורה לגודל התגובה ולמופע שלה ביחס לשדה כשהתדר עולה? מדוע (חשבו על $\tau$ כזמן שלוקח לנושא המטען "להגיב")? קשרו לביטוי $\sigma(\omega)=\sigma_D/(1-i\omega\tau)$ מהתרגול.

> ✅ **סיימתם?** תשובות לשאלות הדיון אינן להגשה. הסעיף החישובי בתרגיל הבית נשען על המחברת הזו. בהצלחה!
"""),
]

def build(name, cells, out):
    nb = nb_from(cells)
    NotebookClient(nb, timeout=600, kernel_name="python3").execute()
    nbformat.write(nb, out)
    print("wrote", out)

if __name__ == "__main__":
    build("w1", w1, ROOT / "week01.ipynb")
    build("w2", w2, ROOT / "week02.ipynb")
