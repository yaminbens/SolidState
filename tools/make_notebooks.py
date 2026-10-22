"""Generate the weekly demo notebooks (Hebrew RTL markdown + English code), then check that they run
(the saved notebooks are unrun, so students see no results until they run the code themselves).
usage: python tools/make_notebooks.py

The notebooks keep an RTL <div> wrapper in every markdown cell (right-to-left in Colab / local Jupyter).
The site build strips it first (tools/prepare_site.py), because MyST flattens math inside raw HTML.
"""
import json, pathlib
import copy
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

> 🧭 **איך עובדים עם המחברת?** ראשית, לפני כל גרף, נסו לנחש מה תראו. שנית, הריצו את הקוד (*Run all*, או תא אחר תא), באתר או ב-Colab; כל המחברת רצה בתוך דקה (`numpy`, `scipy`, `matplotlib`). לבסוף, ענו לעצמכם על שאלות הדיון שבסוף. אין צורך לכתוב קוד: הסעיף החישובי בתרגיל הבית נשען על ההדגמה הזו.
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
## תיאור ההדגמה
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
### 4. הגרף של $\mu(T)$
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
## תיאור ההדגמה
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

# ------------------------------------------------------------------ week 3
w3 = [
("md", header(3, "שבוע 3 · קיבול החום של מתכת", "הדגמה לתרגול 3 · מבוא למצב מוצק תשפ״ז · אוניברסיטת תל אביב") + r"""
## תיאור ההדגמה
קיבול החום של מתכת הוא סכום של שתי תרומות: תנודות הסריג (פונונים) והאלקטרונים. המחברת מחשבת כל אחת מהן בנפרד:

- **סריג:** מודל Debye, ולשם השוואה גם מודל איינשטיין (כל האופנים באותה תדירות).
- **אלקטרונים:** גז פרמי חופשי בשלושה ממדים. הפוטנציאל הכימי נקבע בכל טמפרטורה מהדרישה שמספר האלקטרונים קבוע, ואז מחשבים את האנרגיה ואת הנגזרת שלה לפי $T$. החישוב מדויק, ובטמפרטורות נמוכות הוא מתלכד עם התוצאה של זומרפלד, $c=\gamma T$.

**יחידות:** קיבול חום למול אטומים ב-$\mathrm{J/(mol\,K)}$, וטמפרטורה בקלווין. ערך דולון-פטי הוא $3R\approx24.9\ \mathrm{J/(mol\,K)}$.
"""),
("code", """
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit   # expit(x) = 1/(1+exp(-x)), numerically stable

R = 8.314   # gas constant, J/(mol K)
"""),
("md", r"""
### ⚙️ פרמטרים
ערכי ברירת המחדל מתאימים לנחושת.

- `T_F` – טמפרטורת פרמי של האלקטרונים, $T_F=\varepsilon_F/k_B$ (קלווין).
- `T_D` – טמפרטורת Debye של הסריג (קלווין).
- `z` – מספר האלקטרונים החופשיים לאטום.
- `T_E` – טמפרטורת איינשטיין, להשוואה בחלק 1 בלבד (קלווין).
- `T_low` – הטמפרטורה הגבוהה ביותר בגרף של $c/T$ כפונקציה של $T^2$ (חלק 3).
- `T_max` – הטמפרטורה הגבוהה ביותר בגרפים של חלקים 2 ו-4.
"""),
("code", """
# ---- Parameters ----
T_F = 8.16e4    # Fermi temperature of the electrons (K)
T_D = 343.0     # Debye temperature (K)
z = 1           # free electrons per atom
T_E = 266.0     # Einstein temperature, only for the comparison in part 1 (K)
T_low = 6.0     # highest temperature in the c/T vs T^2 plot (K)
T_max = 300.0   # highest temperature in the plots of parts 2 and 4 (K)
"""),
("md", r"""
### 1. הסריג: Debye מול איינשטיין
בשני המודלים יש $3$ אופני תנודה לאטום, ובטמפרטורה גבוהה שניהם נותנים $3R$. ההבדל הוא בתדירויות: באיינשטיין לכולם אותה תדירות $\omega_E$, וב-Debye התדירויות פרושות מאפס עד $\omega_D$.

> 🔮 **נחשו:** בגרף לוגריתמי, איזה מהמודלים יורד מהר יותר כשמקררים? מה יהיה השיפוע של עקומת Debye?
"""),
("code", """
def c_debye(T, T_D):
    # Debye heat capacity per mole of atoms (integrand written to avoid overflow)
    f = lambda x: x**4 * np.exp(-x) / (-np.expm1(-x))**2
    T = np.atleast_1d(np.asarray(T, dtype=float))
    return np.array([9 * R * (t / T_D)**3 * quad(f, 0, min(T_D / t, 200.0))[0] for t in T])

def c_einstein(T, T_E):
    # all 3 modes per atom at the same frequency omega_E = k_B T_E / hbar
    x = T_E / np.asarray(T, dtype=float)
    return 3 * R * x**2 * np.exp(-x) / (-np.expm1(-x))**2

T = np.geomspace(0.01 * T_D, 2 * T_D, 300)
plt.loglog(T / T_D, c_debye(T, T_D) / (3 * R), "-", label="Debye")
plt.loglog(T / T_D, c_einstein(T, T_E) / (3 * R), "--", label=f"Einstein, T_E = {T_E:g} K")
plt.ylim(1e-6, 2)
plt.xlabel("T / T_D"); plt.ylabel("c / 3R"); plt.legend(frameon=False)
plt.show()
"""),
("md", r"""
### 2. האלקטרונים מול הסריג
> 🔮 **נחשו:** האלקטרונים מתנהגים כמו $T$ והסריג כמו $T^3$. באיזה צד של נקודת החיתוך כל אחד מהם שולט?
"""),
("code", """
def mu_of_t(t):
    # chemical potential (units of eps_F) of a 3D free-electron gas at t = T/T_F; g(e) = 1.5 sqrt(e)
    def n(mu):
        top = max(mu, 0.0) + 40 * t
        return quad(lambda e: 1.5 * np.sqrt(e) * expit(-(e - mu) / t), 0, top, limit=200)[0]
    return brentq(lambda mu: n(mu) - 1.0, -50 * t - 1.0, 1.5, xtol=1e-13)

def u_of_t(t):
    # energy per electron (units of eps_F)
    mu = mu_of_t(t)
    top = max(mu, 0.0) + 40 * t
    return quad(lambda e: 1.5 * e**1.5 * expit(-(e - mu) / t), 0, top, limit=200)[0]

# tabulate c per electron, d u / d t (units of k_B), once
t_grid = np.geomspace(0.002, 5.0, 180)
u_grid = np.array([u_of_t(t) for t in t_grid])
phi_grid = np.gradient(u_grid, t_grid)

def c_electrons(T, T_F, z=1):
    # electronic heat capacity per mole of atoms; below t = 0.005 the Sommerfeld result is exact enough
    t = np.asarray(T, dtype=float) / T_F
    phi = np.where(t < 0.005, np.pi**2 / 2 * t, np.interp(t, t_grid, phi_grid))
    return z * R * phi

# temperature where the two contributions are equal
T_star = brentq(lambda T: c_electrons(T, T_F, z) - c_debye(T, T_D)[0], 0.01, T_max)
print(f"T* = {T_star:.2f} K")

T = np.geomspace(0.5, T_max, 300)
plt.loglog(T, c_electrons(T, T_F, z), "-", label="electrons")
plt.loglog(T, c_debye(T, T_D), "--", label="lattice (Debye)")
plt.loglog(T, c_electrons(T, T_F, z) + c_debye(T, T_D), ":", color="k", label="total")
plt.axvline(T_star, color="0.7", lw=0.8)
plt.xlabel("T (K)"); plt.ylabel("c (J / mol K)"); plt.legend(frameon=False)
plt.show()
"""),
("md", r"""
### 3. הגרף של $c/T$ כפונקציה של $T^2$
בטמפרטורות נמוכות $c\approx\gamma T+\beta T^3$, ולכן $c/T\approx\gamma+\beta T^2$ הוא קו ישר כפונקציה של $T^2$. הקו המקווקו הוא הקו התיאורטי, עם $\gamma=\tfrac{\pi^2}{2}zR/T_F$ מזומרפלד ו-$\beta=\tfrac{12\pi^4}{5}R/T_D^3$ מ-Debye.

> 🔮 **נחשו:** מה קובע את נקודת החיתוך עם הציר האנכי, ומה קובע את השיפוע?
"""),
("code", """
gamma = np.pi**2 / 2 * z * R / T_F
beta = 12 * np.pi**4 / 5 * R / T_D**3
print(f"gamma = {1e3 * gamma:.4g} mJ/(mol K^2),  beta = {1e3 * beta:.4g} mJ/(mol K^4),  T* = {T_star:.2f} K")

T = np.linspace(0.05 * T_low, T_low, 200)
c_tot = c_electrons(T, T_F, z) + c_debye(T, T_D)
plt.plot(T**2, 1e3 * c_tot / T, "-", label="computed")
plt.plot(T**2, 1e3 * (gamma + beta * T**2), "--", label="gamma + beta T^2")
plt.xlim(0, T_low**2); plt.ylim(0, None)
plt.xlabel("T^2 (K^2)"); plt.ylabel("c / T (mJ / mol K^2)"); plt.legend(frameon=False)
plt.show()
"""),
("md", r"""
### 4. כל תחום הטמפרטורות
> 🔮 **נחשו:** בטמפרטורת החדר, איזה חלק מקיבול החום הוא אלקטרוני?
"""),
("code", """
T = np.linspace(1.0, T_max, 300)
c_el, c_lat = c_electrons(T, T_F, z), c_debye(T, T_D)
plt.plot(T, (c_el + c_lat) / (3 * R), "-", label="total")
plt.plot(T, c_lat / (3 * R), "--", label="lattice (Debye)")
plt.plot(T, c_el / (3 * R), ":", label="electrons")
plt.axhline(1, color="0.7", lw=0.8)
plt.xlabel("T (K)"); plt.ylabel("c / 3R"); plt.legend(frameon=False)
plt.show()
print(f"at T = {T_max:g} K: electrons / total = {c_el[-1] / (c_el[-1] + c_lat[-1]):.4f}")
"""),
("md", r"""
### 💬 שאלות לדיון
1. למה קיבול החום במודל איינשטיין יורד אקספוננציאלית בקירור, ואילו ב-Debye רק כחזקה של $T$? (רמז: כמה אופנים פעילים בטמפרטורה $T$ בכל מודל?)
2. האלקטרונים במתכת רבים כמו האטומים. למה התרומה שלהם בטמפרטורת החדר כמעט אינה נראית?
3. איזה מידע על האלקטרונים ועל הסריג אפשר לקרוא מנקודת החיתוך ומהשיפוע בגרף של חלק 3?

<details>
<summary>🗝️ הכוונה לתשובות (לחצו לפתיחה)</summary>

1. תרגול 3, סעיף 3: ספירת דרגות החופש הפעילות, הפסקאות על איינשטיין ועל Debye והטבלה שבסוף הסעיף.
2. תרגול 3, סעיף 3 (הפסקה על האלקטרונים ועקרון פאולי), ותרגיל 2 בסעיף 4 (החלק השלישי שלו).
3. תרגול 3, סעיף 4 (הנוסחה $c/T=\gamma+\beta T^2$) ותרגיל 2.

אם אינכם יודעים לענות על אחת השאלות, חזרו לסעיף המתאים בתרגול.

</details>

> ✅ **סיימתם?** תשובות לשאלות הדיון אינן להגשה. הסעיף החישובי בתרגיל הבית נשען על המחברת הזו. בהצלחה!
"""),
]

# ------------------------------------------------------------------ week 4
w4 = [
("md", header(4, "שבוע 4 · בונים גביש: סריג ובסיס", "הדגמה לתרגול 4 · מבוא למצב מוצק תשפ״ז · אוניברסיטת תל אביב") + r"""
## תיאור ההדגמה
גביש הוא סריג ברווה, $\mathbf R=n_1\mathbf a_1+n_2\mathbf a_2+n_3\mathbf a_3$ עם $n_i$ שלמים, ועל כל נקודת סריג אותו בסיס של אטומים במיקומים $\mathbf d_j$. מיקומי כל האטומים הם $\mathbf R+\mathbf d_j$. המחברת בונה חתיכה גדולה של גביש מהווקטורים הפרימיטיביים ומהבסיס, ומחשבת ממנה שלושה דברים:
1. נפח התא הפרימיטיבי $v_p=|\mathbf a_1\cdot(\mathbf a_2\times\mathbf a_3)|$, ומספר האטומים בקובייה של צלע $a$;
2. קליפות השכנים: לכל מרחק $r$, כמה אטומים נמצאים בדיוק במרחק הזה מאטום נתון, והפונקציה $N(r)$ שסופרת את האטומים במרחק של עד $r$;
3. יחס האריזה.

**יחידות:** אורכים נמדדים ביחידות של צלע הקובייה $a$.

לצד המחברת יש גם [דף אינטראקטיבי](https://yaminbens.github.io/SolidState/week04-crystals) שמציג את המבנים בתלת-ממד.
"""),
("code", """
import itertools
import numpy as np
import matplotlib.pyplot as plt
"""),
("md", r"""
### ⚙️ פרמטרים
- `a1`, `a2`, `a3` – הווקטורים הפרימיטיביים, ביחידות של $a$. ברירת המחדל היא סריג BCC.
- `basis` – רשימת וקטורי הבסיס: המיקומים של האטומים ביחס לכל נקודת סריג. לסריג ברווה יש אטום אחד בבסיס, בראשית.
- `r_max` – המרחק הגדול ביותר שמוצג בניתוח השכנים.
"""),
("code", """
# ---- Parameters ----
a1 = [-0.5, 0.5, 0.5]       # primitive vectors (units of the cube edge a)
a2 = [0.5, -0.5, 0.5]
a3 = [0.5, 0.5, -0.5]
basis = [[0.0, 0.0, 0.0]]   # positions of the atoms attached to every lattice point
r_max = 1.6                 # largest distance in the neighbour analysis (units of a)
"""),
("md", r"""
### 1. בניית הגביש וספירת האטומים בקובייה
יוצרים את כל וקטורי הסריג עם $|n_i|\le 6$, ומוסיפים לכל אחד מהם את וקטורי הבסיס. את האטומים בקובייה סופרים בתוך הקובייה ה"חצי-פתוחה" $0\le x,y,z<a$: אטום שנמצא על הפאה, על המקצוע או בפינה של הקובייה משויך כך לקובייה אחת בלבד מבין השכנות.

> 🔮 **נחשו:** כמה אטומים יש בקובייה של צלע $a$, וכמה זה ביחס ל-$1/v_p$?
"""),
("code", """
A = np.array([a1, a2, a3], dtype=float)
B = np.array(basis, dtype=float)
M = 6
n = np.array(list(itertools.product(range(-M, M + 1), repeat=3)))
R = n @ A                                            # lattice vectors
atoms = (R[:, None, :] + B[None, :, :]).reshape(-1, 3)
kind = np.tile(np.arange(len(B)), len(R))            # which basis atom each position is

v_p = abs(np.linalg.det(A))
eps = 1e-9
in_cube = np.all((atoms > -eps) & (atoms < 1 - eps), axis=1)   # half-open cube [0, a)^3
print(f"v_p = {v_p:.4f} a^3   ->   lattice points per a^3 = 1/v_p = {1 / v_p:.3f}")
print(f"atoms in the basis = {len(B)}   ->   atoms per a^3 = {len(B) / v_p:.3f}")
print(f"atoms counted in the half-open cube = {in_cube.sum()}")
"""),
("md", r"""
### 2. תמונה של הקובייה
מציגים את כל האטומים בקובייה הסגורה $0\le x,y,z\le a$ (כולל הפינות והפאות). הצבע מבחין בין אטומי הבסיס השונים.
"""),
("code", """
closed = np.all((atoms > -eps) & (atoms < 1 + eps), axis=1)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")
corners = np.array(list(itertools.product([0, 1], repeat=3)))
for i, j in itertools.combinations(range(8), 2):
    if np.abs(corners[i] - corners[j]).sum() == 1:
        ax.plot(*zip(corners[i], corners[j]), color="0.6", lw=0.8)
for k in range(len(B)):
    sel = closed & (kind == k)
    ax.scatter(*atoms[sel].T, s=60, depthshade=False, label=f"basis atom {k}")
ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.legend(frameon=False, loc="upper left")
plt.show()
"""),
("md", r"""
### 3. קליפות של שכנים
מחשבים את המרחקים מהאטום הראשון בבסיס (בראשית) לכל שאר האטומים, ומקבצים מרחקים שווים לקליפות. הטבלה מציגה לכל קליפה את המרחק ואת מספר האטומים בה; הקליפה הראשונה נותנת את מספר הקואורדינציה. הגרף מציג את $N(r)$: מספר האטומים במרחק של עד $r$ מהאטום שבראשית.

> 🔮 **נחשו:** בסריג ברירת המחדל, כמה שכנים יש בקליפה הראשונה ובשנייה, ובאילו מרחקים?
"""),
("code", """
d = np.linalg.norm(atoms - B[0], axis=1)
d = np.sort(d[(d > eps) & (d <= r_max + eps)])
shell_r, shell_n = np.unique(np.round(d, 6), return_counts=True)
print(" shell   distance / a   (distance / a)^2   atoms")
for i, (r, c) in enumerate(zip(shell_r, shell_n), start=1):
    print(f"{i:5d}   {r:12.4f}   {r**2:16.4f}   {c:5d}")

r_grid = np.linspace(0, r_max, 2000)
N_of_r = np.searchsorted(d, r_grid, side="right")    # number of atoms within distance r
plt.figure(figsize=(6, 3.4))
plt.plot(r_grid, N_of_r, lw=1.8)
plt.xlabel("r / a"); plt.ylabel("N(r)"); plt.title("atoms within distance r of the atom at the origin")
plt.show()
"""),
("md", r"""
### 4. יחס אריזה
מציבים בכל אטום כדור. הרדיוס הגדול ביותר שאפשר, כשכל הכדורים שווים, הוא חצי מהמרחק לשכן הקרוב, $R_{\max}=r_1/2$. יחס האריזה הוא החלק מהנפח שהכדורים ממלאים:
$$\eta=\frac{m\cdot\tfrac{4\pi}{3}R_{\max}^3}{v_p},$$
כאשר $m$ הוא מספר האטומים בבסיס.
"""),
("code", """
R_max = shell_r[0] / 2
eta = len(B) * (4 * np.pi / 3) * R_max**3 / v_p
print(f"nearest-neighbour distance r_1 = {shell_r[0]:.4f} a,  coordination number = {shell_n[0]}")
print(f"packing fraction eta = {eta:.4f}")
"""),
("md", r"""
### 💬 שאלות לדיון
1. למה הספירה בקובייה החצי-פתוחה נותנת את אותה תוצאה כמו הספירה בחלקים ($8\times\tfrac18+\dots$)?
2. בסריג ברירת המחדל, בכמה אחוזים הקליפה השנייה רחוקה יותר מהראשונה? איך זה קשור לצורה של תא ויגנר-זייץ שראיתם בתרגול?
3. החליפו את הווקטורים הפרימיטיביים בבחירה אחרת של אותו סריג, למשל $\mathbf a_1=(1,0,0)$, $\mathbf a_2=(0,1,0)$, $\mathbf a_3=(\tfrac12,\tfrac12,\tfrac12)$. מה משתנה בתוצאות ומה לא? למה?

> ✅ **סיימתם?** תשובות לשאלות הדיון אינן להגשה. הסעיף החישובי בתרגיל הבית נשען על המחברת הזו. בהצלחה!
"""),
]

# ------------------------------------------------------------------ week 5
w5 = [
("md", header(5, "שבוע 5 · אזורי ברילואן בדו-ממד", "הדגמה לתרגול 5 · מבוא למצב מוצק תשפ״ז · אוניברסיטת תל אביב") + r"""
## תיאור ההדגמה
לסריג דו-ממדי עם וקטורים פרימיטיביים $\mathbf a_1,\mathbf a_2$ המחברת מחשבת את הסריג ההופכי מהתנאי $\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}$, ובונה מפה של אזורי ברילואן: נקודה $\mathbf k$ שייכת לאזור ה-$n$ אם בדיוק $n-1$ נקודות של הסריג ההופכי קרובות אליה יותר מהראשית. לבסוף היא מודדת את השטח של כל אזור על ידי ספירת נקודות רשת.

**יחידות:** אורכים במרחב הישיר ביחידות של $|\mathbf a_1|$, ובמרחב ההופכי ביחידות של $1/|\mathbf a_1|$.

לצד המחברת יש גם [דף אינטראקטיבי](https://yaminbens.github.io/SolidState/week05-reciprocal) שמציג את הסריג ההופכי ואת האזורים בזמן אמת.
"""),
("code", """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
"""),
("md", r"""
### ⚙️ פרמטרים
- `a1`, `a2` – הווקטורים הפרימיטיביים של הסריג הישיר. ברירת המחדל היא סריג ריבועי.
- `n_max` – מספר האזורים שמחשבים ומציגים.
- `N_grid` – מספר נקודות הרשת בכל כיוון (יותר נקודות: שטחים מדויקים יותר, חישוב איטי יותר).
"""),
("code", """
# ---- Parameters ----
a1 = [1.0, 0.0]     # primitive vectors of the direct lattice
a2 = [0.0, 1.0]
n_max = 4           # number of Brillouin zones to compute
N_grid = 700        # grid points per direction
"""),
("md", r"""
### 1. הסריג ההופכי
כותבים את הווקטורים הפרימיטיביים כשורות של מטריצה $A$. התנאי $\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}$ הוא $AB^{T}=2\pi I$, ולכן $B=2\pi\,(A^{-1})^{T}$.

> 🔮 **נחשו:** אם מאריכים את $\mathbf a_2$ פי שניים, מה קורה ל-$\mathbf b_2$? ולכיוון שלו?
"""),
("code", """
A = np.array([a1, a2], dtype=float)
B = 2 * np.pi * np.linalg.inv(A).T          # rows are b1, b2
print("b1 =", np.round(B[0], 4), "   b2 =", np.round(B[1], 4))
print("A B^T / (2 pi) =\\n", np.round(A @ B.T / (2 * np.pi), 10))

fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, V, name, lab in [(axs[0], A, "direct lattice", "a"), (axs[1], B, "reciprocal lattice", "b")]:
    n = np.arange(-4, 5)
    P = np.array([i * V[0] + j * V[1] for i in n for j in n])
    ax.plot(P[:, 0], P[:, 1], "k.", ms=5)
    for i, c in enumerate(["C0", "C1"]):
        ax.annotate("", xy=V[i], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
        ax.text(*(V[i] * 1.12), f"{lab}{i+1}", color=c, fontsize=12)
    r = 2.6 * np.abs(V).max()
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_aspect("equal"); ax.set_title(name)
plt.tight_layout(); plt.show()
"""),
("md", r"""
### 2. מפת אזורי ברילואן
לכל נקודה $\mathbf k$ ברשת סופרים כמה וקטורים $\mathbf G\neq0$ מקיימים $|\mathbf k-\mathbf G|<|\mathbf k|$, כלומר $\mathbf k\cdot\mathbf G>|\mathbf G|^2/2$. האזור של הנקודה הוא המספר הזה ועוד אחד.

> 🔮 **נחשו:** מה הצורה של האזור הראשון בסריג הריבועי, ומכמה חתיכות נפרדות בנוי האזור השני?
"""),
("code", """
bmax = np.linalg.norm(B, axis=1).max()
L = 0.75 * np.sqrt(n_max + 2) * bmax                 # half-width of the k-grid
x = np.linspace(-L, L, N_grid)
KX, KY = np.meshgrid(x, x)
K = np.column_stack([KX.ravel(), KY.ravel()])

m = np.arange(-12, 13)
G = np.array([i * B[0] + j * B[1] for i in m for j in m if (i, j) != (0, 0)])
G = G[np.linalg.norm(G, axis=1) < 2.2 * L]            # only nearby G can matter

zone = np.ones(len(K), dtype=int)
for chunk in np.array_split(np.arange(len(K)), 20):   # count closer G points, in chunks
    zone[chunk] += ((K[chunk] @ G.T) > 0.5 * (G ** 2).sum(axis=1)).sum(axis=1)
zone = zone.reshape(KX.shape)

cmap = ListedColormap(plt.cm.tab10.colors[:n_max])
shown = np.where(zone <= n_max, zone, np.nan)
plt.figure(figsize=(5.6, 5))
plt.imshow(shown, origin="lower", extent=[-L, L, -L, L], cmap=cmap, vmin=0.5, vmax=n_max + 0.5)
plt.colorbar(ticks=range(1, n_max + 1), label="zone")
pts = G[np.all(np.abs(G) < L, axis=1)]
plt.plot(pts[:, 0], pts[:, 1], "k.", ms=4); plt.plot(0, 0, "k+", ms=10)
plt.xlabel("k_x"); plt.ylabel("k_y"); plt.title("Brillouin zones")
plt.show()
"""),
("md", r"""
### 3. השטח של כל אזור
סופרים את נקודות הרשת בכל אזור ומכפילים בשטח של משבצת רשת. מציגים את השטח של כל אזור ביחס לשטח של האזור הראשון.

> 🔮 **נחשו:** האם האזורים הרחוקים גדולים יותר מהאזור הראשון, קטנים יותר, או שווים לו?
"""),
("code", """
cell = (x[1] - x[0]) ** 2
areas = np.array([(zone == n).sum() * cell for n in range(1, n_max + 1)])
for n, s in enumerate(areas, start=1):
    print(f"zone {n}:  area = {s:8.4f}   area / area(zone 1) = {s / areas[0]:.3f}")
plt.figure(figsize=(5, 3.2))
plt.bar(range(1, n_max + 1), areas / areas[0], color="C0")
plt.axhline(1, color="0.6", lw=0.8)
plt.xlabel("zone n"); plt.ylabel("area / area of zone 1"); plt.ylim(0, 1.3)
plt.show()
"""),
("md", r"""
### 💬 שאלות לדיון
1. מה קורה ל-$\mathbf b_1$ ול-$\mathbf b_2$ כשמסובבים את $\mathbf a_2$ כך שהזווית בין $\mathbf a_1$ ל-$\mathbf a_2$ קטנה? מה לגבי הזווית בין $\mathbf b_1$ ל-$\mathbf b_2$?
2. מכמה חתיכות נפרדות בנוי האזור השני בסריג הריבועי? והשלישי?
3. נסו רשת גסה יותר (`N_grid` קטן). איך משתנים השטחים, ולמה?

> ✅ **סיימתם?** תשובות לשאלות הדיון אינן להגשה. הסעיף החישובי בתרגיל הבית נשען על המחברת הזו. בהצלחה!
"""),
]

# ------------------------------------------------------------------ week 6
w6 = [
("md", header(6, "שבוע 6 · סימולציית פיזור אבקה", "הדגמה לתרגול 6 · מבוא למצב מוצק תשפ״ז · אוניברסיטת תל אביב") + r"""
## תיאור ההדגמה
המחברת מחשבת תבנית פיזור של אבקה (שיטת Debye-Scherrer) לגביש קובי. הגביש מוגדר על ידי סריג ברווה קובי (SC, BCC או FCC) ובסיס של אטומים, לכל אחד מיקום וגורם צורה $f$. לכל וקטור הופכי $\mathbf G_{hkl}=\frac{2\pi}{a}(h,k,l)$ מחשבים את גורם המבנה של התא הקובי,
$$S(hkl)=\sum_j f_j\,e^{-2\pi i(hx_j+ky_j+lz_j)},$$
ואת זווית הפיזור מתנאי בראג, $\sin\theta=\frac{\lambda}{2a}\sqrt{h^2+k^2+l^2}$. באבקה כל הווקטורים באותו אורך נותנים שיא באותה זווית, ולכן העוצמה של שיא היא $|S|^2$ כפול **הריבוי**: מספר הווקטורים $(h,k,l)$ במשפחה. רוחב השיא נקבע לפי גודל הגרגירים.

**יחידות:** אורכים ב-nm, זוויות במעלות. גורמי הצורה $f_j$ נלקחים קבועים (לא תלויים בזווית).

לצד המחברת יש גם [דף אינטראקטיבי](https://yaminbens.github.io/SolidState/week06-diffraction) עם בניית אוולד ותבניות אבקה.
"""),
("code", """
import itertools
import numpy as np
import matplotlib.pyplot as plt
"""),
("md", r"""
### ⚙️ פרמטרים
- `lam` – אורך הגל של קרני ה-X (nm).
- `a` – צלע הקובייה של הסריג (nm).
- `centering` – סריג הברווה: `"SC"`, `"BCC"` או `"FCC"`.
- `basis` – האטומים שמוצמדים לכל נקודת סריג: לכל אחד המיקום ביחידות של $a$ וגורם הצורה $f$.
- `N_cells` – מספר התאים לאורך צלע של גרגיר (קובע את רוחב השיאים).
"""),
("code", """
# ---- Parameters ----
lam = 0.154                          # X-ray wavelength (nm)
a = 0.40                             # cubic lattice constant (nm)
centering = "BCC"                    # Bravais lattice: "SC", "BCC" or "FCC"
basis = [((0.0, 0.0, 0.0), 1.0)]     # (position in units of a, form factor f) for each atom of the basis
N_cells = 40                         # unit cells along the edge of a grain
"""),
("md", r"""
### 1. גורם המבנה וטבלת השיאים
בונים את כל האטומים בתא הקובי (הזזות הסריג בתוך הקובייה כפול הבסיס), ומחשבים את $S$ לכל $(h,k,l)$. השורות בטבלה הן משפחות: כל התמורות וכל הסימנים של אותם אינדקסים.

> 🔮 **נחשו:** בסריג ברירת המחדל, אילו משפחות מבין $(100)$, $(110)$, $(111)$, $(200)$ יופיעו בתבנית?
"""),
("code", """
shifts = {"SC": [(0, 0, 0)],
          "BCC": [(0, 0, 0), (.5, .5, .5)],
          "FCC": [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]}[centering]
atoms = [(np.mod(np.add(t, r), 1.0), f) for t in shifts for (r, f) in basis]   # all atoms of the cubic cell

def S_of(h, k, l):
    return sum(f * np.exp(-2j * np.pi * (h * x + k * y + l * z)) for (x, y, z), f in atoms)

hmax = int(2 * a / lam) + 1
fam = {}
for h, k, l in itertools.product(range(-hmax, hmax + 1), repeat=3):
    s2 = h * h + k * k + l * l
    if s2 == 0 or lam * np.sqrt(s2) / (2 * a) >= 1:
        continue
    key = tuple(sorted((abs(h), abs(k), abs(l)), reverse=True))   # the family {hkl}
    F = fam.setdefault(key, {"s2": s2, "m": 0, "I": 0.0, "S2": abs(S_of(h, k, l)) ** 2})
    F["m"] += 1                                                    # multiplicity
    F["I"] += abs(S_of(h, k, l)) ** 2
rows = sorted(fam.items(), key=lambda kv: kv[1]["s2"])
for key, F in rows:
    F["tt"] = 2 * np.degrees(np.arcsin(lam * np.sqrt(F["s2"]) / (2 * a)))
Imax = max(F["I"] for _, F in rows)
peaks = [(key, F) for key, F in rows if F["I"] > 1e-9 * Imax]
s2_first = peaks[0][1]["s2"]

print(" {hkl}    h2+k2+l2   2theta    mult    |S|^2     I/Imax   sin2/sin2(first)")
for key, F in rows[:14]:
    hkl = "".join(map(str, key))
    if F["I"] > 1e-9 * Imax:
        print(f" {hkl:>5}   {F['s2']:6d}   {F['tt']:7.2f}   {F['m']:5d}   {F['S2']:7.3f}   {F['I']/Imax:7.3f}   {F['s2']/s2_first:6.3f}")
    else:
        print(f" {hkl:>5}   {F['s2']:6d}   {F['tt']:7.2f}   {F['m']:5d}   {F['S2']:7.3f}   (absent)")
"""),
("md", r"""
### 2. תבנית הפיזור
כל שיא מצויר כגאוסיאן ברוחב שנקבע מגודל הגרגיר: $\Delta(2\theta)\approx\frac{0.9\lambda}{N a\cos\theta}$ (נוסחת שרר). שיאים בעלי אותו $h^2+k^2+l^2$ נופלים באותה זווית ומתחברים.

> 🔮 **נחשו:** איזה שיא יהיה הגבוה ביותר: זה שה-$|S|^2$ שלו הכי גדול, או זה שהריבוי שלו הכי גדול?
"""),
("code", """
tt = np.linspace(10, 120, 4000)
pattern = np.zeros_like(tt)
for key, F in peaks:
    th = np.radians(F["tt"] / 2)
    fwhm = np.degrees(0.9 * lam / (N_cells * a * np.cos(th)))
    pattern += F["I"] / Imax * np.exp(-4 * np.log(2) * ((tt - F["tt"]) / fwhm) ** 2)

plt.figure(figsize=(8, 3.6))
plt.plot(tt, pattern, lw=1.4)
for key, F in peaks:
    if F["tt"] < 120 and F["I"] / Imax > 0.02:
        plt.text(F["tt"], F["I"] / Imax + 0.04, "".join(map(str, key)), ha="center", fontsize=8)
plt.xlabel("2 theta (deg)"); plt.ylabel("intensity / max"); plt.ylim(0, 1.2)
plt.title(f"{centering}, a = {a} nm, lambda = {lam} nm")
plt.show()
"""),
("md", r"""
### 3. רוחב השיאים וגודל הגרגירים
מציירים את השיא הראשון עבור גרגירים בגדלים שונים. גרגיר עם $N$ תאים לאורך צלע הוא "שריג עקיפה" עם $N$ חריצים, והשיא שלו ברוחב של בערך $2\pi/(Na)$ במרחב ההופכי.
"""),
("code", """
key0, F0 = peaks[0]
th0 = np.radians(F0["tt"] / 2)
x = np.linspace(F0["tt"] - 3, F0["tt"] + 3, 1500)
plt.figure(figsize=(6, 3.2))
for N, ls in [(5, ":"), (20, "--"), (80, "-")]:
    fw = np.degrees(0.9 * lam / (N * a * np.cos(th0)))
    plt.plot(x, np.exp(-4 * np.log(2) * ((x - F0["tt"]) / fw) ** 2), ls, label=f"N = {N}  (FWHM = {fw:.2f} deg)")
plt.xlabel("2 theta (deg)"); plt.ylabel("intensity / peak"); plt.legend(frameon=False, fontsize=8)
plt.title("first peak, " + "".join(map(str, key0)))
plt.show()
"""),
("md", r"""
### 💬 שאלות לדיון
1. אילו משפחות חסרות בטבלה עבור ברירת המחדל, ואיזה כלל על $h,k,l$ מתאר אותן?
2. מדוע שני שיאים עם אותו $|S|^2$ יכולים להיות בגבהים שונים?
3. איך אפשר להעריך מתבנית אבקה אמיתית את גודל הגרגירים באבקה?

> ✅ **סיימתם?** תשובות לשאלות הדיון אינן להגשה. הסעיף החישובי בתרגיל הבית נשען על המחברת הזו. בהצלחה!
"""),
]

def build(name, cells, out):
    nb = nb_from(cells)
    # Execute a copy only to verify that every cell runs; the notebook that is saved stays
    # unrun (no outputs), so students see the results only after they run the code themselves.
    NotebookClient(copy.deepcopy(nb), timeout=600, kernel_name="python3").execute()
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
    nbformat.write(nb, out)
    print("wrote (unrun)", out)

if __name__ == "__main__":
    build("w1", w1, ROOT / "week01.ipynb")
    build("w2", w2, ROOT / "week02.ipynb")
    build("w3", w3, ROOT / "week03.ipynb")
    build("w4", w4, ROOT / "week04.ipynb")
    build("w5", w5, ROOT / "week05.ipynb")
    build("w6", w6, ROOT / "week06.ipynb")
