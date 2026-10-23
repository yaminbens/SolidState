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

<details>
<summary>🗝️ הכוונה לתשובות (לחצו לפתיחה)</summary>

1. תרגול 1, סעיף 1 (ההערה על הפוטנציאל הכימי ואנרגיית פרמי). החישוב הכמותי של השינוי בטמפרטורות נמוכות יגיע בתרגול 3, סעיף 2 (פיתוח זומרפלד).
2. תרגול 1, סעיף 1 (ההערה: $\mu$ נקבע מהדרישה על $n$), ותרגיל בית 1, שאלה 3: מהצורה הסגורה של $\mu$ בשני ממדים רואים מה קורה כש-$T\gg T_F$.
3. תרגול 1, סעיף 3 (צפיפות המצבים בכל מימד והטבלה). ההסבר המלא, כולל הסימן של השינוי בכל מימד, בתרגול 3, סעיף 2.

אם אינכם יודעים לענות על אחת השאלות, חזרו לסעיף המתאים בתרגול.

</details>

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

<details>
<summary>🗝️ הכוונה לתשובות (לחצו לפתיחה)</summary>

1. תרגול 2, סעיף 1 (המצב העמיד תחת שדה קבוע וחוק אוהם).
2. תרגול 2, סעיפים 2 ו-3 (הטנזור כמספר מרוכב והצורה הפולרית שלו, עם $\tan\theta_H=\omega_c\tau$).
3. תרגול 2, סעיף 4 (מוליכות AC ותדר הפלזמה).

אם אינכם יודעים לענות על אחת השאלות, חזרו לסעיף המתאים בתרגול.

</details>

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

<details>
<summary>🗝️ הכוונה לתשובות (לחצו לפתיחה)</summary>

1. תרגול 4, סעיף 1 (כלל הספירה $V/v_p$) ותרגיל 1.
2. תרגול 4, סעיף 2 (קליפות שכנים), וסעיף 3, תרגיל 3 והאיור של תאי ויגנר-זייץ (אילו פאות נקבעות על ידי השכנים השניים).
3. תרגול 4, סעיף 1 (בחירות שונות של וקטורים פרימיטיביים, $A'=AM$ עם $\det M=\pm1$) ותרגיל 1.

אם אינכם יודעים לענות על אחת השאלות, חזרו לסעיף המתאים בתרגול.

</details>

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

<details>
<summary>🗝️ הכוונה לתשובות (לחצו לפתיחה)</summary>

1. תרגול 5, סעיף 1 (התנאי $\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}$ והניצבות $\mathbf b_1\perp\mathbf a_2$, $\mathbf b_2\perp\mathbf a_1$).
2. תרגול 5, סעיף 3 (ההגדרה של אזור ברילואן ה-$n$) ותרגיל 4.
3. זו שאלה על השיטה הנומרית ולא על התרגול: ראו בחלק 3 של המחברת איך השטח מחושב מספירה של נקודות רשת.

אם אינכם יודעים לענות על אחת השאלות, חזרו לסעיף המתאים בתרגול.

</details>

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

<details>
<summary>🗝️ הכוונה לתשובות (לחצו לפתיחה)</summary>

1. תרגול 6, סעיף 3 (גורם המבנה של SC, BCC ו-FCC והטבלה).
2. תרגול 6, סעיף 2 (הפסקה על אבקה: בתבנית רואים רק את האורכים של הווקטורים ההופכיים), והניחוש שלפני חלק 2 במחברת.
3. תרגול 6, סעיף 4 (רוחב השיאים וגודל הגרגירים), והרוחב של פונקציית השריג בסעיף 1.

אם אינכם יודעים לענות על אחת השאלות, חזרו לסעיף המתאים בתרגול.

</details>

> ✅ **סיימתם?** תשובות לשאלות הדיון אינן להגשה. הסעיף החישובי בתרגיל הבית נשען על המחברת הזו. בהצלחה!
"""),
]

# ------------------------------------------------------------------ week 7
w7 = [
("md", header(7, "שבוע 7 · שרשרת הרמונית: ממטריצת הכוחות ליחס הנפיצה", "הדגמה לתרגול 7 · מבוא למצב מוצק תשפ״ז · אוניברסיטת תל אביב") + r"""
## תיאור ההדגמה
שרשרת סגורה (טבעת) של $N$ אטומים זהים במסה $m$, במרחק $a$ זה מזה. כל אטום קשור בקפיץ $K_1$ לשכנים הקרובים ביותר, ובקפיץ $K_2$ לשכנים מסדר שני. בקירוב ההרמוני משוואות התנועה הן
$$m\ddot u_n=-\sum_{n'}\Phi_{n n'}\,u_{n'} ,$$
כאשר $\Phi$ היא מטריצת הכוחות, מטריצה בגודל $N\times N$. המחברת עושה שני דברים שונים ומשווה ביניהם:
1. מלכסנת את $\Phi$ ישירות, כמו כל בעיית מתנדים צמודים, ומקבלת $N$ תדירויות;
2. משתמשת בסימטריית ההזזה: כיוון ש-$\Phi_{nn'}$ תלוי רק ב-$n-n'$, התמרת פורייה של שורה אחת של $\Phi$ נותנת את כל התדירויות, כל אחת עם וקטור גל $k$ משלה.

אחר כך היא בונה מהתדירויות את צפיפות המצבים ואת קיבול החום.

**יחידות:** $m=K_1=a=1$, ולכן תדירויות נמדדות ביחידות של $\sqrt{K_1/m}$. בחלק 4 גם $\hbar=k_B=1$.

לצד המחברת יש גם [דף אינטראקטיבי](https://yaminbens.github.io/SolidState/week07-chain) שמראה את תנועת האטומים בכל אופן תנודה של שרשרת עם בסיס.
"""),
("code", """
import numpy as np
import matplotlib.pyplot as plt
"""),
("md", r"""
### ⚙️ פרמטרים
- `K1` – קבוע הקפיץ בין שכנים קרובים ביותר.
- `K2` – קבוע הקפיץ בין שכנים מסדר שני.
- `m` – מסת כל אטום.
- `N` – מספר האטומים בטבעת בחלק 1 (הלכסון הישיר).
"""),
("code", """
# ---- Parameters ----
K1 = 1.0     # nearest-neighbour spring
K2 = 0.0     # second-neighbour spring
m = 1.0      # mass of every atom
N = 60       # atoms in the ring of part 1
"""),
("md", r"""
### 1. אותן תדירויות בשתי דרכים
בונים את $\Phi$ מהאנרגיה $U=\frac{K_1}{2}\sum_n(u_n-u_{n+1})^2+\frac{K_2}{2}\sum_n(u_n-u_{n+2})^2$: כל קפיץ בין אטום $i$ לאטום $j$ תורם $+K$ לאיברים $\Phi_{ii},\Phi_{jj}$ ו-$-K$ לאיברים $\Phi_{ij},\Phi_{ji}$. אחר כך משווים את $N$ הערכים העצמיים של $\Phi/m$ להתמרת הפורייה של השורה הראשונה של $\Phi/m$.

> 🔮 **נחשו:** כמה מהתדירויות יוצאות אפס, ולאיזו תנועה של השרשרת הן מתאימות?
"""),
("code", """
def force_matrix(N, K1, K2):
    # N x N force-constant matrix of a ring: each spring adds +K on the diagonal and -K off it
    Phi = np.zeros((N, N))
    for j, K in [(1, K1), (2, K2)]:
        for i in range(N):
            p = (i + j) % N
            Phi[i, i] += K; Phi[p, p] += K
            Phi[i, p] -= K; Phi[p, i] -= K
    return Phi

Phi = force_matrix(N, K1, K2)
print("row sums of Phi (should all vanish):", np.abs(Phi.sum(axis=1)).max())

w2_direct = np.sort(np.linalg.eigvalsh(Phi / m))      # omega^2 from the full N x N problem
D_k = np.real(np.fft.fft(Phi[0] / m))                 # omega^2(k) from one row, k = 2 pi j / (N a)
w2_fourier = np.sort(D_k)
print("largest difference between the two lists:", np.abs(w2_direct - w2_fourier).max())

plt.figure(figsize=(6, 3.2))
plt.plot(w2_direct, "o", ms=5, mfc="none", label="eigenvalues of Phi / m")
plt.plot(w2_fourier, ".", ms=4, label="Fourier transform of one row")
plt.xlabel("mode index (sorted)"); plt.ylabel("omega^2"); plt.legend(frameon=False)
plt.show()
"""),
("md", r"""
### 2. יחס הנפיצה
מאחר שהתמרת הפורייה נותנת את $\omega^2$ לכל $k$, אפשר לקחת טבעת ארוכה מאוד ולקבל עקומה רציפה באזור ברילואן הראשון, $-\pi/a\le k<\pi/a$. אם $\omega^2(k)<0$ עבור $k$ כלשהו, השרשרת אינה יציבה: המחברת מציינת זאת ומציירת רק את התחום היציב.

> 🔮 **נחשו:** היכן ביחס הנפיצה מהירות החבורה $v_g=d\omega/dk$ מתאפסת?
"""),
("code", """
N_big = 20000
Phi_row = np.zeros(N_big)                              # first row of Phi for a long ring
Phi_row[0] = 2 * K1 + 2 * K2
Phi_row[[1, -1]] -= K1
Phi_row[[2, -2]] -= K2
w2_k = np.real(np.fft.fft(Phi_row / m))
k = 2 * np.pi * np.fft.fftfreq(N_big)                  # k in units of 1/a, inside [-pi, pi)
order = np.argsort(k)
k, w2_k = k[order], w2_k[order]
if w2_k.min() < -1e-9:
    print(f"unstable chain: omega^2 reaches {w2_k.min():.3f} at k a = {k[np.argmin(w2_k)]:.3f}")
w_k = np.sqrt(np.clip(w2_k, 0, None))
v_g = np.gradient(w_k, k)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
axs[0].plot(k / np.pi, w_k)
axs[0].set_xlabel("k a / pi"); axs[0].set_ylabel("omega"); axs[0].set_title("dispersion")
axs[1].plot(k / np.pi, v_g, color="C1"); axs[1].axhline(0, color="0.6", lw=0.8)
axs[1].set_xlabel("k a / pi"); axs[1].set_ylabel("v_g = d omega / dk"); axs[1].set_title("group velocity")
plt.tight_layout(); plt.show()
print(f"sound velocity (slope near k = 0): {w_k[N_big // 2 + 5] / k[N_big // 2 + 5]:.4f}")
print(f"highest frequency: {w_k.max():.4f}")
"""),
("md", r"""
### 3. צפיפות המצבים
צפיפות המצבים $g(\omega)$ סופרת כמה אופני תנודה יש בכל תחום תדירויות. בונים היסטוגרמה של כל $N$ התדירויות של הטבעת הארוכה, מנורמלת כך שהשטח מתחתיה הוא אופן תנודה אחד לכל אטום.

> 🔮 **נחשו:** באילו תדירויות $g(\omega)$ יהיה גדול במיוחד? היעזרו בגרף מהירות החבורה מחלק 2.
"""),
("code", """
hist, edges = np.histogram(w_k, bins=600, density=True)
centers = 0.5 * (edges[1:] + edges[:-1])
plt.figure(figsize=(6, 3.2))
plt.plot(centers, hist)
plt.xlabel("omega"); plt.ylabel("g(omega) per atom"); plt.ylim(0, 3)
plt.title("density of states")
plt.show()
"""),
("md", r"""
### 4. מהתדירויות לקיבול החום
כל אופן תנודה הוא מתנד הרמוני קוונטי, ולכן קיבול החום לאטום הוא ממוצע על כל האופנים של $c(x)=\frac{x^2e^x}{(e^x-1)^2}$ עם $x=\hbar\omega/k_BT$. משווים לקירוב Debye בממד אחד: יחס נפיצה לינארי עם מהירות הקול מחלק 2, עד תדירות קיטעון שנקבעת מכך שיש אופן תנודה אחד לכל אטום.

> 🔮 **נחשו:** בטמפרטורות נמוכות, האם קיבול החום של השרשרת יהיה קרוב לזה של Debye? ובטמפרטורות גבוהות?
"""),
("code", """
def c_mode(x):
    # heat capacity of one harmonic oscillator (units of k_B); written to avoid overflow at large x
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    ok = (x > 1e-8) & (x < 300)
    e = np.exp(-x[ok])
    out[ok] = x[ok] ** 2 * e / (1 - e) ** 2
    out[x <= 1e-8] = 1.0
    return out

c_s = w_k[N_big // 2 + 5] / k[N_big // 2 + 5]         # sound velocity from part 2
w_D = np.pi * c_s                                     # Debye cutoff: one mode per atom in 1D
w_debye = np.linspace(0, w_D, 4000)[1:]
w_modes = w_k[w_k > 1e-9]

Ts = np.linspace(0.02, 3, 150)
c_chain = [c_mode(w_modes / T).mean() for T in Ts]
c_debye = [c_mode(w_debye / T).mean() for T in Ts]
plt.figure(figsize=(6, 3.2))
plt.plot(Ts, c_chain, label="chain (all modes)")
plt.plot(Ts, c_debye, "--", label="Debye, same sound velocity")
plt.axhline(1, color="0.6", lw=0.8)
plt.xlabel("k_B T / (hbar sqrt(K1/m))"); plt.ylabel("c per atom / k_B"); plt.legend(frameon=False)
plt.show()
"""),
("md", r"""
### 💬 שאלות לדיון
1. למה בחלק 1 מתקבלת תדירות אפס, ומה היה קורה לו סכומי השורות של $\Phi$ לא היו מתאפסים?
2. מדוע מספיקה שורה אחת של $\Phi$ כדי לקבל את כל $N$ התדירויות?
3. איך קשורים השיאים של $g(\omega)$ למהירות החבורה?
4. למה Debye ו"השרשרת האמיתית" מסכימים בטמפרטורות נמוכות וגבוהות, ונבדלים ביניהן?

<details>
<summary>🗝️ הכוונה לתשובות (לחצו לפתיחה)</summary>

1. תרגול 7, סעיף 1 (כלל הסכום: הזזה אחידה של כל השרשרת אינה עולה אנרגיה).
2. תרגול 7, סעיף 1 (סימטריית ההזזה והצבת גל מישורי), והרמז בתיאור ההדגמה.
3. תרגול 7, סעיף 2 (מהירות החבורה בקצה אזור ברילואן), ותרגול 1, סעיף 3 (צפיפות מצבים מיחס נפיצה).
4. תרגול 7, סעיף 2 (ההשוואה ל-Debye), ותרגול 3 (קיבול החום כספירה של אופנים פעילים).

אם אינכם יודעים לענות על אחת השאלות, חזרו לסעיף המתאים בתרגול.

</details>

> ✅ **סיימתם?** תשובות לשאלות הדיון אינן להגשה. הסעיף החישובי בתרגיל הבית נשען על המחברת הזו. בהצלחה!
"""),
]

# ------------------------------------------------------------------ week 8
w8 = [
("md", header(8, "שבוע 8 · פונונים בגביש תלת-ממדי", "הדגמה לתרגול 8 · מבוא למצב מוצק תשפ״ז · אוניברסיטת תל אביב") + r"""
## תיאור ההדגמה
גביש קובי (SC או FCC) של אטומים זהים במסה $m$, עם קפיצים מרכזיים בין שכנים קרובים ביותר בלבד. לקפיץ שני קבועים: $\kappa_L$ למתיחה לאורך הקשר ו-$\kappa_T$ להזזה בניצב לו. כמו בתרגול, לכל וקטור גל $\mathbf k$ המטריצה הדינמית היא
$$D(\mathbf k)=\sum_{\mathbf R}2\sin^2\!\left(\tfrac12\mathbf k\cdot\mathbf R\right)\left[\kappa_T\,\mathbb{I}+(\kappa_L-\kappa_T)\,\hat{\mathbf R}\hat{\mathbf R}^{T}\right],$$
כאשר הסכום הוא על השכנים הקרובים ביותר. הערכים העצמיים של $D(\mathbf k)/m$ הם $\omega_s^2(\mathbf k)$, והווקטורים העצמיים הם וקטורי הקיטוב. המחברת:
1. מציירת את שלושת הענפים לאורך מסלול בין נקודות סימטריה, צבועים לפי הקיטוב;
2. מגרילה מאות אלפי וקטורי גל באזור ברילואן ובונה מהם את צפיפות המצבים;
3. מחשבת מהתדירויות את קיבול החום.

**יחידות:** $a=m=\kappa_L=1$ ($a$ צלע הקובייה), ולכן תדירויות ביחידות של $\sqrt{\kappa_L/m}$. בחלק 3 גם $\hbar=k_B=1$.

לצד המחברת יש גם [דף אינטראקטיבי](https://yaminbens.github.io/SolidState/week08-phonons) שמראה את תנועת האטומים בכל אופן תנודה.
"""),
("code", """
import itertools
import numpy as np
import matplotlib.pyplot as plt
"""),
("md", r"""
### ⚙️ פרמטרים
- `structure` – סריג הברווה: `"SC"` או `"FCC"`.
- `kT_over_kL` – היחס $\kappa_T/\kappa_L$ בין הקשיחות הרוחבית לקשיחות האורכית של קשר.
- `n_samples` – מספר וקטורי הגל שמוגרלים לצפיפות המצבים (יותר: היסטוגרמה חלקה יותר, חישוב איטי יותר).
"""),
("code", """
# ---- Parameters ----
structure = "FCC"     # Bravais lattice: "SC" or "FCC"
kT_over_kL = 0.3      # transverse / longitudinal bond stiffness
n_samples = 200000    # random k vectors for the density of states
"""),
("md", r"""
### 1. השכנים והמטריצה הדינמית
בונים את רשימת השכנים הקרובים ביותר, ופונקציה שמחשבת את $D(\mathbf k)$ לכל וקטור גל. בדיקה ראשונה: ב-$\mathbf k=0$ המטריצה מתאפסת (כלל הסכום מהתרגול), ולכן שלוש תדירויות מתאפסות.
"""),
("code", """
kL, kT = 1.0, kT_over_kL
if structure == "SC":
    R = np.array([v for v in itertools.product([-1, 0, 1], repeat=3) if sum(map(abs, v)) == 1], float)
    A = np.eye(3)                                            # primitive vectors (rows)
    sym_points = {"Γ": (0, 0, 0), "X": (0.5, 0, 0), "M": (0.5, 0.5, 0), "R": (0.5, 0.5, 0.5)}
    route = ["Γ", "X", "M", "Γ", "R"]
elif structure == "FCC":
    R = 0.5 * np.array([v for v in itertools.product([-1, 0, 1], repeat=3) if sum(map(abs, v)) == 2], float)
    A = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    sym_points = {"Γ": (0, 0, 0), "X": (1, 0, 0), "X'": (1, 1, 0), "K": (0.75, 0.75, 0), "L": (0.5, 0.5, 0.5)}
    route = ["Γ", "X", "X'", "K", "Γ", "L"]
else:
    raise ValueError("structure must be 'SC' or 'FCC'")
Rhat = R / np.linalg.norm(R, axis=1)[:, None]
B = 2 * np.pi * np.linalg.inv(A).T                           # reciprocal primitive vectors (rows)
print(f"{structure}: {len(R)} nearest neighbours at distance {np.linalg.norm(R[0]):.4f} a")

def dyn_matrix(K):
    # K: array of shape (n, 3). Returns D(k) for every row, shape (n, 3, 3).
    K = np.atleast_2d(K)
    D = np.zeros((len(K), 3, 3))
    for r, rh in zip(R, Rhat):
        s = 2 * np.sin(K @ r / 2) ** 2
        D += s[:, None, None] * (kT * np.eye(3) + (kL - kT) * np.outer(rh, rh))[None]
    return D

print("D(k=0) =\\n", dyn_matrix(np.zeros(3))[0])
"""),
("md", r"""
### 2. הענפים לאורך מסלול בין נקודות סימטריה
הצבע של כל נקודה הוא החלק האורכי של הקיטוב, $|\hat{\mathbf k}\cdot\mathbf e|^2$: כחול לגמרי לגל אורכי, כתום לגמרי לגל רוחבי.

> 🔮 **נחשו:** לאורך איזה קטע של המסלול שני הענפים הרוחביים יתלכדו לקו אחד?
"""),
("code", """
def route_k(names, n=150):
    ks, xs, ticks, x = [], [], [0.0], 0.0
    for a, b in zip(names[:-1], names[1:]):
        p, q = 2 * np.pi * np.array(sym_points[a], float), 2 * np.pi * np.array(sym_points[b], float)
        L = np.linalg.norm(q - p)
        for t in np.linspace(0, 1, n, endpoint=False):
            ks.append(p + t * (q - p)); xs.append(x + t * L)
        x += L; ticks.append(x)
    ks.append(2 * np.pi * np.array(sym_points[names[-1]], float)); xs.append(x)
    return np.array(ks), np.array(xs), ticks

Kp, xp, ticks = route_k(route)
w2, vecs = np.linalg.eigh(dyn_matrix(Kp))                    # eigenvalues and polarisation vectors
wp = np.sqrt(np.clip(w2, 0, None))
khat = Kp / np.maximum(np.linalg.norm(Kp, axis=1), 1e-12)[:, None]
long_frac = np.abs(np.einsum("ni,nij->nj", khat, vecs)) ** 2

plt.figure(figsize=(7.5, 3.8))
for j in range(3):
    sc = plt.scatter(xp, wp[:, j], c=long_frac[:, j], cmap="coolwarm_r", vmin=0, vmax=1, s=4)
for t in ticks: plt.axvline(t, color="0.8", lw=0.7)
plt.xticks(ticks, route); plt.xlim(ticks[0], ticks[-1])
plt.ylabel("omega"); plt.title(f"{structure},  kT/kL = {kT_over_kL}")
plt.colorbar(sc, label="longitudinal fraction")
plt.show()
"""),
("md", r"""
### 3. צפיפות המצבים וקיבול החום
מגרילים וקטורי גל באופן אחיד בתא פרימיטיבי של הסריג ההופכי (שקול לאזור ברילואן, כי התדירויות מחזוריות), ומלכסנים את $D(\mathbf k)$ בכל אחד. ההיסטוגרמה של כל התדירויות היא צפיפות המצבים, מנורמלת לשלושה אופנים לאטום. מאותן תדירויות מחשבים את קיבול החום לאטום, $c=\langle\sum_s c(\hbar\omega_s/k_BT)\rangle$ עם $c(x)=x^2e^x/(e^x-1)^2$, ומציירים אותו בסקלה לוגריתמית.

> 🔮 **נחשו:** איך צפיפות המצבים מתנהגת בתדירויות נמוכות, ומה צריך להיות השיפוע של $c(T)$ בסקלה לוגריתמית בטמפרטורות נמוכות?
"""),
("code", """
rng = np.random.default_rng(0)
K = rng.random((n_samples, 3)) @ B                           # uniform in a primitive cell of the reciprocal lattice
w = np.sqrt(np.clip(np.linalg.eigvalsh(dyn_matrix(K)), 0, None)).ravel()

def c_mode(x):
    # heat capacity of one harmonic oscillator, in units of k_B
    x = np.asarray(x, float)
    out = np.ones_like(x)
    ok = x > 1e-8
    e = np.exp(-np.minimum(x[ok], 700))
    out[ok] = x[ok] ** 2 * e / (1 - e) ** 2
    return out

Ts = np.logspace(-1.7, 0.6, 50)
c = np.array([3 * c_mode(w / T).mean() for T in Ts])        # 3 modes per atom

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
h, edges = np.histogram(w, bins=240, density=True)
axs[0].plot(0.5 * (edges[1:] + edges[:-1]), 3 * h)
axs[0].set_xlabel("omega"); axs[0].set_ylabel("g(omega) per atom"); axs[0].set_title("density of states")
axs[1].loglog(Ts, c, label="all modes")
axs[1].axhline(3, color="0.6", lw=0.8, ls=":")
axs[1].set_xlabel("k_B T"); axs[1].set_ylabel("c per atom / k_B"); axs[1].set_title("heat capacity")
plt.tight_layout(); plt.show()

slope = np.polyfit(np.log(Ts[:8]), np.log(c[:8]), 1)[0]
print(f"low-temperature slope of log c vs log T: {slope:.2f}")
"""),
("md", r"""
### 💬 שאלות לדיון
1. למה בכל סריג ברווה מתקבלים בדיוק שלושה ענפים, ולמה שלושתם מתאפסים ב-$\mathbf k=0$?
2. לאורך אילו כיוונים שני הענפים הרוחביים מתלכדים, ומה מיוחד בכיוונים האלה?
3. מאיפה מגיעים השיאים והשבירות בצפיפות המצבים?
4. מה צריך להיות השיפוע של $c(T)$ בטמפרטורות נמוכות בגביש תלת-ממדי, ומה קובע אותו?

<details>
<summary>🗝️ הכוונה לתשובות (לחצו לפתיחה)</summary>

1. תרגול 8, סעיף 1 (ספירת הענפים וכלל הסכום), ותרגול 7, סעיף 1.
2. תרגול 8, סעיף 3 (ניוון של הענפים הרוחביים לאורך ציר סימטריה).
3. תרגול 7, סעיף 2 (צפיפות המצבים פרופורציונית ל-$1/|v_g|$), ותרגול 1, סעיף 3.
4. תרגול 3 (קיבול החום כספירה של אופנים פעילים) ותרגול 1 (צפיפות מצבים בממד $d$).

אם אינכם יודעים לענות על אחת השאלות, חזרו לסעיף המתאים בתרגול.

</details>

> ✅ **סיימתם?** תשובות לשאלות הדיון אינן להגשה. הסעיף החישובי בתרגיל הבית נשען על המחברת הזו. בהצלחה!
"""),
]

# ------------------------------------------------------------------ week 9
w9 = [
("md", header(9, "שבוע 9 · פסים בפוטנציאל מחזורי: מטריצת המעבר", "הדגמה לתרגול 9 · מבוא למצב מוצק תשפ״ז · אוניברסיטת תל אביב") + r"""
## תיאור ההדגמה
אלקטרון בשרשרת חד-ממדית של בארות פוטנציאל: בכל תא יחידה באר ברוחב $w$ (פוטנציאל 0) ומחסום ברוחב $b$ ובגובה $v_0$, כך שקבוע הסריג הוא $a=w+b$. כמו בתרגול, לכל אנרגיה $\varepsilon$ מחשבים את מטריצת המעבר $M(\varepsilon)$, שמעבירה את $(\psi,\psi')$ מתחילת תא לתחילת התא הבא, ומשפט בלוך נותן
$$\cos ka=\tfrac12\,\mathrm{Tr}\,M(\varepsilon).$$
אנרגיות שבהן $|\tfrac12\mathrm{Tr}\,M|\le1$ שייכות לפסים, והשאר הן פערים. המחברת:
1. בונה את $M(\varepsilon)$ ובודקת ש-$\det M=1$;
2. מציירת את $\tfrac12\mathrm{Tr}\,M(\varepsilon)$ ומוצאת את הפסים;
3. מציירת את יחס הנפיצה $\varepsilon(k)$ לעומת אלקטרון חופשי;
4. מחשבת את הרוחב של הפס הנמוך ביותר כתלות בעובי המחסום.

**יחידות:** אורכים ב-nm ואנרגיות ב-eV, עם מסת אלקטרון חופשי: $\hbar^2/2m=0.0381\ \mathrm{eV\cdot nm^2}$.

לצד המחברת יש גם [דף אינטראקטיבי](https://yaminbens.github.io/SolidState/week09-kp) עם מודל קרוניג-פני של פונקציות דלתא.
"""),
("code", """
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

C = 0.0381          # hbar^2 / (2 m_e) in eV nm^2
"""),
("md", r"""
### ⚙️ פרמטרים
- `v0` – גובה המחסום (eV).
- `w` – רוחב הבאר (nm).
- `b` – עובי המחסום (nm), בחלקים 1 עד 3.
- `E_max` – האנרגיה הגבוהה ביותר בגרפים (eV).
- `b_values` – עוביי המחסום שעבורם מחשבים בחלק 4 את רוחב הפס הנמוך ביותר.
"""),
("code", """
# ---- Parameters ----
v0 = 0.5                              # barrier height (eV)
w = 1.0                               # well width (nm)
b = 0.3                               # barrier thickness (nm), parts 1-3
E_max = 1.5                           # highest energy shown (eV)
b_values = np.linspace(0.2, 1.0, 9)   # barrier thicknesses for part 4 (nm)
"""),
("md", r"""
### 1. מטריצת המעבר
בתחום שבו הפוטנציאל קבוע $V$, הפתרון של משוואת שרדינגר באנרגיה $\varepsilon$ הוא צירוף של $\cos qx$ ו-$\sin qx$ עם $q^2=(\varepsilon-V)/(\hbar^2/2m)$, או של $\cosh$ ו-$\sinh$ כאשר $\varepsilon<V$. המטריצה של תחום אחד מעבירה את $(\psi,\psi')$ מקצה אחד שלו לקצה השני; המטריצה של תא שלם היא המכפלה של המטריצות של הבאר ושל המחסום.

> 🔮 **נחשו:** מה צריכה להיות הדטרמיננטה של $M$ לכל אנרגיה, ולמה?
"""),
("code", """
def region(E, V, L):
    # transfer matrix of a region of length L with constant potential V: (psi, psi') -> (psi, psi')
    q2 = (E - V) / C
    if q2 > 0:
        q = np.sqrt(q2)
        return np.array([[np.cos(q * L), np.sin(q * L) / q], [-q * np.sin(q * L), np.cos(q * L)]])
    if q2 < 0:
        k = np.sqrt(-q2)
        return np.array([[np.cosh(k * L), np.sinh(k * L) / k], [k * np.sinh(k * L), np.cosh(k * L)]])
    return np.array([[1.0, L], [0.0, 1.0]])

def cell(E, v0=v0, w=w, b=b):
    # one unit cell: well (V = 0, width w) followed by a barrier (V = v0, width b)
    return region(E, v0, b) @ region(E, 0.0, w)

def half_trace(E, v0=v0, w=w, b=b):
    return 0.5 * np.trace(cell(E, v0, w, b))

a = w + b
print(f"lattice constant a = {a:.3f} nm")
for E in [0.05, 0.3, 0.9]:
    print(f"E = {E:4.2f} eV:   det M = {np.linalg.det(cell(E)):.12f}   (1/2) Tr M = {half_trace(E):+.4f}")
"""),
("md", r"""
### 2. הפסים
מציירים את $\tfrac12\mathrm{Tr}\,M(\varepsilon)$. התחומים שבהם הגרף בתוך הרצועה $[-1,1]$ הם הפסים (מסומנים בפס כחול בתחתית), ומודפסות הקצוות והרוחב של כל פס.

> 🔮 **נחשו:** האם הפסים יהיו רחבים יותר באנרגיות נמוכות או גבוהות? למה?
"""),
("code", """
E = np.linspace(1e-5, E_max, 30000)
f = np.array([half_trace(e) for e in E])
allowed = np.abs(f) <= 1

def band_edges(E, f, v0=v0, w=w, b=b):
    # contiguous allowed intervals, with edges refined by root finding on (1/2) Tr M = +-1
    ok = np.abs(f) <= 1
    out, i = [], 0
    while i < len(E):
        if ok[i]:
            j = i
            while j + 1 < len(E) and ok[j + 1]:
                j += 1
            g = lambda e, s: half_trace(e, v0, w, b) - s
            lo = brentq(g, E[i - 1], E[i], args=(np.sign(f[i]),)) if i > 0 else E[0]
            hi = brentq(g, E[j], E[j + 1], args=(np.sign(f[j]),)) if j + 1 < len(E) else E[-1]
            out.append((lo, hi))
            i = j + 1
        else:
            i += 1
    return out

bands = band_edges(E, f)
for n, (lo, hi) in enumerate(bands, start=1):
    note = "   (continues above E_max)" if hi >= E[-1] else ""
    print(f"band {n}:  {lo:.4f} eV  to  {hi:.4f} eV   width {hi - lo:.4f} eV{note}")

plt.figure(figsize=(7, 3.4))
plt.plot(E, f)
plt.axhspan(-1, 1, color="C2", alpha=0.12)
plt.axvline(v0, color="0.6", ls=":", lw=1)
for lo, hi in bands:
    plt.plot([lo, hi], [-1.9, -1.9], lw=6, color="C0", solid_capstyle="butt")
plt.ylim(-2.2, 3); plt.xlabel("energy (eV)"); plt.ylabel("(1/2) Tr M")
plt.title(f"v0 = {v0} eV,  w = {w} nm,  b = {b} nm")
plt.show()
"""),
("md", r"""
### 3. יחס הנפיצה
בכל אנרגיה מותרת $k=\frac1a\arccos\left(\tfrac12\mathrm{Tr}\,M\right)$. משווים לאלקטרון חופשי, $\varepsilon=\frac{\hbar^2}{2m}(k+G)^2$, מקופל לאזור ברילואן הראשון.

> 🔮 **נחשו:** היכן יחס הנפיצה ייבדל הכי הרבה מזה של אלקטרון חופשי?
"""),
("code", """
k = np.arccos(np.clip(f[allowed], -1, 1)) / a
plt.figure(figsize=(5.5, 4))
plt.scatter(k * a / np.pi, E[allowed], s=1, color="C0")
plt.scatter(-k * a / np.pi, E[allowed], s=1, color="C0")
kk = np.linspace(-np.pi / a, np.pi / a, 300)
for n in range(-3, 4):
    plt.plot(kk * a / np.pi, C * (kk + 2 * np.pi * n / a) ** 2, "--", color="C1", lw=0.8)
plt.axhline(v0, color="0.6", ls=":", lw=1)
plt.ylim(0, E_max); plt.xlim(-1, 1)
plt.xlabel("k a / pi"); plt.ylabel("energy (eV)"); plt.title("band structure (dashed: free electron, folded)")
plt.show()
"""),
("md", r"""
### 4. רוחב הפס הנמוך ביותר
לכל עובי מחסום ברשימה `b_values` מוצאים את הפס הנמוך ביותר (מתחת לגובה המחסום) ומודדים את הרוחב שלו. הגרף בסקלה לוגריתמית.

> 🔮 **נחשו:** איך הרוחב יקטן כשהמחסום מתעבה: כמו חזקה של $b$, או אקספוננציאלית?
"""),
("code", """
def lowest_band(v0, w, b):
    Es = np.linspace(1e-5, 0.995 * v0, 20000)
    fs = np.array([half_trace(e, v0, w, b) for e in Es])
    found = band_edges(Es, fs, v0, w, b)
    return found[0] if found else (np.nan, np.nan)

widths, centres = [], []
print("   b (nm)    width (eV)    band centre (eV)")
for bb in b_values:
    lo, hi = lowest_band(v0, w, bb)
    widths.append(hi - lo); centres.append(0.5 * (lo + hi))
    print(f"   {bb:5.2f}     {hi - lo:.5f}       {0.5 * (lo + hi):.4f}")

plt.figure(figsize=(5.5, 3.4))
plt.semilogy(b_values, widths, "o-")
plt.xlabel("barrier thickness b (nm)"); plt.ylabel("width of lowest band (eV)")
plt.title(f"v0 = {v0} eV,  w = {w} nm")
plt.show()
"""),
("md", r"""
### 💬 שאלות לדיון
1. למה הדטרמיננטה של מטריצת המעבר שווה ל-1 בכל אנרגיה?
2. מה קורה ל-$\tfrac12\mathrm{Tr}\,M$ בדיוק בקצוות הפסים, ולאילו ערכים של $k$ זה מתאים?
3. למה הפסים הנמוכים צרים יותר, והפערים בין הפסים הגבוהים קטנים יותר?
4. מה הקשר בין רוחב הפס הנמוך ביותר לבין מודל הקשירה ההדוקה?

<details>
<summary>🗝️ הכוונה לתשובות (לחצו לפתיחה)</summary>

1. תרגול 9, סעיף 3 (הוורונסקיאן של שני פתרונות אינו משתנה לאורך התא).
2. תרגול 9, סעיף 3 (קצוות הפסים ב-$\mathrm{Tr}\,M=\pm2$).
3. תרגול 9, סעיף 4, והאיור של מודל הדלתא; תרגיל הבית, שאלה 3.
4. תרגול 9, סעיף 2 (פס של קשירה הדוקה ברוחב $4t$), ותרגיל הבית, שאלה 3.

אם אינכם יודעים לענות על אחת השאלות, חזרו לסעיף המתאים בתרגול.

</details>

> ✅ **סיימתם?** תשובות לשאלות הדיון אינן להגשה. הסעיף החישובי בתרגיל הבית נשען על המחברת הזו. בהצלחה!
"""),
]

# ------------------------------------------------------------------ week 10
w10 = [
("md", header(10, "שבוע 10 · אלקטרונים כמעט חופשיים: המשוואה המרכזית", "הדגמה לתרגול 10 · מבוא למצב מוצק תשפ״ז · אוניברסיטת תל אביב") + r"""
## תיאור ההדגמה
אלקטרון בשרשרת חד-ממדית עם קבוע סריג $a$ ופוטנציאל מחזורי של מחסומים מלבניים ברוחב $b$ ובגובה $v_0$ (מודל קרוניג-פני). כמו בתרגול, כותבים את פונקציית הגל כצירוף של גלים מישוריים $e^{i(k+G)x}$ עם $G=2\pi n/a$, ואז משוואת שרדינגר הופכת למשוואה מטריצית במרחב ה-$G$, "המשוואה המרכזית":
$$\left[\frac{\hbar^2}{2m}(k+G)^2-\varepsilon\right]c_{k+G}+\sum_{G'}V_{G-G'}\,c_{k+G'}=0 .$$
המחברת בונה את המטריצה הזו עם מספר סופי של גלים מישוריים ומלכסנת אותה לכל $k$, כלומר פותרת את הבעיה בלי תורת הפרעות, ומשווה את הפערים שמתקבלים לניבוי של תורת ההפרעות, $2|V_G|$.

**יחידות:** אורכים ב-nm ואנרגיות ב-eV, עם מסת אלקטרון חופשי: $\hbar^2/2m=0.0381\ \mathrm{eV\cdot nm^2}$.

לצד המחברת יש גם [דף אינטראקטיבי](https://yaminbens.github.io/SolidState/week10-nfe) עם אלקטרונים כמעט חופשיים בסריג ריבועי.
"""),
("code", """
import numpy as np
import matplotlib.pyplot as plt

C = 0.0381          # hbar^2 / (2 m_e) in eV nm^2
"""),
("md", r"""
### ⚙️ פרמטרים
- `a` – קבוע הסריג (nm).
- `b` – רוחב המחסום (nm).
- `v0` – גובה המחסום (eV).
- `n_G` – מספר הגלים המישוריים בבסיס (מספר אי-זוגי; יותר: מדויק יותר, איטי יותר).
- `n_gaps` – מספר הפערים שמחשבים ומציגים.
"""),
("code", """
# ---- Parameters ----
a = 1.0        # lattice constant (nm)
b = 0.3        # barrier width (nm)
v0 = 0.05      # barrier height (eV)
n_G = 41       # number of plane waves in the basis (odd)
n_gaps = 6     # number of gaps to compute
"""),
("md", r"""
### 1. רכיבי הפורייה של הפוטנציאל
עבור מחסום מלבני שמרכזו בראשית, $V_G=\frac1a\int_{-b/2}^{b/2}v_0\,e^{-iGx}dx$. מחשבים ומציירים את $|V_G|$ עבור $G=2\pi n/a$.

> 🔮 **נחשו:** האם $|V_G|$ קטן באופן מונוטוני כש-$n$ גדל?
"""),
("code", """
def V_G(n):
    # Fourier coefficient of a box barrier (height v0, width b, centred at 0) for G = 2 pi n / a
    if n == 0:
        return v0 * b / a
    G = 2 * np.pi * n / a
    return v0 * np.sin(G * b / 2) / (G * a / 2)

ns = np.arange(1, 13)
plt.figure(figsize=(6, 3))
plt.bar(ns, [abs(V_G(n)) * 1000 for n in ns], color="C0")
plt.xlabel("n  (G = 2 pi n / a)"); plt.ylabel("|V_G| (meV)"); plt.xticks(ns)
plt.title(f"b = {b} nm,  v0 = {v0} eV")
plt.show()
"""),
("md", r"""
### 2. פתרון המשוואה המרכזית
לכל $k$ באזור ברילואן הראשון בונים את המטריצה בגודל $n_G\times n_G$: על האלכסון $\frac{\hbar^2}{2m}(k+G)^2$, ומחוץ לאלכסון $V_{G-G'}$. הערכים העצמיים הם הפסים.

> 🔮 **נחשו:** היכן יחס הנפיצה ייראה שונה במיוחד מזה של אלקטרון חופשי?
"""),
("code", """
nlist = np.arange(-(n_G // 2), n_G // 2 + 1)
Vmat = np.array([[V_G(n - m) for m in nlist] for n in nlist])

def bands(k, nb=6):
    H = Vmat + np.diag(C * (k + 2 * np.pi * nlist / a) ** 2)
    return np.linalg.eigvalsh(H)[:nb]

ks = np.linspace(-np.pi / a, np.pi / a, 301)
E = np.array([bands(k) for k in ks])

plt.figure(figsize=(5.5, 4))
for j in range(E.shape[1]):
    plt.plot(ks * a / np.pi, E[:, j], color="C0")
for n in range(-3, 4):
    plt.plot(ks * a / np.pi, C * (ks + 2 * np.pi * n / a) ** 2, "--", color="C1", lw=0.8)
plt.ylim(0, E[:, -1].max()); plt.xlim(-1, 1)
plt.xlabel("k a / pi"); plt.ylabel("energy (eV)"); plt.title("bands (dashed: free electron, folded)")
plt.show()
"""),
("md", r"""
### 3. הפערים לעומת תורת ההפרעות
הפער ה-$n$, בין הפס ה-$n$ לפס ה-$n+1$, נפתח בקצה האזור ($k=\pi/a$) כאשר $n$ אי-זוגי ובמרכז האזור ($k=0$) כאשר $n$ זוגי. מחשבים אותו מהפתרון המלא ומשווים ל-$2|V_G|$ עם $G=2\pi n/a$.

> 🔮 **נחשו:** האם הפער המדויק יהיה תמיד קרוב ל-$2|V_G|$?
"""),
("code", """
E_zone_edge = bands(np.pi / a, n_gaps + 2)
E_zone_centre = bands(0.0, n_gaps + 2)
exact, predicted = [], []
print("  n     exact gap (meV)     2|V_G| (meV)")
for n in range(1, n_gaps + 1):
    e = E_zone_edge if n % 2 == 1 else E_zone_centre
    exact.append(e[n] - e[n - 1]); predicted.append(2 * abs(V_G(n)))
    print(f"  {n}       {1000 * exact[-1]:9.4f}          {1000 * predicted[-1]:9.4f}")

plt.figure(figsize=(6, 3.2))
nn = np.arange(1, n_gaps + 1)
plt.plot(nn, 1000 * np.array(exact), "o", label="exact (central equation)")
plt.plot(nn, 1000 * np.array(predicted), "_", ms=18, mew=2, label="2 |V_G|")
plt.xlabel("gap number n"); plt.ylabel("gap (meV)"); plt.xticks(nn); plt.legend(frameon=False)
plt.show()
"""),
("md", r"""
### 💬 שאלות לדיון
1. למה הפער ה-$n$ נפתח בקצה האזור כאשר $n$ אי-זוגי ובמרכזו כאשר $n$ זוגי?
2. מתי ההתאמה בין הפער המדויק ל-$2|V_G|$ טובה, ומתי היא פחות טובה?
3. מה היה קורה לפערים אם הפוטנציאל היה סכום של פונקציות דלתא?

<details>
<summary>🗝️ הכוונה לתשובות (לחצו לפתיחה)</summary>

1. תרגול 10, סעיף 1 (תנאי הניוון $|k|=|k+G|$ והקיפול לאזור הראשון), ותרגול 9, סעיף 4.
2. תרגול 10, סעיף 1 (התנאי לתקפות תורת ההפרעות, $|V_G|\ll$ ההפרש בין האנרגיות החופשיות).
3. תרגיל הבית, שאלה 3.

אם אינכם יודעים לענות על אחת השאלות, חזרו לסעיף המתאים בתרגול.

</details>

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
    build("w7", w7, ROOT / "week07.ipynb")
    build("w8", w8, ROOT / "week08.ipynb")
    build("w9", w9, ROOT / "week09.ipynb")
    build("w10", w10, ROOT / "week10.ipynb")
