# מבוא למצב מוצק תשפ"ז – הדגמות נומריות

הדגמות שבועיות קצרות (Jupyter) לתרגולי הקורס, מפורסמות כאתר Jupyter Book:
**https://yaminbens.github.io/SolidState/**

## מבנה
- `weekNN/demo.ipynb` – מחברת ההדגמה של שבוע NN (עם פלטים שמורים). `weekNN/index.md` – עמוד פתיחה עם כפתור Colab.
- `myst.yml`, `index.md`, `site.css` – הגדרות האתר (Jupyter Book 2 / MyST).
- `tools/make_notebooks.py` – יוצר ומריץ את המחברות (אפשר להוסיף שם שבועות חדשים).
- `.github/workflows/deploy.yml` – בונה ומפרסם את האתר ב-GitHub Pages בכל push ל-`main`.

## הפעלה חד-פעמית
ב-GitHub: *Settings → Pages → Build and deployment → Source: GitHub Actions*.

## בנייה מקומית
```
pip install "jupyter-book>=2"
python tools/make_notebooks.py     # (מחדש את המחברות)
jupyter-book build --html          # האתר ב-_build/html
```
