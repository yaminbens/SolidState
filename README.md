# מבוא למצב מוצק תשפ"ז – הדגמות נומריות

הדגמות שבועיות קצרות (Jupyter) לתרגולי הקורס, מפורסמות כאתר Jupyter Book:
**https://yaminbens.github.io/SolidState/**

## מבנה
- `weekNN.ipynb` – מחברת ההדגמה של שבוע NN (עם פלטים שמורים). עמוד אחד לשבוע באתר: `https://yaminbens.github.io/SolidState/weekNN`.
- `myst.yml`, `index.md`, `site.css` – הגדרות האתר (Jupyter Book 2 / MyST), עמוד הבית והעיצוב.
- `tools/make_notebooks.py` – יוצר ומריץ את המחברות (אפשר להוסיף שם שבועות חדשים).
- `tools/prepare_site.py` – רץ רק ב-CI: מסיר מהמחברות את עטיפת ה-`<div dir="rtl">` (שמשבשת נוסחאות באתר); במחברת עצמה היא נשארת כדי שיהיה RTL ב-Colab.
- `.github/workflows/deploy.yml` – בונה ומפרסם את האתר ב-GitHub Pages בכל push ל-`main`.

## הפעלה חד-פעמית
ב-GitHub: *Settings → Pages → Build and deployment → Source: GitHub Actions*.

## בנייה מקומית
```
pip install "jupyter-book>=2"
python tools/make_notebooks.py     # (מחדש את המחברות)
jupyter-book build --html          # האתר ב-_build/html
```

## הרצת הקוד באתר

בכל עמוד שבועי יש כפתור הפעלה (Binder) שמריץ את המחברת בענן. `requirements.txt` מגדיר את החבילות.
המחברות נשמרות ללא פלט (unrun), כך שהתוצאות מופיעות רק אחרי שמריצים את הקוד.
