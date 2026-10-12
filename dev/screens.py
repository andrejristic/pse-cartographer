"""Render every tab offscreen to PNGs (dev check of the GUI over an existing data folder).
   PSE_CARTOGRAPHER_DATA=<dir> QT_QPA_PLATFORM=offscreen python dev/screens.py <out_dir> [model.gguf]"""
import os, sys, faulthandler
faulthandler.dump_traceback_later(int(os.environ.get("HANG_DUMP", "0")) or 10**6, exit=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from PySide6.QtWidgets import QApplication
from cartographer.gui.main_window import CartographerWindow
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
app = QApplication(sys.argv[:1])
w = CartographerWindow(first_run=False)
w.resize(1500, 950); w.show()
if len(sys.argv) > 2:
    w.model_tab.open_path(sys.argv[2])
app.processEvents()
for i in range(w.tabs.count()):
    w.tabs.setCurrentIndex(i)
    for _ in range(5): app.processEvents()
    name = w.tabs.tabText(i).replace("★ ", "").replace(" ", "_").replace("&", "and")
    w.grab().save(str(out / f"{i:02d}_{name}.png"))
    if w.tabs.widget(i) is w.cartographer:
        for j in range(w.cartographer.tabs.count()):
            w.cartographer.tabs.setCurrentIndex(j)
            for _ in range(5): app.processEvents()
            w.grab().save(str(out / f"{i:02d}_{name}_{j}_{w.cartographer.tabs.tabText(j).replace(' ', '').replace('/', '')}.png"))
w.close()
print("saved", len(list(out.glob('*.png'))))
