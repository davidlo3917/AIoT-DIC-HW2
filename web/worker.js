// 即時重跑：在 Web Worker 裡用 Pyodide 執行和 CI 同一份 regress.py
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

let booting;
async function boot() {
  postMessage({ type: "status", text: "下載 Python 執行環境…" });
  const py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/" });
  postMessage({ type: "status", text: "安裝 scikit-learn 與 pandas…" });
  await py.loadPackage(["scikit-learn", "pandas"]);
  py.FS.writeFile("/home/pyodide/regress.py", await (await fetch("regress.py")).text());
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  return py.pyimport("regress");
}

onmessage = async ({ data: d }) => {
  try {
    const regress = await (booting ||= boot());
    const t = performance.now();
    const json = regress.run_json(d.csv, d.dataset, d.cfg, d.methods, d.models, (done, total) => postMessage({ type: "progress", done, total }));
    postMessage({ type: "done", json, secs: ((performance.now() - t) / 1000).toFixed(1) });
  } catch (e) {
    booting = null;
    postMessage({ type: "error", message: String(e?.message || e).trim().split("\n").pop() });
  }
};
