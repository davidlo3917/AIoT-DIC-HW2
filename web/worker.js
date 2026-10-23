// Pyodide in a Web Worker: the same regress.py as CI, so the browser and CI compute the same numbers.
// In:  {type: "boot"} | {type: "plan", file: ArrayBuffer} | {type: "run", file: ArrayBuffer, plan?: object}
// Out: {type: "status", text} | {type: "progress", done, total} | {type: "ready"}
//      {type: "plan" | "run", json, secs} | {type: "error", message, data: true when the file is the problem}
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

let ready;
async function start() {
  postMessage({ type: "status", text: "Downloading the Python runtime…" });
  const py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/" });
  postMessage({ type: "status", text: "Installing scikit-learn and pandas…" });
  await py.loadPackage(["scikit-learn", "pandas"]);
  py.FS.writeFile("/home/pyodide/regress.py", await (await fetch("regress.py")).text());
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  return { py, regress: py.pyimport("regress") };
}
const boot = () => (ready ||= start().catch((e) => { ready = null; throw e; }));   // a failed download can be retried

onmessage = async ({ data: m }) => {
  try {
    const { py, regress } = await boot();
    if (m.type === "boot") return postMessage({ type: "ready" });
    const path = "/tmp/upload.csv", t = performance.now();
    py.FS.writeFile(path, new Uint8Array(m.file));
    const json = m.type === "plan" ? regress.plan_json(path)
      : regress.run_json(path, m.plan ? JSON.stringify(m.plan) : null, (done, total) => postMessage({ type: "progress", done, total }));
    postMessage({ type: m.type, json, secs: (performance.now() - t) / 1000 });
  } catch (e) {
    const last = String(e?.message || e).trim().split("\n").pop();   // a Python traceback ends with "Type: message"
    const data = last.match(/DataError: (.*)$/);
    postMessage({ type: "error", data: !!data, message: data ? data[1] : `The analysis failed: ${last}` });
  }
};
