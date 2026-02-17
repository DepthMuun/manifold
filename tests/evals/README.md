Arquitectura de testing de LLMs (MMLU, GSM8K, Contexto Largo)

Estructura

- tests/evals/common
  - adapters.py: interfaz BaseAdapter para conectar cualquier modelo. Incluye RandomAdapter y OracleAdapter de referencia.
  - logging_utils.py: logger con niveles y rotación automática.
  - metrics.py: precisión, P/R/F1, perplexity, uso de memoria y utilidades.
  - seeds.py: semillas y determinismo.
  - storage.py: exportación JSONL/CSV.
- tests/evals/mmlu/evaluator.py: evaluación MMLU con métricas por categoría y reporte.
- tests/evals/gsm8k/evaluator.py: evaluación GSM8K con logging paso a paso y métricas.
- tests/evals/longcontext
  - ruler.py: evaluación tipo RULER sobre documentos largos.
  - needle.py: evaluación Needle-in-Haystack con posición de needle y recuperación.
- tests/evals/visualize/live_console.py: visualización “en tiempo real” por consola leyendo JSONL.
- tests/data: muestras mínimas (mmlu_sample.jsonl, gsm8k_sample.jsonl).
- tests/unit/test_eval_suites.py: pruebas unitarias E2E de las suites.

Cómo integrar un modelo

1) Implementa BaseAdapter con tres métodos opcionales:
   - predict_mmlu(question:str, choices:list[str]) -> (pred_idx:int, confidence:float, meta:dict)
   - predict_gsm8k(question:str) -> (predicted_answer:str, reasoning_steps:list[str], confidence:float, meta:dict)
   - retrieve_from_context(context:str, prompt:str) -> (retrieved:str, confidence:float, meta:dict)
2) Pasa la instancia del adapter a la clase evaluadora correspondiente.

Ejecución rápida (datasets de ejemplo)

python -c "import os, json; from tests.evals.common.adapters import RandomAdapter, OracleAdapter; from tests.evals.mmlu.evaluator import MMLUEvaluator; from tests.evals.gsm8k.evaluator import GSM8KEvaluator; from tests.evals.longcontext.ruler import RULEREvaluator, synth_doc; from tests.evals.longcontext.needle import NeedleEvaluator; base='D:/ASAS/manifold_mini/manifold_working'; out=os.path.join(base,'_eval_out'); os.makedirs(out, exist_ok=True); print('MMLU', json.dumps(MMLUEvaluator(RandomAdapter(), os.path.join(out,'mmlu')).run(os.path.join(base,'tests','data','mmlu_sample.jsonl')))); print('GSM8K', json.dumps(GSM8KEvaluator(RandomAdapter(), os.path.join(out,'gsm8k')).run(os.path.join(base,'tests','data','gsm8k_sample.jsonl')))); r=RULEREvaluator(OracleAdapter(), os.path.join(out,'ruler')); doc=synth_doc(2,10)+'\\nNEEDLE:token123\\n'; print('RULER', json.dumps(r.run([(doc,'Where is the token?')]))); n=NeedleEvaluator(OracleAdapter(), os.path.join(out,'needle')); print('NEEDLE', json.dumps(n.run([('a'*1000,'secret',500)])))"

Salidas

- JSONL por muestra y CSV por benchmark (carpeta _eval_out/{mmlu,gsm8k,ruler,needle}).
- Reporte JSON agregado con métricas: accuracy, precision, recall, F1, perplexity (si aplica), latencia media, memoria y métricas específicas por suite.
- Logs estructurados con timestamps y rotación ({suite}.log).

Visualización en tiempo real

python tests/evals/visualize/live_console.py --file _eval_out/mmlu/mmlu_results.jsonl --interval 1.0

Reproducibilidad

- set_global_seed y fix_determinism fijan semillas y opciones deterministas.

Resultados esperados (datasets de ejemplo)

- MMLU (RandomAdapter): accuracy ~0.25 en promedio; en el sample de 2 preguntas varía entre 0.0 y 1.0.
- GSM8K (RandomAdapter): 0% acierto en los ejemplos provistos.
- RULER/Needle (OracleAdapter): 100% recuperación con latencia baja.
