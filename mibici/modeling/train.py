"""Despacho común por pregunta; los runners conservan sus contratos y métricas."""

from mibici.dataset import DataError


def dispatch(question, model, parameters, manifest, data_dir, report_dir, parent_id, commit):
    if question == "p1":
        from mibici.modeling.p1_candidates import run_candidates
        return run_candidates(model, parameters, manifest, data_dir, report_dir, parent_id, commit)
    if question == "p2":
        from mibici.modeling.p2 import run_p2
        return run_p2(manifest_path=manifest, data_dir=data_dir, report_dir=report_dir,
                      model_name=model, model_parameters=parameters)
    if question == "p3":
        from mibici.modeling.p3 import run_p3
        if model != "kmeans":
            raise DataError("P3 admite kmeans; su responsable incorporará otros candidatos.")
        return run_p3(manifest_path=manifest, data_dir=data_dir, report_dir=report_dir,
                      model_parameters=parameters)
    raise DataError(f"Pregunta desconocida: {question}.")
