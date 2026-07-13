from shapcrn.utils.sbml import io as sbml_io


def test_load_and_prepare_runs_reversible_reaction_preparation(monkeypatch, model_path):
    called = []

    def record(model, log_file=None):
        called.append(model)
        return model

    monkeypatch.setattr(sbml_io.sbml_react, "split_all_reversible_reactions", record)
    document, model = sbml_io.load_and_prepare_model(str(model_path))

    assert document.getModel().getId() == model.getId()
    assert len(called) == 1
    assert called[0].getId() == model.getId()
