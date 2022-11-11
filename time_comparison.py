from models import *
from systematic_comparison import *


def construct_study(
    estimator: Estimator,
    feature_group: str,
    features: Iterable[str],
    scii: SCIData,
    sci_train_idx: SCIData,
    sci_test_idx: SCIData,
    outcome_within: int,
    cv=5,
    scoring="average_precision",
    storage=None,
    model_persistence_path=None,
    cv_jobs=1,
    n_trials=100,
    **kwargs,
):
    scii = (
        scii.derive_critical_event(within=outcome_within).omit_redundant().categorize()
    )
    X_train, y_train, X_test, y_test = estimator.get_xy(
        SCIData(scii.loc[sci_train_idx]), SCIData(scii.loc[sci_test_idx]), features
    )

    name = f"{estimator._name}_Within-{outcome_within}_{feature_group}"
    study = optuna.create_study(
        direction="maximize", study_name=name, storage=storage, load_if_exists=True
    )

    pipeline_factory = PipelineFactory(
        estimator=estimator, resampler=None, X_train=X_train, y_train=y_train,
    )

    objective = Objective(
        estimator=estimator,
        resampler=None,
        pipeline_factory=pipeline_factory,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scoring=scoring,
        cv_jobs=cv_jobs,
        n_trials=n_trials,
        stop_callback=study.stop,
    )

    def handle_study_result(model_persistence_path=None, n_resamples=99, **kwargs):
        params = study.best_params

        X, y = X_train, y_train
        if estimator._requirements["oneclass"]:
            X = SCIData(X_train[y_train.eq(0)])
            y = y_train[y_train.eq(0)]

        explanations = []
        if estimator._requirements["calibration"]:
            model = CalibratedClassifierCV(
                pipeline_factory(**params), cv=cv, method="isotonic", n_jobs=cv_jobs,
            ).fit(X, y)
            if estimator._requirements["explanation"]:
                explanations = estimator.explain_calibrated(model, X, X_test)
        else:
            model = pipeline_factory(**params).fit(X, y)
            if estimator._requirements["explanation"]:
                explanations = estimator.explain(model[estimator._name], X, X_test)

        if model_persistence_path is not None:
            with open(f"{model_persistence_path}/{name}.bin", "wb") as file:
                pickle.dump((model, explanations), file)

        metrics, y_pred_proba = evaluate_model(model, X_test, y_test, n_resamples)

        metrics = (
            dict(
                name=name,
                estimator=estimator._name,
                outcome_within=outcome_within,
                features=feature_group,
            )
            | metrics
        )

        return metrics, (name, y_pred_proba)

    def call(model_persistence_path=None, n_resamples=99, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            while True:
                try:
                    study.optimize(objective, **kwargs)
                    return handle_study_result(model_persistence_path, n_resamples)
                except (
                    optuna.exceptions.StorageInternalError,
                    sqlalchemy.exc.OperationalError,
                    AssertionError,
                ):
                    print("################# CAUGHT DB ERROR #################")
                    pass

    return call


def run(args):
    args = vars(args)
    scii = (
        SCIData(
            SCIData.quickload("data/sci_processed.h5").sort_values("AdmissionDateTime")
        )
        .mandate(SCICols.news_data_raw)
        .augment_shmi(onehot=True)
        .derive_ae_diagnosis_stems(onehot=False)
    )

    sci_train, sci_test = train_test_split(
        scii.omit_redundant().categorize(),
        test_size=0.33,
        random_state=42,
        shuffle=False,
    )
    sci_train, sci_test = SCIData(sci_train), SCIData(sci_test)
    sci_train_idx, sci_test_idx = sci_train.index, sci_test.index

    if args["verbose"]:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    if args["persist"] is not None:
        try:
            os.makedirs(args["persist"])
        except FileExistsError:
            pass

    if args["debug"]:
        sci_train_idx = sci_train_idx[:1000]
        sci_test_idx = sci_test_idx[:1000]

    if args["storage"] is not None:
        args["storage"] = optuna.storages.RDBStorage(
            url=args["storage"], engine_kwargs={"connect_args": {"timeout": 100}}
        )

    n_trials = args["trials"] if not args["debug"] else 2

    studies = [
        construct_study(
            **_,
            **args,
            outcome_within=threshold,
            scii=scii,
            sci_train_idx=sci_train_idx,
            sci_test_idx=sci_test_idx,
            n_trials=n_trials,
        )
        for _ in study_grid_from_args(args, sci_train)[:2]
        for threshold in range(1, 3)
    ]

    study_args = dict(
        model_persistence_path=args["persist"],
        n_resamples=args["n_resamples"],
        n_trials=n_trials,
        timeout=args["hours"] * 60 * 60,
    )

    if args["njobs"] > 1:
        print("Starting execution (parallel)")
        with parallel_backend("loky", inner_max_num_threads=args["cv_jobs"]):
            results = Parallel(n_jobs=args["njobs"])(
                delayed(_)(**study_args) for _ in studies
            )
    else:
        print("Starting execution (linear)")
        results = [_(**study_args) for _ in studies]

    metrics, y_preds = list(zip(*results))
    pd.DataFrame(metrics).to_hdf(args["output"], "metrics")
    pd.DataFrame(dict(y_preds)).to_hdf(args["output"], "y_preds")


if __name__ == "__main__":
    run(parser.parse_args())
