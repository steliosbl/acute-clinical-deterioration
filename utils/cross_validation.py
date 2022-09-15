import csv, os.path
from multiprocessing import Pool
from functools import partial
from sklearn.model_selection import cross_validate


def cross_validate_linear(models, X, y, scoring, cv=3):
    keys, values = zip(*models.items())
    return dict(
        zip(
            keys, map(partial(cross_validate, X=X, y=y, cv=cv, scoring=scoring, n_jobs=-1), values)
        )
    )


def cross_validate_parallel(models, X, y, scoring, cv=3, threads=None):
    if not threads:
        threads = len(models)

    with Pool(threads) as p:
        keys, values = zip(*models.items())
        result = zip(
            keys,
            p.map(partial(cross_validate, X=X, y=y, cv=cv, scoring=scoring), values),
        )

    return dict(result)


def _run_cv(kv, X, y, scoring, cv):
    print(f"STARTING {kv[0]}", flush=True)
    try:
        r = cross_validate(kv[1], X=X, y=y, cv=cv, scoring=scoring)
        print(f"FINISHED {kv[0]}", flush=True)
        return [
            {"model": kv[0], "err": False, **{k: v[i] for k, v in r.items()}}
            for i in range(cv)
        ]
    except Exception as e:
        return [{"model": kv[0], "err": e}]


def cross_validate_parallel_file(filename, models, X, y, scoring, cv, threads=None):
    threads = threads or len(models)
    fieldnames = ["model", "err", "fit_time", "score_time"] + [
        f"test_{_}" for _ in scoring.keys()
    ]

    file_exists = os.path.isfile(filename)
    with open(filename, "a") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        with Pool(threads) as p:
            for results in p.imap_unordered(
                partial(_run_cv, X=X, y=y, scoring=scoring, cv=cv), models.items()
            ):
                writer.writerows(results)
