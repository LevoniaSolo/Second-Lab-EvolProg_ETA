"""ЛР2, вариант 17. Только стандартная библиотека + matplotlib для графиков."""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

N_TASKS, N_CPUS = 30, 5
POP, BUDGET, RUNS = 80, 8000, 20
DATA_SEED = 17026


def generate_data():
    r = random.Random(DATA_SEED)
    times = [r.randint(3, 15) for _ in range(N_TASKS)]
    # Несколько обязательных назначений, остальные задачи могут идти на любой процессор.
    fixed = {0: 0, 7: 2, 19: 4}
    # Вместимость каждого процессора, подобранная с запасом относительно средней загрузки.
    cap = math.ceil(sum(times) / N_CPUS * 1.35)
    return {"variant": 17, "seed": DATA_SEED, "times": times,
            "processors": N_CPUS, "capacity": [cap] * N_CPUS,
            "fixed_assignments": {str(k): v for k, v in fixed.items()}}


def loads_of(x, data):
    loads = [0] * N_CPUS
    for i, p in enumerate(x):
        loads[p] += data['times'][i]
    return loads


def violations(x, data):
    loads = loads_of(x, data)
    overflow = sum(max(0, load - cap) for load, cap in zip(loads, data['capacity']))
    wrong_fixed = sum(x[int(i)] != p for i, p in data['fixed_assignments'].items())
    return overflow, wrong_fixed


def feasible(x, data):
    return violations(x, data) == (0, 0)


def score(x, data, mode):
    loads = loads_of(x, data)
    overflow, wrong = violations(x, data)
    # Штраф за 1 единицу превышения и за нарушение обязательного назначения.
    return max(loads) + 20 * overflow + 1000 * wrong if mode == 'penalty' else max(loads)


def repair(x, data, rng):
    """Сохраняем обязательные назначения, затем переносим задачи с перегруженных CPU.
    Если локальный ремонт застрял, перестраиваем распределение жадно.
    """
    x = x.copy()
    fixed = {int(i): p for i, p in data['fixed_assignments'].items()}
    for i, p in fixed.items():
        x[i] = p
    for _ in range(N_TASKS * N_CPUS):
        loads = loads_of(x, data)
        overloaded = [p for p in range(N_CPUS) if loads[p] > data['capacity'][p]]
        if not overloaded:
            return x
        moves = []
        for i, old in enumerate(x):
            if i in fixed or old not in overloaded:
                continue
            t = data['times'][i]
            for new in range(N_CPUS):
                if new == old or loads[new] + t > data['capacity'][new]:
                    continue
                new_loads = loads.copy()
                new_loads[old] -= t
                new_loads[new] += t
                overflow = sum(max(0, z - cap) for z, cap in zip(new_loads, data['capacity']))
                moves.append((overflow, max(new_loads), i, new))
        if not moves:
            break
        _, _, i, new = min(moves)
        x[i] = new
    # Гарантированно допустимая реконструкция для данного набора данных.
    # Сначала обязательные задачи, остальные — по убыванию длительности.
    x = [-1] * N_TASKS
    loads = [0] * N_CPUS
    for i, p in fixed.items():
        x[i] = p
        loads[p] += data['times'][i]
    indices = [i for i in range(N_TASKS) if i not in fixed]
    indices.sort(key=lambda i: (-data['times'][i], i))
    for i in indices:
        t = data['times'][i]
        options = [p for p in range(N_CPUS) if loads[p] + t <= data['capacity'][p]]
        if not options:
            raise RuntimeError('Для заданных данных жадная реконструкция не нашла допустимого решения')
        p = min(options, key=lambda p: (loads[p], p))
        x[i] = p
        loads[p] += t
    return x


def random_chromosome(rng):
    return [rng.randrange(N_CPUS) for _ in range(N_TASKS)]


def crossover(a, b, rng):
    # Равномерный кроссинговер корректен для вектора назначений.
    return [a[i] if rng.random() < .5 else b[i] for i in range(N_TASKS)]


def mutate(x, kind, rng):
    x = x.copy()
    if kind == 'move':
        i = rng.randrange(N_TASKS)
        choices = [p for p in range(N_CPUS) if p != x[i]]
        x[i] = rng.choice(choices)
    elif kind == 'swap':
        i, j = rng.sample(range(N_TASKS), 2)
        x[i], x[j] = x[j], x[i]
    else:
        raise ValueError(kind)
    return x


def tournament(pop, fitness, rng):
    return pop[min((rng.randrange(len(pop)) for _ in range(3)), key=lambda i: fitness[i])]


def run_ga(data, mode, mutation, seed, budget=BUDGET):
    rng = random.Random(seed)
    pop = [random_chromosome(rng) for _ in range(POP)]
    if mode == 'repair':
        pop = [repair(x, data, rng) for x in pop]
    fits = [score(x, data, mode) for x in pop]
    evaluations = len(pop)
    best_x, best_val = None, math.inf
    history = []

    def update_best():
        nonlocal best_x, best_val
        for x in pop:
            if feasible(x, data):
                val = max(loads_of(x, data))
                if val < best_val:
                    best_x, best_val = x.copy(), val

    update_best()
    history.append((evaluations, best_val if best_x is not None else None))
    while evaluations < budget:
        order = sorted(range(len(pop)), key=lambda i: fits[i])
        new_pop = [pop[i].copy() for i in order[:2]]  # элитизм
        while len(new_pop) < POP and evaluations < budget:
            a = tournament(pop, fits, rng)
            b = tournament(pop, fits, rng)
            child = crossover(a, b, rng) if rng.random() < .9 else a.copy()
            if rng.random() < .3:
                child = mutate(child, mutation, rng)
            if mode == 'repair':
                child = repair(child, data, rng)
            new_pop.append(child)
            evaluations += 1
        # Бюджет кратен POP, поэтому размер популяции здесь всегда POP.
        pop = new_pop
        fits = [score(x, data, mode) for x in pop]
        update_best()
        history.append((evaluations, best_val if best_x is not None else None))
    return {"best": best_val, "assignment": best_x, "history": history,
            "feasible": best_x is not None}


def run_baseline(data, seed, budget=BUDGET):
    """Случайный допустимый поиск с равным бюджетом проверенных кандидатов."""
    rng = random.Random(seed)
    best_x, best_val = None, math.inf
    history = []
    for k in range(1, budget + 1):
        x = repair(random_chromosome(rng), data, rng)
        val = max(loads_of(x, data))
        if val < best_val:
            best_x, best_val = x, val
        if k % POP == 0:
            history.append((k, best_val))
    return {"best": best_val, "assignment": best_x, "history": history,
            "feasible": True}


def examples(data):
    good1 = repair([i % N_CPUS for i in range(N_TASKS)], data, random.Random(1))
    good2 = repair([(i * 3 + 1) % N_CPUS for i in range(N_TASKS)], data, random.Random(2))
    bad1 = [0] * N_TASKS  # перегрузка и неверные обязательные назначения
    bad2 = good1.copy()
    bad2[0] = 1  # нарушено обязательное назначение задачи 0 -> CPU 0
    return [('valid_1', good1), ('valid_2', good2),
            ('invalid_1', bad1), ('invalid_2', bad2)]


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def draw_plots(results, output):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib не установлен: графики пропущены (pip install matplotlib)')
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, runs in results.items():
        lengths = min(len(r['history']) for r in runs)
        xs = [runs[0]['history'][i][0] for i in range(lengths)]
        means = []
        for i in range(lengths):
            values = [r['history'][i][1] for r in runs if r['history'][i][1] is not None]
            means.append(statistics.mean(values) if values else math.nan)
        ax.plot(xs, means, label=name)
    ax.set(xlabel='Вычисления целевой функции', ylabel='Среднее лучшее допустимое значение',
           title='Сходимость по 20 независимым запускам')
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'convergence.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(results)
    ax.boxplot([[r['best'] for r in results[name]] for name in labels])
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set(ylabel='Максимальная загрузка', title='Распределение результатов по запускам')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(output / 'boxplot.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Быстрый тест: 3 запуска по 800 оценок')
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / 'results'
    output.mkdir(exist_ok=True)
    data = generate_data()
    (output / 'instance.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    # Проверяем, что данные допускают жадное построение.
    repair([0] * N_TASKS, data, random.Random(0))
    exrows = []
    for label, x in examples(data):
        overflow, wrong = violations(x, data)
        exrows.append(dict(example=label, feasible=feasible(x, data),
                           max_load=max(loads_of(x, data)), overflow=overflow,
                           wrong_fixed=wrong, assignment=' '.join(map(str, x))))
    write_csv(output / 'examples.csv', exrows, list(exrows[0]))

    runs = 3 if args.quick else RUNS
    budget = 800 if args.quick else BUDGET
    settings = [('GA_repair_move', 'repair', 'move'),
                ('GA_penalty_move', 'penalty', 'move'),
                ('GA_repair_swap', 'repair', 'swap')]
    results, rows = {}, []
    for name, mode, mutation in settings:
        results[name] = []
        for k in range(runs):
            r = run_ga(data, mode, mutation, 10000 + k, budget)
            results[name].append(r)
            rows.append(dict(method=name, run=k + 1, seed=10000 + k,
                             best=r['best'] if r['feasible'] else '',
                             feasible=r['feasible']))
        print(name, 'лучшее:', min(r['best'] for r in results[name]))
    results['Random_feasible'] = []
    for k in range(runs):
        r = run_baseline(data, 20000 + k, budget)
        results['Random_feasible'].append(r)
        rows.append(dict(method='Random_feasible', run=k + 1, seed=20000 + k,
                         best=r['best'], feasible=True))
    write_csv(output / 'runs.csv', rows, ['method', 'run', 'seed', 'best', 'feasible'])
    summaries = []
    for name, rr in results.items():
        vals = [r['best'] for r in rr if r['feasible']]
        summaries.append(dict(method=name, runs=runs, feasible_runs=len(vals),
                              best=min(vals) if vals else '',
                              mean=round(statistics.mean(vals), 4) if vals else '',
                              median=statistics.median(vals) if vals else '',
                              std=round(statistics.stdev(vals), 4) if len(vals) > 1 else 0,
                              worst=max(vals) if vals else ''))
    write_csv(output / 'summary.csv', summaries, list(summaries[0]))
    # Лучшее решение среди всех ГА, с понятным распределением по процессорам.
    candidates = [(r['best'], name, r['assignment']) for name, rr in results.items()
                  if name.startswith('GA') for r in rr if r['feasible']]
    if candidates:
        val, name, x = min(candidates, key=lambda t: t[0])
        allocation = [{"processor": p, "tasks_1_based": [i + 1 for i, q in enumerate(x) if q == p],
                       "load": loads_of(x, data)[p], "capacity": data['capacity'][p]}
                      for p in range(N_CPUS)]
        (output / 'best_solution.json').write_text(json.dumps(
            {"method": name, "max_load": val, "assignment": x,
             "allocation": allocation}, ensure_ascii=False, indent=2), encoding='utf-8')
    draw_plots(results, output)
    print('Результаты:', output)


if __name__ == '__main__':
    main()
