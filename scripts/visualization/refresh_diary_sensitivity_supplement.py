"""Refresh organized sensitivity tables S3/S4/S7/S8 from aggregate CSVs only.

Allows added diary log-NA rows anchored to existing raw-NA associations.
Preserves all other tables, figures and DOCX package components. Run from the
project root with --document PATH. Does not fit models or read participant data.
"""
import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from add_supplement_method_comparisons import NS, label, number, interval, pvalue
from reorganize_supplement import G, TITLES, table


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(path):
    parts = path.parts
    tail = parts[parts.index('tables') + 1:-1]
    return ' / '.join(G.get(x, x.replace('_', ' ')) for x in tail)


def cells(row, method):
    coefficient = 'r' if method == 'Pearson' else 'beta'
    symbol = 'r' if method == 'Pearson' else 'β'
    return (f"N = {row['n']}; {symbol} = {number(row[coefficient])}\n"
            f"95% CI {interval(row)}\np (analysis) {pvalue(row['p'])}; "
            f"p (2-sided) {pvalue(row['p_two_tailed'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--document', type=Path, required=True)
    args = ap.parse_args()
    root = Path.cwd()
    fresh = {i: {} for i in [3, 4, 7, 8]}
    additions = {i: {} for i in fresh}
    sources, comparison_sources = {}, {}
    for prefix, primary, primary_table, comparison_table in [
            ('', 'mlm.csv', 3, 7), ('MR1_validation', 'regressions.csv', 4, 8)]:
        for family in ['02_persistence_affect', '04_fc_affect']:
            for directory in sorted((root / prefix / 'results/tables' / family).glob('sensitivity_*')):
                datasets = {}
                for filename in set([primary, 'regressions.csv', 'correlations.csv']):
                    path = directory / filename
                    rows = read(path)
                    datasets[filename] = {(r['predictor'], r['outcome']): r for r in rows}
                    assert len(rows) == len(datasets[filename]), 'Duplicate model keys'
                    sources[str(path.relative_to(root))] = sha(path)
                    if filename != 'mlm.csv':
                        comparison_sources[str(path.relative_to(root))] = (
                            'OLS' if filename == 'regressions.csv' else 'Pearson', rows)
                keys = set(datasets[primary])
                assert all(set(d) == keys for d in datasets.values()), 'Methods have different model identities'
                base = group(directory / primary)
                for key, row in datasets[primary].items():
                    predictor, outcome = key
                    g = base + (' / log-NA robustness' if outcome.endswith('_log') else '')
                    primary_row = [g, label(predictor), label(outcome), row['n'], number(row['beta']),
                                   interval(row), pvalue(row['p']), pvalue(row['p_two_tailed'])]
                    comparison_row = [g, label(predictor) + ' → ' + label(outcome),
                                      cells(datasets['regressions.csv'][key], 'OLS'),
                                      cells(datasets['correlations.csv'][key], 'Pearson')]
                    for index, values, width in [(primary_table, primary_row, 3), (comparison_table, comparison_row, 2)]:
                        identity = tuple(values[:width])
                        assert identity not in fresh[index]
                        fresh[index][identity] = values
                        if outcome == 'NA_score_log':
                            raw_identity = ((base, label(predictor), label('NA_score')) if width == 3 else
                                            (base, label(predictor) + ' → ' + label('NA_score')))
                            additions[index][raw_identity] = identity
    with ZipFile(args.document) as z:
        entries = [(i, z.read(i.filename)) for i in z.infolist()]
    tree = E.fromstring(dict((i.filename, b) for i, b in entries)['word/document.xml'])
    body = tree.find('w:body', NS)
    tables = body.findall('w:tbl', NS)
    assert len(tables) == 11
    text = ''.join(tree.xpath('//w:t/text()', namespaces=NS))
    assert all(f'Table S{i}. {TITLES[i]}' in text for i in fresh)
    untouched = {i: E.tostring(t) for i, t in enumerate(tables, 1) if i not in fresh}
    counts = {}
    for index in fresh:
        old = tables[index-1]
        rows = [[ '\n'.join(''.join(p.xpath('.//w:t/text()', namespaces=NS)) for p in c.findall('w:p', NS))
                  for c in tr.findall('w:tc', NS)] for tr in old.findall('w:tr', NS)]
        width = 3 if index < 5 else 2
        existing = {tuple(r[:width]) for r in rows[1:]}
        assert len(existing) == len(rows)-1
        allowed_new = set(additions[index].values()) - existing
        assert set(fresh[index]) - existing == allowed_new, 'Unexpected new association'
        output, used = [], set()
        for row in rows[1:]:
            identity = tuple(row[:width])
            output.append(fresh[index].get(identity, row))
            if identity in fresh[index]:
                used.add(identity)
            added = additions[index].get(identity)
            if added in allowed_new:
                output.append(fresh[index][added]); used.add(added)
        assert used == set(fresh[index]), 'Missing raw-NA anchor or model row'
        widths = [2450,2450,1300,450,800,2000,950,1000] if index < 5 else [2450,3100,3650,3650]
        body.replace(old, table(rows[0], output, widths))
        counts[f'S{index}'] = {'before': len(rows)-1, 'after': len(output), 'added': len(allowed_new)}
    for index, original in untouched.items():
        assert E.tostring(body.findall('w:tbl', NS)[index-1]) == original
    folder = root / 'results/tables/supplementary_method_comparisons'
    comparison = folder / 'comparisons.csv'
    with comparison.open(newline='') as f:
        reader = csv.DictReader(f); columns = reader.fieldnames; old_records = list(reader)
    records, emitted = [], set()
    for row in old_records:
        source = row['source']
        if source not in comparison_sources:
            records.append(row)
        elif source not in emitted:
            method, fresh_rows = comparison_sources[source]
            for r in fresh_rows:
                item = dict(table_method=method, source=source, **r)
                assert set(item).issubset(columns)
                records.append(item)
            emitted.add(source)
    # Earlier comparison exports stored MR1 Pearson rows only (OLS was in the
    # primary tables). Include both methods in the refreshed aggregate export.
    for source in sorted(set(comparison_sources) - emitted):
        method, fresh_rows = comparison_sources[source]
        assert source.startswith('MR1_validation/') and method == 'OLS'
        for r in fresh_rows:
            item = dict(table_method=method, source=source, **r)
            assert set(item).issubset(columns)
            records.append(item)
    metadata = json.loads((folder / 'sources.json').read_text())
    for row in metadata['sources']:
        if row['source'] in comparison_sources:
            row['sha256'] = sources[row['source']]
            row['rows'] = len(comparison_sources[row['source']][1])
    listed = {r['source'] for r in metadata['sources']}
    for source in sorted(set(comparison_sources) - listed):
        metadata['sources'].append({'source': source, 'sha256': sources[source],
                                    'rows': len(comparison_sources[source][1])})
    metadata['diary_log_sensitivity_refresh'] = {'current_table_changes': counts}
    reorg = json.loads((folder / 'reorganization.json').read_text())
    reorg['sources'].update(sources)
    reorg['table_rows'].update({k: v['after'] for k, v in counts.items()})
    reorg['diary_log_sensitivity_refresh'] = {'table_changes': counts, 'other_tables_and_package_parts_preserved': True}
    backup = root / 'workspace_snapshots/author_review' / ('diary_log_supplement_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    for p in [args.document, comparison, folder/'sources.json', folder/'reorganization.json']:
        shutil.copy2(p, backup / p.name)
    with ZipFile(args.document, 'w') as z:
        for info, content in entries:
            z.writestr(info, E.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
                       if info.filename == 'word/document.xml' else content)
    with ZipFile(args.document) as z:
        assert z.testzip() is None
        for info, content in entries:
            if info.filename != 'word/document.xml':
                assert z.read(info.filename) == content
    with comparison.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(records)
    (folder/'sources.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (folder/'reorganization.json').write_text(json.dumps(reorg, indent=2)+'\n')
    (backup/'report.json').write_text(json.dumps(counts, indent=2)+'\n')
    print(json.dumps(counts, indent=2))
    print('Other tables and DOCX package components preserved. Backup:', backup)


if __name__ == '__main__':
    main()
