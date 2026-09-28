#!/usr/bin/env python3
"""Refresh MR1 tables S2/S4/S6/S8 in the organized supplement from aggregates.

Run from project root with --document PATH. Preserves other tables and DOCX
package parts; backs up the document and comparison exports privately.
No participant data access or model fitting.
"""
import argparse
import copy
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def build_tables(root):
    buckets = {2: [], 4: [], 6: [], 8: []}
    sources = {}
    for family in ['02_persistence_affect', '03_persistence_age', '04_fc_affect', '05_fc_persistence']:
        for ols in sorted((root / 'MR1_validation/results/tables' / family).rglob('regressions.csv')):
            if 'full_sample' in ols.parts:
                continue
            corr = ols.with_name('correlations.csv')
            regressions, correlations = read(ols), read(corr)
            lookup = {(r['predictor'], r['outcome']): r for r in correlations}
            assert len(lookup) == len(correlations) == len(regressions)
            assert set(lookup) == {(r['predictor'], r['outcome']) for r in regressions}
            for path in [ols, corr]:
                sources[str(path.relative_to(root))] = sha(path)
            tail = ols.relative_to(root / 'MR1_validation/results/tables' / family).parts[:-1]
            for r in regressions:
                sensitivity = r['outcome'].endswith('_log') or any('sensitivity' in x for x in tail)
                group = ' / '.join([G[family]] + [G.get(x, x.replace('_', ' ')) for x in tail])
                if r['outcome'].endswith('_log'):
                    group += ' / log-NA robustness'
                predictor, outcome = label(r['predictor']), label(r['outcome'])
                values = [r['n'], number(r['beta']), interval(r), pvalue(r['p']), pvalue(r['p_two_tailed'])]
                buckets[4 if sensitivity else 2].append([group, predictor, outcome, *values])
                def cell(row, symbol, coefficient):
                    return (f"N = {row['n']}; {symbol} = {number(row[coefficient])}\n"
                            f"95% CI {interval(row)}\n"
                            f"p (analysis) {pvalue(row['p'])}; p (2-sided) {pvalue(row['p_two_tailed'])}")
                c = lookup[r['predictor'], r['outcome']]
                buckets[8 if sensitivity else 6].append([group, predictor + ' → ' + outcome,
                                                        cell(r, 'β', 'beta'), cell(c, 'r', 'r')])
    return buckets, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document', required=True, type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    buckets, sources = build_tables(root)
    with ZipFile(args.document) as z:
        entries = [(i, z.read(i.filename)) for i in z.infolist()]
    tree = E.fromstring(dict((i.filename, b) for i, b in entries)['word/document.xml'])
    body = tree.find('w:body', NS)
    oldtables = body.findall('w:tbl', NS)
    assert len(oldtables) == 11
    text = ''.join(tree.xpath('//w:t/text()', namespaces=NS))
    assert all(f'Table S{i}. {TITLES[i]}' in text for i in buckets)
    untouched = {i: E.tostring(t) for i, t in enumerate(oldtables, 1) if i not in buckets}
    for i, rows in buckets.items():
        old = oldtables[i-1]
        identities = [[ ''.join(c.xpath('.//w:t/text()', namespaces=NS)) for c in tr.findall('w:tc', NS)[:3 if i <= 4 else 2]] for tr in old.findall('w:tr', NS)[1:]]
        assert identities == [r[:3 if i <= 4 else 2] for r in rows], f'S{i} row identities changed'
        headers = ['Analysis / specification', 'Predictor / interaction', 'Outcome', 'N', 'β', '95% CI', 'p (analysis)', 'p (2-sided)'] if i <= 4 else ['Analysis / specification', 'Association / interaction', 'OLS', 'Pearson correlation']
        widths = [2450,2450,1300,450,800,2000,950,1000] if i <= 4 else [2450,3100,3650,3650]
        body.replace(old, table(headers, rows, widths))
    for i, old in untouched.items():
        assert E.tostring(body.findall('w:tbl', NS)[i-1]) == old
    folder = root / 'results/tables/supplementary_method_comparisons'
    comparison = folder / 'comparisons.csv'
    with comparison.open(newline='') as f:
        reader = csv.DictReader(f); columns = reader.fieldnames; records = list(reader)
    cache = {}
    for row in records:
        if row['source'].startswith('MR1_validation/'):
            path = root / row['source']
            if row['source'] not in cache:
                cache[row['source']] = {(r['predictor'], r['outcome']): r for r in read(path)}
            fresh = cache[row['source']][row['predictor'], row['outcome']]
            assert set(fresh).issubset(columns)
            row.update(fresh)
    metadata = json.loads((folder / 'sources.json').read_text())
    for row in metadata['sources']:
        if row['source'] in sources:
            row['sha256'] = sources[row['source']]
            assert row['rows'] == len(read(root / row['source']))
    reorganization = json.loads((folder / 'reorganization.json').read_text())
    for path, digest in sources.items():
        assert path in reorganization['sources']
        reorganization['sources'][path] = digest
    reorganization['MR1_refresh'] = {'table_rows': {f'S{i}':len(v) for i,v in buckets.items()},
                                    'other_tables_and_package_parts_preserved': True}
    backup = root / 'workspace_snapshots/author_review' / ('mr1_supplement_refresh_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    for path in [args.document, comparison, folder/'sources.json', folder/'reorganization.json']:
        shutil.copy2(path, backup / path.name)
    with ZipFile(args.document, 'w') as z:
        for info, content in entries:
            z.writestr(info, E.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True) if info.filename == 'word/document.xml' else content)
    with ZipFile(args.document) as z:
        assert z.testzip() is None
        for info, content in entries:
            if info.filename != 'word/document.xml':
                assert z.read(info.filename) == content
    with comparison.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(records)
    (folder/'sources.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (folder/'reorganization.json').write_text(json.dumps(reorganization, indent=2)+'\n')
    print('Refreshed MR1 supplement tables:', {f'S{i}':len(v) for i,v in buckets.items()})
    print('Other seven tables and all other DOCX package parts preserved. Backup:', backup)


if __name__ == '__main__':
    main()
