import pandas as pd
from typing import Dict
from collections import Counter
from typing import List, Dict
from .helpers.tsv import HipeDocument, parse_tsv, ENTITY_TYPES

def count_entities(docs: List[HipeDocument]) -> Dict:

    counts = {}
    
    for e_type in ENTITY_TYPES:
        for doc in docs:
            if e_type in doc.entities:
                if e_type not in counts:
                    counts[e_type] = 0
                counts[e_type] += len(doc.entities[e_type])

    return counts


def compute_entities_stats(docs: List[HipeDocument]) -> Dict[str, pd.DataFrame]:
    counts = {}
    
    for e_type in ENTITY_TYPES:
        for doc in docs:
            if e_type in doc.entities:
                if e_type not in counts:
                    counts[e_type] = Counter()
                
                # counts per entity tag 
                doc_entity_counts = Counter(
                    [
                        entity.tag
                        for entity in doc.entities[e_type]
                    ]
                )

                # update total counts
                counts[e_type] += doc_entity_counts

    for e_type in counts:
        df = pd.DataFrame.from_dict(
            dict(counts[e_type]), orient='index'
        ).rename(columns={0: 'count'}).sort_index()
        counts[e_type] = df

    return counts


def describe_dataset(**kwargs) -> str:
    """Create a plain text summary of a dataset in HIPE TSV-format.

    `kwargs` must contain one of these three possible values:
    - `file_path` (`str`)
    - `file_url` (`str`)
    - `documents` (`List[HipeDocument]`)

    :return: Dataset description as plain text.
    :rtype: str
    """    
    if "file_path" in kwargs:
        path = kwargs['file_path']
        docs = parse_tsv(file_path=path)
    elif "file_url" in kwargs:
        path = kwargs['file_url']
        docs = parse_tsv(file_url=path)
    elif "documents" in kwargs:
        docs = kwargs['documents']
        path = docs[0].path
    else:
        raise

    total_n_tokens = sum([
        doc.n_tokens
        for doc in docs
    ])

    entity_stats = compute_entities_stats(docs)

    entity_breakdown = ""
    for e_type in entity_stats.keys():
        entity_breakdown += f'{e_type}\n'
        entity_breakdown += f"{entity_stats[e_type].to_markdown(tablefmt='grid')}\n"

    desc = ""
    desc += f'\nPath of the TSV file: {path} \n'
    desc += f'Number of documents: {len(docs)} \n'
    desc += f'Number of entities: {count_entities(docs)} \n'
    desc += f'Number of tokens: {total_n_tokens} \n'
    desc += f'Entity breakdown by type: {entity_breakdown}'
    return desc