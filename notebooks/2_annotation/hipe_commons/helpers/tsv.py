import os
import io
import urllib.request
from typing import Set, List, Union, NamedTuple, Dict, Optional
import pandas as pd
from dataclasses import dataclass
import dataclasses

# ======================================================================================================================
#                                                       VARIABLES
# ======================================================================================================================
# V1 used for HIPE-2020 and HIPE-2022
COL_LABELS = [
    "TOKEN",
    "NE-COARSE-LIT",
    "NE-COARSE-METO",
    "NE-FINE-LIT",
    "NE-FINE-METO",
    "NE-FINE-COMP",
    "NE-NESTED",
    "NEL-LIT",
    "NEL-METO",
    "MISC",
]

# V2 used for hipe-newsbench
COL_LABELS_V2 = [
    "TOKEN",
    "NE-COARSE-LIT",
    "NE-COARSE-METO",
    "NE-FINE-LIT",
    "NE-FINE-METO",
    "NE-FINE-COMP",
    "NE-NESTED",
    "NEL-LIT",
    "NEL-METO",
    "RENDER",
    "SEG",
    "OCR-INFO",
    "MISC"
]
NE_ANNOTATION_TYPES = [
    "coarse_lit",
    "coarse_meto",
    "fine_lit",
    "fine_meto",
    "fine_comp",
    "nested"
]

ENTITY_TYPES = [
    "coarse_lit",
    "coarse_meto",
    "fine_lit",
    "fine_meto",
    "fine_comp",
    "nested"
]

PARTIAL_FLAG = "Partial"
NO_SPACE_AFTER_FLAG = "NoSpaceAfter"
END_OF_LINE_FLAG = "EndOfLine"
END_OF_SENTENCE_FLAG = "EndOfSentence"
NIL_FLAG = "NIL"
BLANK_LINE_FLAG = "BLANK_LINE"

IOB_FIRST_LINE = "# global.columns = TOKEN NE-COARSE-LIT NE-COARSE-METO NE-FINE-LIT NE-FINE-METO NE-FINE-COMP " \
                 "NE-NESTED NEL-LIT NEL-METO RENDER SEG OCR-INFO MISC"


# ======================================================================================================================
#                                                     CUSTOM TYPES AND OBJECTS
# ======================================================================================================================

class TSVComment(NamedTuple):
    """Namedtuple representing a commented tsv-line.

    ..note:: In the tsv, commented lines comply the following format: `# some_key = some_value`. """
    n: int
    field: str
    value: str

    def __repr__(self):
        return f"# {self.field} = {self.value}"


@dataclass(frozen=False)
class TSVAnnotation_v2():
    """Data class representing a HIPE tsv-line containing annotations, v2 (13 columns)."""
    n: int
    token: str
    ne_coarse_lit: str
    ne_coarse_meto: str
    ne_fine_lit: str
    ne_fine_meto: str
    ne_fine_comp: str
    ne_nested: str
    nel_lit: str
    nel_meto: str
    render: str
    seg: str
    ocr_info: str
    misc: str

    def __repr__(self):
        return (
            f"{self.token}\t{self.ne_coarse_lit}\t"
            f"{self.ne_coarse_meto}\t{self.ne_fine_lit}\t"
            f"{self.ne_fine_meto}\t{self.ne_fine_comp}\t{self.ne_nested}\t"
            f"{self.nel_lit}\t{self.nel_meto}\t{self.render}\t"
            f"{self.seg}\t{self.ocr_info}\t{self.misc}"
        )


class TSVAnnotation(NamedTuple):
    """Namedtuple representing an ordinary tsv-line, containing annotations."""
    n: int
    token: str
    ne_coarse_lit: str
    ne_coarse_meto: str
    ne_fine_lit: str
    ne_fine_meto: str
    ne_fine_comp: str
    ne_nested: str
    nel_lit: str
    nel_meto: str
    misc: str

    def __repr__(self):
        return (
            f"{self.token}\t{self.ne_coarse_lit}\t"
            f"{self.ne_coarse_meto}\t{self.ne_fine_lit}\t"
            f"{self.ne_fine_meto}\t{self.ne_fine_comp}\t{self.ne_nested}\t"
            f"{self.nel_lit}\t{self.nel_meto}\t{self.misc}"
        )

    def convert2_tsv_annotation_v2(self) -> TSVAnnotation_v2:
        return TSVAnnotation_v2(
            n=self.n,
            token=self.token,
            ne_coarse_lit=self.ne_coarse_lit,
            ne_coarse_meto=self.ne_coarse_meto,
            ne_fine_lit=self.ne_fine_lit,
            ne_fine_meto=self.ne_fine_meto,
            ne_fine_comp=self.ne_fine_comp,
            ne_nested=self.ne_nested,
            nel_lit=self.nel_lit,
            nel_meto=self.nel_meto,
            render=None,
            seg=None,
            ocr_info=None,
            misc=self.misc
        )


TSVLine = Union[TSVAnnotation, TSVComment]

TSVLine_v2 = Union[TSVAnnotation_v2, TSVComment]


class HipeEntity(object):
    def __init__(self, text, ne_type, ne_tag, wikidata_id, line_numbers):
        self.text = text
        self.type = ne_type
        self.tag = ne_tag
        self.wikidata_id = wikidata_id
        self.line_numbers = line_numbers

    def __repr__(self):
        return f"[{self.tag}] {self.text} ({self.wikidata_id})"


class HipeDocument(object):
    def __init__(self, path, tsv_lines):
        self.path = path
        self._tsv_lines = tsv_lines
        self.entities = {}

        self.n_tokens = sum([
            1
            for line in self._tsv_lines
            if not isinstance(line, TSVComment)
        ])

        self.metadata = {
            line.field: line.value
            for line in self._tsv_lines
            if isinstance(line, TSVComment)
        }

        for e_type in NE_ANNOTATION_TYPES:
            entities = self._lines2entities(e_type)
            if entities:
                self.entities[e_type] = entities

    def _lines2entities(self, entity_type: str, hipe_format_version="v2") -> List[HipeEntity]:
        """Parse a token-based entity representation into a `HipeEntity` object.

        :param entity_type: The entity type to consider (any of the values in `ENTITY_TYPES`)
        :type entity_type: str
        :return: A list of `HipeEntity` objects.
        :rtype: List[HipeEntity]
        """
        entities = []
        line_groups = []
        current_entity_lines = []

        # 1) identify group of lines, each group corresponding to one entity
        for line in self._tsv_lines:
            if isinstance(line, TSVComment):
                continue

            # depending on the entity type, the NE tag is found in a diff TSV column
            if entity_type == "coarse_lit":
                ne_tag = line.ne_coarse_lit
            elif entity_type == "coarse_meto":
                ne_tag = line.ne_coarse_meto
            elif entity_type == "fine_lit":
                ne_tag = line.ne_fine_lit
            elif entity_type == "fine_meto":
                ne_tag = line.ne_fine_meto
            elif entity_type == "fine_comp":
                ne_tag = line.ne_fine_comp
            elif entity_type == "nested":
                ne_tag = line.ne_nested
            else:
                ne_tag = None

            if ne_tag:
                # beginning of a new entity
                if ne_tag.startswith('B-'):

                    # case of two consecutive B-* tags
                    if current_entity_lines:
                        line_groups.append(current_entity_lines)
                        current_entity_lines = []

                    current_entity_lines.append(line)

                # continuation of an entity
                elif ne_tag.startswith('I-'):
                    current_entity_lines.append(line)

                # O tag
                else:
                    if current_entity_lines:
                        line_groups.append(current_entity_lines)
                        current_entity_lines = []

        if current_entity_lines:
            line_groups.append(current_entity_lines)

        # 2) parse line groups into HipeEntity class instances                
        for group in line_groups:
            entities.append(lines2entity(group, entity_type, hipe_format_version=hipe_format_version))

        return entities


# ======================================================================================================================
#                                                       HELPER FUNCTIONS
# ======================================================================================================================

def lines2entity(tsv_lines: List[TSVLine], entity_type: str, hipe_format_version="v2") -> HipeEntity:
    """Converts a group of TSV lines into a `HipeEntity` object.

    :param tsv_lines: List of TSV lines corresponding to a token-based representation of an entity.
    :type tsv_lines: List[TSVLine]
    :param entity_type: The type of entities to select.
    :type entity_type: str
    :return: Returns a list of `HipeEntity` objects.
    :rtype: HipeEntity
    """

    # reconstruct the entity surface form from its tokens;
    # white space insertion is controlled via the `NoSpaceAfterFlag`
    # contained in the `Misc` column of the TSV file
    surface_form = ""
    for line in tsv_lines:
        surface_form += line.token
        if hipe_format_version == "v1" and not "NoSpaceAfter" in line.misc:
            surface_form += " "
        if hipe_format_version == "v2" and not "NoSpaceAfter" in line.render:
            surface_form += " "

    # keep track of the TSV line numbers over which the entity spans.
    # this information can be useful mostly for diagnostics and debugging.
    line_numbers = [
        line.n + 1
        for line in tsv_lines
    ]

    # the NE tag and NE link information is contained in different columns
    # depending on the selected entity_type 
    if entity_type == "coarse_lit":
        ne_tag = [
            line.ne_coarse_lit.split('-')[1]
            for line in tsv_lines
        ]

        ne_link = [
            line.nel_lit
            for line in tsv_lines
        ]
    elif entity_type == "coarse_meto":
        ne_tag = [
            line.ne_coarse_meto.split('-')[1]
            for line in tsv_lines
        ]

        ne_link = [
            line.nel_meto
            for line in tsv_lines
        ]
    elif entity_type == "fine_lit":
        ne_tag = [
            line.ne_fine_lit.split('-')[1]
            for line in tsv_lines
        ]

        ne_link = [
            line.nel_lit
            for line in tsv_lines
        ]
    elif entity_type == "fine_meto":
        ne_tag = [
            line.ne_fine_meto.split('-')[1]
            for line in tsv_lines
        ]

        ne_link = [
            line.nel_meto
            for line in tsv_lines
        ]
    elif entity_type == "fine_comp":
        ne_tag = [
            line.ne_fine_comp.split('-')[1]
            for line in tsv_lines
        ]

        # entity components don't come with EL info
        ne_link = None
    elif entity_type == "nested":
        ne_tag = [
            line.ne_nested.split('-')[1]
            for line in tsv_lines
        ]

        # entity components don't come with EL info
        ne_link = None

    # if the NE link is absent, replace it with `None`
    if ne_link:
        ne_link = ne_link[0] if ne_link[0] != "_" else None

    if ne_tag:
        # now we can build and return the entity object
        return HipeEntity(surface_form, entity_type, ne_tag[0], ne_link, line_numbers)


def find_datasets_files(base_dir: str) -> List[str]:
    """Finds recursively TSV file in a folder.

    ..note::
        The expected folder structure is one sub-folder per language.

    :param str base_dir: Description of parameter `base_dir`.
    :return: A list of TSV file paths.
    :rtype: List[str]

    """
    datasets_files = []
    for lang in os.listdir(base_dir):
        for file in os.listdir(os.path.join(base_dir, lang)):
            if ".tsv" in file and "orig" not in file:
                datasets_files.append(os.path.join(base_dir, lang, file))
    return datasets_files


def find_missing_iiif_links(input_tsv_file: str) -> Set[str]:
    """Finds which content items don't have IIIF links.

    :param str input_tsv_file: Input TSV file in HIPE format.
    :return: A set of content item IDs which don't have IIIF links.
    :rtype: Set[str]

    """

    missing_links = set()
    with open(input_tsv_file, "r") as f:

        doc_sections = f.read().split("\n\n")

        for doc in doc_sections:
            doc_id = [
                line.strip().split("=")[-1].strip()
                for line in doc.split("\n")
                if "document_id" in line
            ][0]

            iiif_link = [
                line.strip().split("=")[-1].strip()
                for line in doc.split("\n")
                if "segment_iiif_link" in line
            ][0]

            if iiif_link == "_":
                missing_links.add(doc_id)

    return missing_links


def is_tsv_complete(dataset_path: str, expected_doc_ids: List[str]) -> bool: #TODO: move to test?
    """Verifies whether a given TSV file (dataset) contains all expected documents  

    :param dataset_path: Path to the TSV file.
    :type dataset_path: str
    :param expected_doc_ids: List of documents identifiers that are expected to be contained in the dataset.
    :type expected_doc_ids: List[str]
    :return: Returns `True` if the file is complete, otherwise `False`.
    :rtype: bool
    """
    with open(dataset_path, "r") as f:
        tsv_doc_ids = [
            line.strip().split("=")[-1].strip()
            for line in f.readlines()
            if line.startswith("#") and "document_id" in line
        ]

    difference = set(expected_doc_ids).difference(set(tsv_doc_ids))

    try:
        assert difference == set()
        return True
    except AssertionError:
        print(f"Following documents are missing from {dataset_path}: {difference}")
        return False


def get_tsv_data(path: Optional[str] = None, url: Optional[str] = None) -> str:
    """Fetches tsv data from a path or an url."""

    assert path or url, """`path` or `url` must be provided"""

    if url:
        response = urllib.request.urlopen(url)
        return response.read().decode('utf-8')

    elif path:
        with open(path) as f:
            return f.read()


def parse_tsv(mask_nerc: bool = False, mask_nel: bool = False, hipe_format_version="v2", **kwargs) -> List[
    HipeDocument]:

    if "file_url" in kwargs and kwargs['file_url']:
        file_path = kwargs['file_url']
        tsv_data = get_tsv_data(url=file_path)

    elif "file_path" in kwargs and kwargs['file_path']:
        file_path = kwargs['file_path']
        tsv_data = get_tsv_data(path=file_path)

    else:
        raise

    if hipe_format_version == "v1":
        documents = [
            HipeDocument(
                path=file_path,
                tsv_lines=[
                    parse_tsv_line(line, line_number, mask_nerc, mask_nel)
                    for line_number, line in enumerate(document.split("\n"))
                    if line.split('\t') != COL_LABELS and line != ""
                ]
            )
            for document in tsv_data.split("\n\n")
        ]
        return documents
    elif hipe_format_version == "v2":
        documents = [
            HipeDocument(
                path=file_path,
                tsv_lines=[
                    parse_tsv_line(line, line_number, mask_nerc, mask_nel, hipe_format_version=hipe_format_version)
                    for line_number, line in enumerate(document.split("\n"))
                    if line.split('\t') != COL_LABELS_V2 and line != ""
                ]
            )
            for document in tsv_data.split("\n\n")
        ]
        return documents


def is_comment(line: str) -> bool:
    return line.startswith("#") and "=" in line


def parse_comment(comment_line: str, line_number: int) -> TSVComment:
    """Parses a line of TSV file into a `TSVComment` object.

    :param str comment_line: A commented line, in the format `'# key = value"`.
    :param int line_number: Description of parameter `line_number`.
    :return: Description of returned object.
    :rtype: TSVComment

    """
    try:
        key, value = [el.strip() for el in comment_line.replace("#", "").split("=")]
    except:
        import ipdb

        ipdb.set_trace()

    return TSVComment(n=line_number, field=key, value=value)


def parse_annotation(line: str, line_number: int) -> TSVAnnotation:  ##TODO: add suffix _v1 if solution is retained.
    """Parses a TSV line into a `TSVAnnotation` object."""
    values = line.split("\t")
    return TSVAnnotation(
        n=line_number,
        token=values[0],
        ne_coarse_lit=values[1] if len(values) >= 2 else None,
        ne_coarse_meto=values[2] if len(values) >= 3 else None,
        ne_fine_lit=values[3] if len(values) >= 4 else None,
        ne_fine_meto=values[4] if len(values) >= 5 else None,
        ne_fine_comp=values[5] if len(values) >= 6 else None,
        ne_nested=values[6] if len(values) >= 7 else None,
        nel_lit=values[7] if len(values) >= 8 else None,
        nel_meto=values[8] if len(values) >= 9 else None,
        misc=values[9] if len(values) >= 10 else None,
    )


def parse_annotation_v2(line: str, line_number: int) -> TSVAnnotation_v2:
    """Parses a TSV line into a `TSVAnnotation_v2` object."""
    values = line.split("\t")
    return TSVAnnotation_v2(
        n=line_number,
        token=values[0],
        ne_coarse_lit=values[1] if len(values) >= 2 else None,
        ne_coarse_meto=values[2] if len(values) >= 3 else None,
        ne_fine_lit=values[3] if len(values) >= 4 else None,
        ne_fine_meto=values[4] if len(values) >= 5 else None,
        ne_fine_comp=values[5] if len(values) >= 6 else None,
        ne_nested=values[6] if len(values) >= 7 else None,
        nel_lit=values[7] if len(values) >= 8 else None,
        nel_meto=values[8] if len(values) >= 9 else None,
        render=values[9] if len(values) >= 10 else None,
        seg=values[10] if len(values) >= 11 else None,
        ocr_info=values[11] if len(values) >= 12 else None,
        misc=values[12] if len(values) >= 13 else None,
    )


def parse_tsv_line(line: str, line_number: int, mask_nerc: bool = False, mask_nel: bool = False,
                   hipe_format_version="v2") -> TSVLine:
    """General parser for tsv lines, leveraging `parse_comment` and `parse_annotation` for annotations
    and commented lines respectively."""

    if is_comment(line):
        return parse_comment(line, line_number)
    else:
        ann = parse_annotation(line, line_number) if hipe_format_version == "v1" else parse_annotation_v2(line,
                                                                                                          line_number)
        if mask_nerc and mask_nel:
            return mask_all_groundtruth(ann)
        elif mask_nel and not mask_nerc:
            return mask_nel_groundtruth(ann)
        else:
            return ann


def mask_all_groundtruth(annotation: TSVAnnotation, mask: str = "_") -> TSVAnnotation:
    """Hides annotations from an input annotation.

    This is used when preparing the test data for the shared task competition.
    Only neutral fields are kept (`token` and `misc`), while all the rest is
    replaced with the `mask` character
    """
    masked_annotation = TSVAnnotation(
        n=annotation.n,
        token=annotation.token,
        ne_coarse_lit=mask,
        ne_coarse_meto=mask,
        ne_fine_lit=mask,
        ne_fine_meto=mask,
        ne_fine_comp=mask,
        ne_nested=mask,
        nel_lit=mask,
        nel_meto=mask,
        misc=annotation.misc,
    )
    return masked_annotation


def mask_nel_groundtruth(annotation: TSVAnnotation, mask: str = "_") -> TSVAnnotation:
    """Hides only NEL annotations from an input annotation.

    This is used when preparing the test data for bundle 5 of the shared task competition.
    Only NERC-related + neutral fields are kept (`token` and `misc`), while all the rest is
    replaced with the `mask` character.
    """
    masked_annotation = TSVAnnotation(
        n=annotation.n,
        token=annotation.token,
        ne_coarse_lit=annotation.ne_coarse_lit,
        ne_coarse_meto=annotation.ne_coarse_meto,
        ne_fine_lit=annotation.ne_fine_lit,
        ne_fine_meto=annotation.ne_fine_meto,
        ne_fine_comp=annotation.ne_fine_comp,
        ne_nested=mask,
        nel_lit=mask,
        nel_meto=mask,
        misc=annotation.misc,
    )
    return masked_annotation


def write_tsv(documents: List[List[TSVLine]], output_path: str, hipe_format_version: str = "v1") -> None:
    """
    Write TSVlines to .tsv file, with appropriate hipe headers.
    :param  List[List[TSVLine]] documents: HIPE formatted document lines
    :param str output_path: the file where the data will be written
    :param hipe_format_v: which version of hipe format to serialise to. "v1" (default) or "v2"
    :rtype: object
    """
    headers = COL_LABELS if hipe_format_version == "v1" else COL_LABELS_V2
    raw_csv = "\n\n".join(
        ("\n".join((str(line) for line in document)) for document in documents)
    )
    headers_line = "\t".join(headers)
    preamble = headers_line if hipe_format_version == "v1" else f"{IOB_FIRST_LINE}"
    csv_content = f"{preamble}\n{raw_csv}\n"

    with io.open(output_path, "w", encoding="utf-8") as f:
        f.write(csv_content)


def tsv_to_dict(path: Optional[str] = None, url: Optional[str] = None, keep_comments: bool = False,
                hipe_format_version: str = "v1") -> Dict[
    str, List[str]]:
    """The simplest and most straightforward way to get tsv-data into a python structure. This function is used as the
    basis for other converters
    :param path:
    :param keep_comments:
    :param url:
    :param hipe_format_version: """

    data = get_tsv_data(path, url).split('\n')
    header = data[0].split('\t')
    
    if not keep_comments:
        dict_ = {k: [] for k in ['n'] + header}

        for i, line in enumerate(data[1:]):  # As data[0] is the header
            if line and not line.startswith('#'):
                line = line.split('\t')
                dict_['n'].append(i + 1)  # as we are starting with data[1:]
                for j, k in enumerate(header):
                    dict_[k].append(line[j])
            else:
                continue

    else:
        comments, dict_ = {}, None

        for i, line in enumerate(data[1:]):  # As data[0] is the header

            if line:
                parsed_line = parse_tsv_line(line, i + 1, hipe_format_version=hipe_format_version)

                if isinstance(parsed_line, TSVComment):  # If comment, stock comment's field and value
                    comments[parsed_line.field] = parsed_line.value

                else:  # else, appends annotations and comments values to `dict_`
                    dict_ = {k: [] for k in ['n'] + header + list(comments.keys())} if not dict_ else dict_
                    for k in dict_.keys():
                        formated_k = k.lower().replace('-', '_')
                        if hipe_format_version == "v1":
                            dict_[k].append(
                                getattr(parsed_line, formated_k)
                                if formated_k in parsed_line._fields else comments[k])
                        elif hipe_format_version == "v2":
                            dict_[k].append(
                                getattr(parsed_line, formated_k)
                                if formated_k in [f.name for f in dataclasses.fields(parsed_line)] else comments[k])

    return dict_


def tsv_to_dataframe(path: Optional[str] = None, url: Optional[str] = None,
                     keep_comments: bool = False,
                     hipe_format_version: str = "v2") -> pd.DataFrame:
    """Converts a HIPE-compliant tsv to a `pd.DataFrame`, keeping comment fields as columns.

    Each row corresponds to an annotation row of the tsv (i.e. a token). Commented fields (e.g. `'document_id`) are
    added to the dataframe as columns.

    ..note:: In the output-dataframe, column 'n' corresponds to the line number in the original tsv file, not in the
    dataframe.

    :param hipe_format_version:
    :param keep_comments:
    :param str path: Path to a HIPE-compliant tsv file
    :param str url: url to a HIPE-compliant tsv file
    """
    return pd.DataFrame(tsv_to_dict(path=path,
                                    url=url,
                                    keep_comments=keep_comments,
                                    hipe_format_version=hipe_format_version))


def tsv_to_segmented_lists(labels: List[str],
                           path: Optional[str] = None,
                           url: Optional[str] = None,
                           segmentation_flag: Union[str, int] = 'EndOf',
                           hipe_format_version: str = "v1" ) -> Dict[str, List[List[str]]]:
    """Converts a HIPE-compliant tsv to lists of examples containing lists of tokens,
    with their aligned labels and doc_ids.

    Generally used to make data amenable to a HuggingFace Tokenizer.

    The output is a dict containing tokens, labels and document_ids, all in the format:
        ```
        {'texts': [[sentence1_word1, sentence1_word2...],[sentence2_word1,...]...],
         'doc_ids': [[doc_id, doc_id...],[doc_id,...]...],
         'your_labels': [[sentence1_label1, sentence1_label2...],[sentence2_label1,...]...]}
        ```
    :param List[str] labels: The desired column labels (e.g. `['NE-COARSE-LIT','NEL-LIT'])
    :param str path: Path to a HIPE-compliant tsv file
    :param str url: url to a HIPE-compliant tsv file
    :param str segmentation_flag: The flag to look up for in the MISC column and to use as a separator.
        Should be `'EndOfLine'` or `'EndOfSentence'` or a any flag listed in the `tsv` module.

    :returns: Dict, see above
    """
    df = tsv_to_dataframe(path=path, url=url, keep_comments=True, hipe_format_version=hipe_format_version)
    d = {k: [] for k in ['texts', 'doc_ids'] + labels}

    doc_id_col = [col for col in df.columns if 'document_id' in col][0]

    example_tokens = []
    example_labels = {k: [] for k in labels}
    example_doc_ids = []

    for i in range(len(df)):
        example_tokens.append(df['TOKEN'][i])
        example_doc_ids.append(df[doc_id_col][i])
        for label in labels:
            example_labels[label].append(df[label][i])

        if (hipe_format_version == "v1" and segmentation_flag in df['MISC'][i]) \
            or (hipe_format_version == "v2" and segmentation_flag in {df['RENDER'][i], df['SEG'][i]}):
            d['texts'].append(example_tokens)
            d['doc_ids'].append(example_doc_ids)
            for label in labels:
                d[label].append(example_labels[label])

            example_tokens = []
            example_labels = {k: [] for k in labels}
            example_doc_ids = []

    if example_tokens:
        d['texts'].append(example_tokens)
        d['doc_ids'].append(example_doc_ids)
        for label in labels:
            d[label].append(example_labels[label])

    return d


def tsv_to_huggingface_dataset(
        labels: List[str],
        path: Optional[str] = None,
        url: Optional[str] = None,
        segmentation_flag: str = 'EndOf'
):
    """Converts a HIPE-compliant tsv to a custom `torch.utils.data.Dataset`, making it directly amenable to
       HuggingFace transformers.

       ..note.: Unlike `tsv_to_torch_dataset`, this function does NOT tokenize the texts, and simply converts it
       to the datasets pyarrow datastructure."""

    from datasets import Dataset
    data = tsv_to_segmented_lists(labels=labels, path=path, url=url, segmentation_flag=segmentation_flag)
    return Dataset.from_dict(data)


def tsv_to_torch_dataset(
        label_type: str,
        labels_to_ids: Dict[str, int],
        tokenizer: Union['transformers.PreTrainedTokenizer', 'transformers.PreTrainedTokenizerFast'],
        path: Optional[str] = None,
        url: Optional[str] = None,
        segmentation_flag: Union[str, int] = 'EndOf',
        label_all_tokens: bool = False,
        **tokenizer_kwargs):
    """Converts a HIPE-compliant tsv to a custom `torch.utils.data.Dataset`, making it directly amenable to
    torch and HuggingFace transformers.

    What this does is:
        1) Segmenting the tsv into annotated lists of examples using `tsv_to_lists`
        2) Aokenizing the created lists, using `tokenizer()`
        3) Aligning labels, using `align_and_pad_labels`.
    Please customize these calls using additional `tokenizer_kwargs`, such as `padding` (see docs).

    ..note.: If you use the tokenizer to truncate sentences, the overflowing will be lost. Do handle this, please use
    `tsv_to_huggingface_dataset` instead.

    This will return a `torch.utils.data.Dataset` with tokens and their corresponding labels. Note that this will only
    work with a single label column.

    :param str label_type: The desired label type in the tsv, e.g. `'NE-COARSE-LIT'`
    :param labels_to_ids: A Dict[str,int] mapping labels to their respective ids
    :param tokenizer: A transformer tokenizer
    :param str path: The path to the tsv file
    :param str url: The url of the tsv file (must be provided if path is not
    :param segmentation_flag: See `tsv_to_lists`.
    :param label_all_tokens: See `align_and_pad_tags`.

    :returns: A `torch.utils.data.Dataset` with tokens and their corresponding labels
    """

    import torch

    def align_and_pad_labels(texts: "BatchEncoding",
                             labels: List[List[str]],
                             labels_to_ids: Dict[str, int],
                             label_all_tokens: bool = False,
                             null_label: object = -100) -> List[List[int]]:
        """Converts labels to labelids, aligns and pads labels to tokenized texts, using token indices.

        If `label_all_tokens` is `True`, labels attributed to a word in data are broadcast to all its corresponding
        subtokens ; else, only the first subtoken is marked with a label, the rest being
        marked with the `null_label`. Padded labels (i.e. labels between initial sequence length and `max_sequence_length`)
        are marked with `null_label`.

        :param texts: a BatchEncoding-object. Attribute `input_ids` contains tokenized text, in the format :
        List[List[str]], e.g. `[["example", "one"],...]`.
        :param labels: should come in the format outputed by `read_line_json` or `tokenize_and_pad_tokens`:
        List[List[str]], e.g. `[["O", "B-AAWORK"],...]`.
        :param labels_to_ids: A Dict[str,int] mapping labels to their respective ids.

        :returns: The List[List[int]] of labels.
        """

        # Changes labels to id, keeping the list[list] architecture
        original_labelids = [[labels_to_ids[label] for label in instance_labels] for instance_labels in labels]

        all_labels = []
        for i in range(len(texts["input_ids"])):
            token_indices = texts.word_ids(batch_index=i)
            previous_token_index = None
            instance_labels = []

            for token_index in token_indices:
                if token_index is None:
                    instance_labels.append(null_label)

                elif token_index != previous_token_index:
                    instance_labels.append(original_labelids[i][token_index])

                else:
                    b_to_i_label = labels_to_ids['I' + labels[i][token_index][1:]] if labels[i][
                                                                                          token_index] != 'O' else 'O'
                    instance_labels.append(b_to_i_label if label_all_tokens else null_label)
                previous_token_index = token_index

            all_labels.append(instance_labels)

        return all_labels

    class HipeTorchDataset(torch.utils.data.Dataset):
        """A custom class to make HIPE-data amenable to pytorch and transformers"""

        def __init__(self, encodings: "BatchEncoding", labels: List[List[int]]):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    data = tsv_to_segmented_lists(labels=[label_type], path=path, url=url, segmentation_flag=segmentation_flag)

    tokenized_texts = tokenizer(data['texts'], is_split_into_words=True, **tokenizer_kwargs)

    aligned_labels = align_and_pad_labels(tokenized_texts, labels=data[label_type],
                                          labels_to_ids=labels_to_ids,
                                          label_all_tokens=label_all_tokens)

    return HipeTorchDataset(tokenized_texts, aligned_labels)


def get_unique_labels(path: Optional[str] = None, url: Optional[str] = None, label_type: Optional[str] = None,
                      label_list: Optional[List[str]] = None) -> List[str]:
    """Returns a list of unique labels contained in a HIPE-tsv file or directly in a label list"""

    if not label_list:
        label_list = tsv_to_dataframe(path, url)[label_type].tolist()

    labels = ['O']

    for label in sorted(set([label_[2:] for label_ in label_list if label_ != 'O'])):
        labels.append('B-' + label)
        labels.append('I-' + label)

    return labels
