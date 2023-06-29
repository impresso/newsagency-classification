"""
Add automatic annotation of sentence boundaries using pySBD

Usage:
    lib/sent_splitter.py --schema=<fpath> --input-dir=<dir> --output-dir=<dir> --lang=<lang> --log=<fpath> [--check_segments]

Options:
    -h --help               Show this screen.
    -s --schema=<fpath>     File path to schema file.
    -i --input-dir=<dir>    Folder with the original  files (.xmi format).
    -o --output-dir=<dir>   Folder where the new files ares saved (.xmi format).
    -l --lang=<lang>        Language code to apply specific rules for segmenting.
    --check_segments        Print segments yielded by pySBD to console [default: False].
    --log=<fpath>           Name of log file.
"""


import sys
import logging
from pathlib import Path

from tqdm import tqdm
from cassis import Cas, load_cas_from_xmi, load_typesystem

from docopt import docopt

import pysbd


sys.path.append("../")
sys.path.append("../../")


class Splitter:
    """Segment original text and add automatic annotation of sentence boundaries

    :param str xmi: path to existing xmi annotation file.
    :param str xml: path to xml schema file.
    :param str lang: Abbreviation of language used for segmenting (e.g., de).
    :param bool check_segments: Print segmented sentences to console.
    :return: None
    :rtype: None

    """

    def __init__(self, xml: str, lang: str, check_segments: bool = False):

        self.xml = xml

        with open(self.xml, "rb") as f:
            self.typesystem = load_typesystem(f)

        self.segmentType = "webanno.custom.PySBDSegment"

        self.splitter = pysbd.Segmenter(language=lang, clean=False)
        logging.info(f"Load pySBD for the following language: {lang.upper()}")

        self.check_segments = check_segments

    def sent_split(self, f_xmi: str) -> Cas:
        """Add annotation for sentence segments automatically produced with pySBD.

        :param str f_xmi: XMI file containing all annotations.
        :return: Original cassis object with added annotations for sentence segments.
        :rtype: Cas

        """

        with open(f_xmi, "rb") as f:
            cas = load_cas_from_xmi(f, typesystem=self.typesystem)

        Segment = self.typesystem.get_type(self.segmentType)

        text = cas.sofa_string

        # don't consider line breaks as they are not reliable due to OCR issues
        segments = self.splitter.segment(text.replace("\n", " "))

        start = 0
        anno_segments = []

        for i, seg in enumerate(segments):

            # sentence end after removing trailing whitespace
            end_stripped = start + len(seg.rstrip())

            # sentence start after removing initial whitespace
            start_stripped = start + len(seg) - len(seg.lstrip())

            anno_segments.append(Segment(begin=start_stripped, end=end_stripped))

            try:
                assert seg.strip() == text[start_stripped:end_stripped]
            except AssertionError:
                print(f"Error in document {f_xmi}")
                print("|||" + seg + "|||pysbd")
                print("|||" + text[start_stripped:end_stripped] + "|||cassis")
                print("|||" + text[start : start + len(seg)] + "|||segment in orig")
                # import ipdb; ipdb.set_trace()

            # current position in original text
            start = start + len(seg)

            # workarounds for pySBD v0.3.3 as it is destructive in case non-standard punctuation
            # e.g. multiple periods (OCR errors)
            # https://github.com/nipunsadvilkar/pySBD/issues/83
            # code block can be removed after this is fixed

            try:
                if seg in (".", "...", "!!!") and segments[i - 1][-1] == ".":
                    # wrongly removed space by pysbd
                    start += 1
                elif seg == '...' and segments[i - 1].endswith(" "):
                    # wrongly removed space by pysbd
                    start += 1

                elif seg == ",." and text[end_stripped] == ' ':
                    # wrongly removed space by pysbd
                    start += 1
                elif segments[i - 1][-1] == "!" and seg in "!"  and text[end_stripped] == ' ':
                    # wrongly removed space by pysbd
                    start += 1
                elif seg == ". " and text[end_stripped - 1] == "." and text[end_stripped] != " ":
                    # hallucinated space by pysbd
                    start -= 1
                elif seg.endswith(". ") and segments[i + 1].startswith(" . . ."):
                    # hallucinated space by pysbd
                    start -= 1
            except IndexError:
                # lookahead at the end of a document
                pass

        cas.add_all(anno_segments)

        if self.check_segments:
            self._check_auto_segment(cas)

        return cas

    def _check_auto_segment(self, cas):
        print("#" * 20)
        print("### ORIGINAL TEXT")
        # print(cas.sofa_string)
        print("#" * 20)

        print("#" * 20)
        print("### pySBD SEGMENTED TEXT")
        print("#" * 20)

        for i, seg in enumerate(cas.select(self.segmentType)):
            print(i, "\t", f"|||{seg.get_covered_text()}|||")


def index_inception_files(dir_data, suffix=".xmi") -> list:
    """Return all .xmi files in the provided directory

    :param type dir_data: Path to top-level dir.
    :param type suffix: Only consider files of this type.
    :return: List of found files.
    :rtype: list

    """

    return sorted([path for path in Path(dir_data).rglob("*" + suffix)])


def sentence_segmenting_batch(
    dir_in: str, dir_out: str, f_schema: str, language: str, check_segments: bool
) -> None:
    """Perform automatic sentence segmentation for all xmi files in the provided folder.

    :param str dir_in: Folder containing the .xmi-files.
    :param str dir_out: Output folder for the .xmi-files containing segment annotation.
    :param str f_schema: Path to the .XML-file of the schema.
    :param str language: Language code to apply specific rules for segmenting.
    :param bool check_segments: Print segments yielded by pySBD to console.
    :return: None.
    :rtype: None

    """

    xmi_in_files = index_inception_files(dir_in)
    xmi_out_files = [Path(str(p).replace(dir_in, dir_out)) for p in xmi_in_files]

    splitter = Splitter(f_schema, language, check_segments)

    logging.info(f"Start segmenting of {len(xmi_in_files)} files.")

    for f_xmi_in, f_xmi_out in tqdm(list(zip(xmi_in_files, xmi_out_files))):
        f_xmi_out.parent.mkdir(parents=True, exist_ok=True)

        new_cas = splitter.sent_split(f_xmi_in)
        new_cas.to_xmi(f_xmi_out, pretty_print=True)

    logging.info(f"Finished auto-segmenting of sentences.")


def main(args):

    dir_in = args["--input-dir"]
    dir_out = args["--output-dir"]
    f_schema = args["--schema"]
    language = args["--lang"]
    check_segments = args["--check_segments"]

    f_log = args["--log"]

    logging.basicConfig(
        filename=f_log,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    sentence_segmenting_batch(dir_in, dir_out, f_schema, language, check_segments)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
