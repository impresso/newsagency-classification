"""
Retokenize UIMA CAS XMI data and save in the original format.

Usage:
    lib/retokenization.py --input-dir=<dir> --output-dir=<dir> --schema=<fpath> --log=<fpath>

Options:
    -h --help               Show this screen.
    -s --schema=<fpath>     File path to schema file (.xml format).
    -i --input-dir=<dir>    Folder with the original files (.xmi format).
    -o --output-dir=<dir>   Folder where the new files are saved (.tsv format).
    -l --log=<fpath>        Name of the log file.
"""



import sys
import logging
from pathlib import Path
from unicodedata import category

from tqdm import tqdm
from docopt import docopt
from cassis import load_cas_from_xmi, load_typesystem


sys.path.append("../")
sys.path.append("../../")




class Retokenizer:
    def __init__(self, xmi: str, xml: str):
        """ Change token annotations

        :param str xmi: path to existing xmi annotation file.
        :param str xml: path to xml schema file.
        :return: None
        :rtype: None

        """

        self.xmi = xmi
        self.xml = xml

        self.typesystem = None
        self.cas = None

        self.tokenType = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"

    def retokenize(self):
        """Retokenize the document.

        Perform splitting off of apostrophes and hyphens from tokens into new tokens
        and remove tokens with unprintable characters.
        """

        with open(self.xml, "rb") as f:
            self.typesystem = load_typesystem(f)

        with open(self.xmi, "rb") as f:
            self.cas = load_cas_from_xmi(f, typesystem=self.typesystem)

        self.split_off_apostrophes()
        self.split_off_hyphens()
        self.remove_unprintable_tokens()

    def split_off_apostrophes(self):
        """
        Split off apostrophes when glued together with token

        Example:
        2020-02-17 10:20:28,567 - root - INFO - Token 'Finanz-Budger's' was tokenized into ['Finanz-Budger', "'", 's']
        """

        cas = self.cas
        tokenType = self.tokenType

        tokens = []
        for tok in cas.select(tokenType):
            tok_text = tok.get_covered_text()
            if "'" in tok_text and len(tok_text) > 1:
                tokens += self.splitting_at_symbol(tok, "'")

        cas.add_all(tokens)

    def split_off_hyphens(self):
        """
        Split off hyphens when glued together with token

        Example:
        2020-02-17 10:20:28,568 - root - INFO - Token 'Finanz-Budger' was tokenized into ['Finanz', '-', 'Budger']
        """

        cas = self.cas
        tokenType = self.tokenType

        tokens = []
        for tok in cas.select(tokenType):
            tok_text = tok.get_covered_text()
            if "-" in tok_text and len(tok_text) > 1:
                tokens += self.splitting_at_symbol(tok, "-")

        cas.add_all(tokens)

    def remove_unprintable_tokens(self):
        """
        Remove annotations of tokens that contain non-printable characters.

        Non-printable characters are Unicode control sequences which may cause
        problems when processing further (e.g. line-feed characters).
        Non-printable characters are replaced with an underscore in the original text.

        """

        cas = self.cas
        tokenType = self.tokenType

        remove_tokens = []

        for i, tok in enumerate(cas.select(tokenType)):
            tok_text = tok.get_covered_text()
            tok_clean = "".join(ch for ch in tok_text if category(ch)[0] != "C")

            if tok_text != tok_clean:
                logging.info(
                    f"Token '{tok_text}' contains unprintable characters and its annotation will be removed from document: {self.xmi}."
                )
                remove_tokens.append(i)

        for idx in reversed(remove_tokens):
            del cas._current_view.type_index[tokenType][idx]

        text = cas.sofa_string

        # as the offset builds on all characters, they cannot be removed
        text_clean = "".join(ch if category(ch)[0] != "C" else " " for ch in text)
        cas.sofa_string = text_clean

        assert len(text) == len(text_clean)

    def splitting_at_symbol(self, tok, symbol: str):
        """Split a token into its subtokens at a particular symbol (e.g., apostroph).
        A token may yield multiple subtokens.

        :param type tok: Original Token object .
        :param str symbol: Symbol at which token get splitted into subtokens.
        :return: Atomic subtokens equivalent to token when concatenated.
        :rtype: List of Token objects

        """

        Token = self.typesystem.get_type(self.tokenType)

        tokens = []
        tok_text = tok.get_covered_text()
        tok_splits = tok_text.split(symbol)

        # insert splitting symbol at every second position
        i = 1
        while i < len(tok_splits):
            tok_splits.insert(i, symbol)
            i += 2

        # consider the empty string when the splitting symbols
        # occurs at the end of a token (e.g. Alex')
        if not tok_splits[-1]:
            del tok_splits[-1]

        # redefine the boundary of the first token up to the first split
        split_pos = len(tok_splits[0])
        tok.end = tok.begin + split_pos

        # add new segments for remaining tokens
        # apostrophes are also tokens
        for split in tok_splits[1:]:
            start = tok.begin + split_pos
            end = tok.begin + split_pos + len(split)
            tokens.append(Token(begin=start, end=end))
            split_pos += len(split)

        try:
            assert tok_text == "".join(tok_splits)
            logging.info(
                f"Token '{tok_text}' was tokenized into {tok_splits} in document: {self.xmi}"
            )
        except AssertionError:
            logging.error(
                f"Token '{tok_text}' couldn't be properly tokenized into {tok_splits} in document: {self.xmi}"
            )

        return tokens


def index_inception_files(dir_data, suffix=".xmi") -> list:
    """Index all .xmi files in the provided directory

    :param type dir_data: Path to top-level dir.
    :param type suffix: Only consider files of this type.
    :return: List of found files.
    :rtype: list

    """

    return sorted([path for path in Path(dir_data).rglob("*" + suffix)])


def batch_retokenization(dir_in: str, dir_out: str, f_schema: str):
    """Start a batch retokenization of xmi files in the given folder .

    :param str dir_in: Top-level folder containing the .xmi-files.
    :param str dir_out: Top-level output folder for the converted documents.
    :param str f_schema: Path to the .XML-file of the schema.
    :return: None.
    :rtype: None

    """

    xmi_in_files = index_inception_files(dir_in)
    xmi_out_files = [Path(str(p).replace(dir_in, dir_out)) for p in xmi_in_files]
    import pdb; pdb.set_trace()

    msg = f"Start retokenization of {len(xmi_in_files)} files."
    logging.info(msg)
    print(msg)

    for f_xmi_in, f_xmi_out in tqdm(list(zip(xmi_in_files, xmi_out_files))):
        f_xmi_out.parent.mkdir(parents=True, exist_ok=True)
        retokenizer = Retokenizer(f_xmi_in, f_schema)
        retokenizer.retokenize()
        retokenizer.cas.to_xmi(f_xmi_out, pretty_print=True)

    logging.info(f"Retokenization completed.")


def main(args):

    dir_in = args["--input-dir"]
    dir_out = args["--output-dir"]
    f_schema = args["--schema"]
    f_log = args['--log']

    logging.basicConfig(
        filename=f_log,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    batch_retokenization(dir_in, dir_out, f_schema)


################################################################################
if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
