import os
import pandas as pd
import random
import dataclasses
from typing import List, Dict, Optional
from os import listdir
from os.path import join, isfile


from hipe_commons.helpers.tsv import ENTITY_TYPES, COL_LABELS_V2, HipeDocument, HipeEntity, \
get_tsv_data, parse_tsv_line, TSVComment, write_tsv


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
    header = data[0][19:].split(' ') #changed this line

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


def dataframe_to_tsv(df, output_dir: str):
    """ 
    takes pandas Dataframe which contains one article, transforms it to tsv and saves it in output_dir 
    """
    rows = []
    
    # header
    #rows.append("# global.columns = TOKEN NE-COARSE-LIT NE-COARSE-METO NE-FINE-LIT NE-FINE-METO NE-FINE-COMP NE-NESTED NEL-LIT NEL-METO RENDER SEG OCR-INFO MISC"])
    rows.append(f"# language = {df.loc[0, 'language']}")
    rows.append(f"# newspaper = {df.loc[0, 'newspaper']}")
    rows.append(f"# date = {df.loc[0, 'date']}")
    rows.append(f"# document_id = {df.loc[0, 'document_id']}")
    rows.append(f"# news-agency-as-source = {df.loc[0, 'news-agency-as-source']}")

    #get values of document
    cols = ["n"] + COL_LABELS_V2 + ["segment_iiif_link"]
    iiif_data = df[cols].to_dict('tight')["data"]

    data = []
    i = 6

    #insert iiif-links
    for entry in iiif_data:
        #no new iiif-link
        if i == entry[0]:
            #append entry with tab spaces to data
            data.append("\t".join(entry[1:-1]))
            i += 1
        #new iiif-link
        else:
            data.append(f"# segment_iiif_link = {entry[-1]}")
            data.append("\t".join(entry[1:-1]))
            i += 2
    
    rows += data

    output_path = join(output_dir, df.loc[0, "document_id"] + ".tsv")
    write_tsv([rows], output_path, hipe_format_version = "v2")

    return f"document saved to {output_path}"


def contains_news_agency_name(newsag_name: str, doc_path: str) -> bool:
    """  
    Check if a document contains a certain news agency (by its WikiID)
    :return: True if newsag_name (WikiID) is in document, else return False
    """
    data = get_tsv_data(doc_path).split('\n')
    return data[5].__contains__(newsag_name)


def get_dataframes_with_newsag_name(newsag_name: str, docs_dir: str, docs_filenames: List[str]) -> List:
    """  
    :param newsag_name: newsagency WikiID which will be searched for
    :param docs_dir: directory where the tsv-docs are stored
    :param docs_filenames: list of tsv-filenames
    :return: list of pandas Dataframes, each dataframe representing one article 
                    which contains a mention of "newsag_name"
    """
    newsag_df_list = []

    for doc_path in [join(docs_dir, filename) for filename in docs_filenames]:
        #only load and store document if it contains the newsag_name tag 
        if contains_news_agency_name(newsag_name, doc_path):
            doc = tsv_to_dataframe(doc_path, keep_comments=True, hipe_format_version="v2")
            newsag_df_list.append(doc)
    
    return newsag_df_list



def get_full_mentions_and_position(newsag_name: str, doc_df) -> dict:
    """  
    :param newsag_name: name which will be searched for in the column NE-FINE-LIT (e.g. "unk")
    :param doc_df: pandas Dataframe which contains one parsed article
    :returns: dictionary of the form
            [{'name': '- e -', 'n': [21, 22, 23]},
                {'name': 'ag', 'n': [1572]},
                {'name': 'ag .', 'n': [1813, 1814]},
                {'name': 'ag .', 'n': [2101, 2102]}]
            (name: concatenated news agency token (concatenation with space);
                n: row number(s) where token occurs)
    """
    mentions = []
    mention = dict()
    mention["doc_id"] = doc_df['document_id'].values[0]

    for row in doc_df[doc_df["NE-FINE-LIT"].str.contains(newsag_name)].iterrows():
        #row[1][4]: NIL fine
        if row[1][4].startswith("B"):
            #"try" works if this is not the first row in iterrows
            try:
                mention["name"] = " ".join(mention["name"])
                mentions.append(mention)
                mention = dict()
                mention["doc_id"] = doc_df['document_id'].values[0]
            #otherwise you don't need to store anything
            except:
                pass

            mention["name"] = [row[1][1]]
            mention["n"] = [row[1][0]]
        else:
            mention["name"].append(row[1][1])
            mention["n"].append(row[1][0])

    #append last entry
    if "name" in mention.keys():
        mention["name"] = " ".join(mention["name"])
        mentions.append(mention)
    
    return mentions



def modify_NIL_fine_by_dict(substitutions: Dict[tuple, tuple], 
                            docs_dir: str, docs_filenames: str, different_out_dir: Optional[str] = None):
    """  
    Modifies all mentions specified in "substitutions" within all documents "docs_filenames" 
            in the folder "docs_dir"

    :param substitutions: of form {(token_form1, token_form2, ...) : ((old_agency_tag, new_agency_tag), 
                    {old_Wikidata_ID : new_Wikidata_ID)}
            e.g.{("ag", "ag .", "Agency", "Ag", "Ag. ") : ({"unk": "ag"}, {"unk": "unk"})} 
    """
    for ag_to_substitute, substitute_dicts in substitutions.items():
        old_ag_tag = list(substitute_dicts[0].keys())[0]

        #get dataframes of all documents which contain news agency tag "ag_tag"
        docs = get_dataframes_with_newsag_name(old_ag_tag, docs_dir, docs_filenames)
        
        for doc in docs:
            is_modified = False
            #get the tokens of all "ag_tag"
            mentions = get_full_mentions_and_position(old_ag_tag, doc)

            #use line number as index
            doc.set_index("n", inplace=True)
            for mention in mentions:
                if mention["name"] in ag_to_substitute:
                    is_modified = True

                    doc.loc[mention["n"], "NE-FINE-LIT"] = doc.loc[mention["n"], "NE-FINE-LIT"].replace(substitute_dicts[0], regex=True)
                    print(f"replaced {old_ag_tag} with {list(substitute_dicts[0].values())[0]} in {list(doc.head(1)['document_id'])[0]}")


            #save modifications
            if is_modified:
                #change metadata
                old_wiki_id, new_wiki_id = list(substitute_dicts[1].items())[0]

                newsag_metadata = list(doc.head(1)['news-agency-as-source'])[0]
                newsag_metadata = set(newsag_metadata.split(", "))
                
                #delete old wikidata ID
                if doc[doc["NE-FINE-LIT"].str.contains(old_ag_tag)].empty:
                    newsag_metadata.remove(old_wiki_id)

                #add new wikidata ID
                newsag_metadata.add(new_wiki_id)
                doc["news-agency-as-source"] = ", ".join(sorted(list(newsag_metadata)))

                #save changes
                doc.reset_index(inplace=True)
                if different_out_dir:
                    dataframe_to_tsv(doc, different_out_dir)
                else:
                    dataframe_to_tsv(doc, docs_dir)
            

def get_newsag_df(newsag: str, in_dir: str, docs_filenames: List[str]):
    """  
    takes a news agency tag and the position of tsv files and returns a dataframe with all the tokens (mentions)
    which occur in all the files
    """
    dfs = get_dataframes_with_newsag_name(newsag, in_dir, docs_filenames)
    newsag_list = []
    for df in dfs:
        newsag_list += get_full_mentions_and_position(newsag, df)

    newsag_dict = dict()
    newsag_dict["name"] = []
    newsag_dict["n"] = []
    newsag_dict["doc_id"] = []

    for dict_ in newsag_list:
        newsag_dict["name"].append(dict_["name"])
        newsag_dict["n"].append(dict_["n"])
        newsag_dict["doc_id"].append(dict_["doc_id"])
    
    return pd.DataFrame.from_dict(newsag_dict)