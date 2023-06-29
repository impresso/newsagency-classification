import pandas as pd
from typing import List, Dict
import re
import os
import json



def parse_ground_truth_and_predictions(pred_lines: List[str], true_lines: List[str]):
    output = pd.DataFrame()
    pos_counter = 0
    
    for i, line in enumerate(pred_lines):
        #document is finished
        if line == '\n':
            # save length of doc
            doc_out["len_doc"] = [pos_counter] * len(doc_out["token"])
            # add data to final df
            output = pd.concat([output, pd.DataFrame(doc_out)])
        
        pos_counter += 1

        #new document
        if "# news-agency-as-source" in line:
            pos_counter = 0
            doc_out = {"token": [],
                    "pred_ag": [],
                    "true_ag": [],
                    "pos_in_doc": [],
                    "len_doc": [],
                    "LED": []}
        
        #ignore rest of comments
        elif "# " in line:
            pos_counter -= 1
            continue

        #save data if agency mentions in lines
        elif ("org.ent.pressagency" in line) or ("org.ent.pressagency" in true_lines[i]):
            pred_line = line.split("\t")
            true_line = true_lines[i].split("\t")

            #token
            doc_out["token"].append(pred_line[0])
            #fine NE
            doc_out["pred_ag"].append(pred_line[3].split(".")[-1])
            doc_out["true_ag"].append(true_line[3].split(".")[-1])
            #LED
            doc_out["LED"].append(re.findall(r"LED([0-9\.]+)", true_line[-2])[0] if "LED" in true_line[-2] else None)
            #position in file
            doc_out["pos_in_doc"].append(pos_counter)
    
    #save last entry
    doc_out["len_doc"] = [pos_counter] * len(doc_out["token"])
    output = pd.concat([output, pd.DataFrame(doc_out)])

    #LED to float
    output["LED"] = output["LED"].astype(float)
    #add information
    # about correct classification
    output["is_correct_class"] = output["pred_ag"] == output["true_ag"] 
    # about position of agency mentions
    output["at_beginning"] = output["pos_in_doc"] < 11
    output["at_end"] = (output["len_doc"] - output["pos_in_doc"]) < 11
    output["in_middle"] = ~(output["at_beginning"] | output["at_end"])

    return output


def parse_model_name(model: str):
    """  
    Takes model name of form "model_bert_base_multilingual_cased_max_sequence_length_512_epochs_3_run_multilingual_1"
    and parses it into dictionary 
    :return: dictionary with keys: dict_keys(['model', 'max_sequence_length', 'epochs', 'suffix', 'language', 'run'])
    """
    params = dict()
    params["model"] = re.findall(r"model_(.*)_max", model)[0]
    params["max_sequence_length"] = re.findall(r"max_sequence_length_([0-9]*)_epochs", model)[0]
    params["epochs"] = re.findall(r"epochs_([0-9]*)", model)[0]
    params["suffix"] = re.findall(r"epochs_[0-9](.*)", model)[0]
    
    try:
        #if suffix of form "run_de_1" or "run_multilingual-de_1"
        params["language"] = re.findall(r"run_([a-z\-]*)_", model)[0]
        params["run"] = int(re.findall(r"run_[a-z\-]*_([0-9]*)", model)[0])
    except:
        #if suffix of form "_de"
        params["language"] = re.findall(r"_([a-z\-]*)", model)[-1]
        params["run"] = None
    return params



def parse_global_data(metrics_dict: Dict[str, float], model_name: str):
    """  
    Takes a dictionary with metric scores and a model name and outputs everything as Dataframe.

    Parameters:
    :metrics_dict: dictionary with metrics, e.g. {'loss': 0.06, 'precision': 0.73, 'recall': 0.85, 'f1': 0.79}
    :model_name: str, name of the model the metrics belong to

    :return: pd.DataFrame with 1 row, with columns = ['model', 'max_sequence_length', 'epochs', 'suffix', 'language', 'run'] + 
                                                    all keys from global_dict
    """
    final_dict = parse_model_name(model_name)
    final_dict.update(metrics_dict)
    return pd.DataFrame(final_dict, index=[0])



def parse_class_report(report: str):
    """  
    Parse detailed class report

    Parameters:
    :report: class report, in the form of one string; rows split with \n

    :return: pd.DataFrame of class report    
    """
    #split data in rows
    all_rows = [entry.strip() for entry in report.split("\n")]
    rows = [entry for entry in all_rows if entry]

    #get header
    header = ["label"]
    header +=  re.findall(r"([^\s]+)", rows[0].replace("-score", ""))
    rows = rows[1:]

    #parse values per row and save them in "data"
    data = []
    for row in rows:
        row = row.replace(" avg", "_avg")
        row = row.replace("accuracy", "accuracy _ _")
        data.append(re.findall(r"([^\s]+)", row))
    return pd.DataFrame(data=data, columns=header)



def parse_all_class_reports(report_list: List[str]):
    """  
    Parse a list of class reports
    
    Parameters:
    :report_list: list of class reports, one class report as string

    :return: list of pd.DataFrame, one df per class report
    """
    all_reports = []
    for report in report_list:
        all_reports.append(parse_class_report(report))
    return all_reports


def parse_model_results(model_output_dir):
    ne_global_metrics = pd.DataFrame()
    ne_dev_metrics = dict()
    ne_test_metrics = dict()
    sent_dev_metrics = dict()
    sent_test_metrics = dict()

    for model in os.listdir(model_output_dir):
        #ignore everything which is not an experiments folder
        if not os.path.isdir(os.path.join(model_output_dir, model)):
            continue
        
        result_files = [ x for x in os.listdir(os.path.join(model_output_dir, model)) if "all_results" in x ]
        
        #if model folder exists, but no json (results) file in the folder
        if not result_files:
            print("Was not able to get results for model:", model)
            continue
        

        #for all json (results) files, parse results and save them in the dataframe/dictionaries
        for file in result_files:
            filepath = os.path.join(model_output_dir, model, file)

            modelname = model
            #if trained multilingual, but evaluated separately for fr and de, need to store language information 
            if "run_multilingual" in modelname:
                try:
                    lang = re.findall(r"all_results_([a-z]+).json", file)[0]
                except:
                    print("Was not able to infer language from files in:" , modelname)
                modelname = modelname.replace("run_multilingual", f"run_multilingual-{lang}")

            with open(filepath, "r") as f:
                model_res = json.load(f)

                #parse data and save it in different formats    
                try:
                    #if several dev results, take last one
                    dev_global = parse_global_data(model_res["dev"]["global"][-1], modelname)
                    ne_dev_metrics[modelname] = parse_class_report(model_res["dev"]["token-level"][-1])
                    sent_dev_metrics[modelname] = parse_class_report(model_res["dev"]["sent-level"][-1])
                except:
                    dev_global = parse_global_data(model_res["dev"]["global"], modelname)
                    ne_dev_metrics[modelname] = parse_class_report(model_res["dev"]["token-level"])
                    sent_dev_metrics[modelname] = parse_class_report(model_res["dev"]["sent-level"])
                dev_global.insert(4, "set", "dev") #insert column at 5th position

                test_global = parse_global_data(model_res["test"]["global"], modelname)          
                test_global["set"] = "test"
                
                ne_global_metrics = pd.concat([ne_global_metrics, dev_global, test_global], ignore_index=True)
                ne_test_metrics[modelname] = parse_class_report(model_res["test"]["token-level"])
                sent_test_metrics[modelname] = parse_class_report(model_res["test"]["sent-level"])
            
    #change str to int       
    ne_global_metrics["max_sequence_length"] = ne_global_metrics["max_sequence_length"].astype(int)
    ne_global_metrics["epochs"] = ne_global_metrics["epochs"].astype(int)

    return ne_global_metrics, ne_dev_metrics, ne_test_metrics,  sent_dev_metrics, sent_test_metrics



def import_HIPE_results(times=["TIME-ALL"], noise_levels=["LED-ALL"], set="dev", metrics=["F1_micro", "F1_macro_doc"], suffix="_time_noise", results_dir="experiments/"):
    df = pd.DataFrame()

    for model in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model)
        #exclude files and early experiments
        if not os.path.isdir(model_path):
            continue
        if not "run" in model:
            continue
        
        json_files = [ x for x in os.listdir(os.path.join(results_dir, model)) if f"pred_nerc_fine{suffix}.json" in x ]
        #continue if no nerc_fine predictions in experiments folder
        if not json_files:
            continue
        try:
            json_test_files = [x for x in json_files if set in x]
        except:
            print(f"No {set} file in {model}, but:", json_files)
            continue

        #save stats for every dev/test file (1 for de/fr, 2 for multilingual)       
        for json_test_file in json_test_files:    
            #get the data    
            with open(os.path.join(model_path, json_test_file), "r") as f:
                model_res = json.load(f)

            #collect the results
            res_dict = {"time": [], "noise_level": []}
            res_dict.update({metric: [] for metric in metrics})

            
            for time in times:
                for noise_level in noise_levels:
                    res_dict["time"].append(time)
                    res_dict["noise_level"].append(noise_level)
                    for metric in metrics:
                        try:
                            res_dict[metric].append(model_res["NE-FINE-LIT"][time][noise_level]["ALL"]["ent_type"][metric])
                        except:
                            print("Could not retrive results for model:", model)
                            if len(model_res["NE-FINE-LIT"].keys()) >1:
                                print("Keys present:", model_res["NE-FINE-LIT"].keys())
                            continue
            
            
            #parse modelname
            modelname = model
            #if trained multilingual, but evaluated separately for fr and de, need to store language information 
            if "run_multilingual" in modelname:
                try:
                    lang = re.findall(r"-([a-z]+)_pred_nerc_fine", json_test_file)[0]
                except:
                    print("Was not able to infer language from files in:" , modelname)
                modelname = modelname.replace("run_multilingual", f"run_multilingual-{lang}")
            
            modelname_df = pd.DataFrame(parse_model_name(modelname), index=[0])  
            modelname_df = pd.concat([modelname_df]*len(times)*len(noise_levels), ignore_index=True)
            
            #concat modelname and results and add dataframe to overall df
            try:
                cur_df = pd.concat([modelname_df, pd.DataFrame(res_dict)], axis=1)
                df = pd.concat([df, cur_df], ignore_index=True)
            except:
                #print("NO RESULTS", model)
                pass
    if df.empty:
        print("No data could be imported.")
        return None

    df["max_sequence_length"] = df["max_sequence_length"].astype(int)
    df["epochs"] = df["epochs"].astype(int)
    df = df.sort_values(by=["model", "max_sequence_length", "suffix"])

    return df