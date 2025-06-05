# --------------------------------------------------------------------------------
#
# WARNING: This file checks your solution and zips the files for submission.
# Please DO NOT CHANGE ANY PART of this file unless you are absolutely sure of
# the consequences and have consulted with the TA.
#
# --------------------------------------------------------------------------------


import doctest


from rich import print as rprint
from rich.console import Console
import builtins 
builtins.print = rprint
console = Console()

from individual_project import CPD

import argparse
import os
from datetime import datetime
import numpy as np
import shutil

import zipfile

P_ID = "1"
DATA_ROOT = "./data/"


def add_folder_to_zip(zipf, folder_path):
    root_len = len(os.path.dirname(os.path.abspath(folder_path)))
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            zip_path = file_path[root_len+1:]
            zipf.write(file_path, zip_path)
            print(f"Zipping {file_path} as {zip_path}...\t [green]Done.[/green]")

def collect_solution_br(c:CPD, test_br_root, test_br_t_l, save_p="./results/"):
    data_dict = {}
    for test_br_t in test_br_t_l:
        test_br_p = os.path.join(test_br_root, test_br_t)
        test_p_l = [f for f in os.listdir(test_br_p) if f.endswith(".pickle")]
        for test_p in test_p_l:
            acf, t_s, lag, ms, ms_bar, combined_acf, peak_index, br_t, average_br = c.test_br(data_r=test_br_root,
                                                                                              data_t=test_br_t,
                                                                                              data_p=test_p,
                                                                                              save_p=save_p,
                                                                                              plot=True)
            
            key_name = f"breath_{test_br_t}_{test_p.split('.')[0]}"
            data_dict[key_name] = {
                "t_s": t_s,
                "lag": lag,
                "average_br": average_br
            }
    return data_dict

def collect_solution_pr(c: CPD, test_pr_root, test_pr_t_l, test_pr_e_l, save_p="./results"):
    data_dict = {}
    for test_pr_t in test_br_t_l:
        for test_pr_e in test_pr_e_l:
            test_pr_p = os.path.join(test_pr_root, test_pr_t, test_pr_e)
            test_p_l = [f for f in os.listdir(test_pr_p) if f.endswith(".pickle")]
            acf, t_s, lag, ms, ms_bar, presence = c.test_presence(data_r=test_pr_root, 
                                                        data_t=test_pr_t, 
                                                        data_e=test_pr_e, 
                                                        data_p_list=test_p_l, 
                                                        save_p=save_p,
                                                        plot=True)
            key_name = f"presence_{test_pr_t}_{test_pr_e}"
            data_dict[key_name] = {
                "t_s": t_s,
                "lag": lag,
                "presence": presence
            }
    return data_dict
                



        
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Check the solution and zip the files.')
    parser.add_argument("--uid", type=str, help="Your Uiversity ID. e.g. 1234567")
    parser.add_argument("--pdf", type=str, default="./report.pdf", help="The path to the report file. Default is report.pdf")
    args = parser.parse_args()
    
        
    print(f"********* Collecting the solution *********")
    
    data_dict = dict().fromkeys(["br", "pr"])
    
    
    test_br_root = "./data/breath"
    test_br_t_l = ["train", "test"]
    
    test_pr_root = "./data/presence"
    test_pr_t_l = ["train", "test"]
    test_pr_e_l = ["env_0", "env_1"]
    
    cpd = CPD()
    data_dict["br"] = collect_solution_br(cpd, test_br_root, test_br_t_l)
    data_dict["pr"] = collect_solution_pr(cpd, test_pr_root, test_pr_t_l, test_pr_e_l)
    
    console.print(f"********* Zipping the files *********")
    console.print(f"Your UID is {args.uid}. Is it correct? Please enter (y/n): ", end="", style="bold blue")
    if input() == "y":
        answer_sheet_fn = f"{args.uid}_P_{P_ID}_answer-sheet.npy"
        with open(answer_sheet_fn, "wb") as f:
            print(f"Saving the answer sheet to {answer_sheet_fn}...", end="\t")
            np.save(f, data_dict)
            print(f"[green]Done.[/green]")
        
        submit_files = [
            "__init__.py",
            "individual_project.py",
            answer_sheet_fn,
            args.pdf
        ]
        submit_folder = "./results/"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        zip_folder_name = f"{args.uid}_P_{P_ID}_{timestamp}"
        zip_file_name = f"{args.uid}_P_{P_ID}_{timestamp}.zip"
       
       # cp submit_files and submit_folder to folder
        os.makedirs(zip_folder_name, exist_ok=True)
        for f in submit_files:
            try:
                shutil.copy(f, zip_folder_name)
            except FileNotFoundError:
                raise FileNotFoundError(f"Cannot find the file {f}. Please make sure the file exists.")
        
        try:
            shutil.copytree(submit_folder, os.path.join(zip_folder_name, "results"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find the folder {submit_folder}. Please make sure the folder exists.")
        
        shutil.make_archive(zip_folder_name, 'zip', zip_folder_name)
        
        print(f"[green]Please submit {zip_file_name} to Moodle[/green].")
                

