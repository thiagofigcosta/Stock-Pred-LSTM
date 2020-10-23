#!/bin/python3
# -*- coding: utf-8 -*-

import codecs
import re 
import os
import shutil
import getopt
import sys

TMP_FOLDER=".tmp_pythoN"
if os.name == 'nt': # verificar se eh windows
     SEPARATOR="\\"
else:
     SEPARATOR="/"

def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def read_source(path):
    work_dir_regex=re.search(r'(.*)[\/|\\]',path)
    if work_dir_regex:
        work_dir=work_dir_regex.group()
    else:
        work_dir='./'
    file = codecs.open(path, "r", "utf-8")
    file_content = file.read()
    file.close()
    return file_content,work_dir

def save_source(path,content):
    work_dir=re.search(r'(.*)[\/|\\]',path).group()
    create_folder_if_not_exists(work_dir)
    file = codecs.open(path, "w", "utf-8")
    file.write(content)
    file.close()

def remove_string_contents(str):
    return re.sub(r'([\"\'])(?:(?=(\\?))\2.)*?\1','',str)

def get_custom_imported_files(path):
    file_content,work_dir=read_source(path)
    file_content=remove_string_contents(file_content)
    libs = re.findall(r'.*from (.*?)[ .*|.*\n|.*\r]', file_content) + re.findall(r'.*import (.*?)[ .*|.*\n|.*\r]', file_content)
    ext_files=[]
    for lib in libs:
        filename=lib+".py"
        filepath=work_dir+filename
        if os.path.exists(filepath):
            ext_files.append(filepath)
    return ext_files

def _get_custom_imported_files_recursively(ext_files):
    if len(ext_files)==0:
        return ext_files
    else:
        new_ext_files=[]
        for ext_file in ext_files:
            new_ext_files.extend(_get_custom_imported_files_recursively(get_custom_imported_files(ext_file)))
        return ext_files+new_ext_files

def get_custom_imported_files_recursively(ext_file):
    return [ext_file]+_get_custom_imported_files_recursively(get_custom_imported_files(ext_file))

def replace_last(str,a,b):
    return str[::-1].replace(a,b,1)[::-1]

def replace_first(str,a,b):
    return str.replace(a,b,1)

def delete_folder(path):
    try:
        shutil.rmtree(path)
    except OSError:
        pass

def pre_compile(content):
    parsed_lines=""
    lines = content.split('\n')
    needed_tabs=0
    dictionary=False
    for line in lines:
        turn_off_dictionary=False
        line=line.strip()
        if re.match(r'^.*[ |\t]*=[ |\t]*({)\s*(.*})?$', line):
            dictionary=True
        if re.match(r'^.*}((?=(?:[^\"\']*\"\'[^\"\']*\"\')*[^\"\']*\Z)|\s*(else|elif))', line): 
            if not dictionary:
                line=re.sub('} *', '}', line)
                line=replace_last(line,'}','')
                needed_tabs-=1
            else:
                turn_off_dictionary=True
        if (needed_tabs>0):
            line="\t"*needed_tabs+line
        if not dictionary:
            if re.match(r'^.*({)\s*(#.*)?$', line):
                line=replace_last(line,'{',':')
                needed_tabs+=1
        parsed_lines+=line+'\n'
        if turn_off_dictionary:
            dictionary=False
    # print(parsed_lines)
    return parsed_lines            

def pre_compile_and_save_all_files(source_path):
    ext_files=get_custom_imported_files_recursively(source_path)
    for ext_file in ext_files:
        save_source(TMP_FOLDER+SEPARATOR+ext_file,pre_compile(read_source(ext_file)[0]))

def PythoN(python_ver,filename,args=''):
    if filename:
        pre_compile_and_save_all_files(filename)
        filename=TMP_FOLDER+SEPARATOR+filename
        full_cmd='python{} {} {}'.format(python_ver,filename,args)
    else:
        full_cmd='python{} {}'.format(python_ver,args)
    print('{}\n'.format(full_cmd))
    try:
        os.system(full_cmd)
    except Exception as e: 
        delete_folder(TMP_FOLDER)
        raise e
    finally:
        delete_folder(TMP_FOLDER)


def main(argv):
    HELP_STR=r'Pytho{N}.py [-v <value>]'
    python_ver=3
    args=[]
    try:
        opts, args = getopt.getopt(argv,"hv:",["version="])
    except getopt.GetoptError:
        print (HELP_STR)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (HELP_STR)
            sys.exit()
        elif opt in ("-v", "--version"):
            python_ver=arg
    sourcename=""
    for arg in args:
        if re.match(r'^.*\.py$', arg):
            sourcename=arg
            args.remove(arg)
            break
    PythoN(python_ver,sourcename,' '.join(args))


if __name__ == "__main__":
    main(sys.argv[1:])