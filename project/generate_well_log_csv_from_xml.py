#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:09:41 2024

@author: musthafa
"""

import xml.etree.ElementTree as ET
import pandas as pd

def parse_xml_to_dict(xml_file, csv_file):
    tree = ET.parse(xml_file)
    print(tree)
    root = tree.getroot()
    print(f"Root tag: {root.tag}")
    namespaces = {'': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
    print(f"Namespaces: {namespaces}")
    mnemonic_list_element = root.find('.//log/logData/mnemonicList', namespaces)
    if mnemonic_list_element is None:
        print("Error: mnemonicList element is missing.")
        return
    if mnemonic_list_element.text is None:
        print("Error: mnemonicList text is missing.")
        return
    headers = mnemonic_list_element.text.split(',')
    data_elements = root.findall('.//log/logData/data', namespaces)
    
    data_rows = []
    
    for data_element in data_elements:
        data_text = data_element.text.strip() if data_element.text else ""
        if data_text:
            data_values = data_text.split(',')
            data_rows.append(data_values)
            
            
    df = pd.DataFrame(data_rows, columns=headers)
    df.to_csv(csv_file, index=False)
    

xml_file = 'data.xml'
csv_file =  '/Users/musthafa/softway/DAI/spark-data-processing/csv_well_data/0010.csv'
parse_xml_to_dict("/Users/musthafa/softway/DAI/DATASET/sitecom14.statoil.no/Norway-StatoilHydro-15_$47$_9-F-5/1/log/1/1/1/00001.xml", csv_file)
    

