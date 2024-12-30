import os
import xml.etree.ElementTree as ET
import pandas as pd
import logging

# Set up logging to output information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_xml_to_dict(xml_file, output_dir):
    try:
        logging.info(f"Processing XML file: {xml_file}")
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Define namespaces
        namespaces = {'': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
        
        # Find the mnemonic list element
        mnemonic_list_element = root.find('.//log/logData/mnemonicList', namespaces)
        
        if mnemonic_list_element is None:
            logging.warning(f"mnemonicList element is missing in {xml_file}.")
            return
        
        if mnemonic_list_element.text is None:
            logging.warning(f"mnemonicList text is missing in {xml_file}.")
            return
        
        headers = mnemonic_list_element.text.split(',')
        data_elements = root.findall('.//log/logData/data', namespaces)
        
        data_rows = []
        
        for data_element in data_elements:
            data_text = data_element.text.strip() if data_element.text else ""
            if data_text:
                data_values = data_text.split(',')
                
                # Find the index of SPPA and filter rows where SPPA is empty
                if 'SPPA' in headers:
                    sppa_index = headers.index('SPPA')
                    # Only append rows where SPPA value is not empty
                    if data_values[sppa_index]:  # If SPPA value is not empty
                        data_rows.append(data_values)
        
        if data_rows:  # If there are valid rows
            # Save the CSV file with the '000' prefix in the output directory
            file_name = os.path.join(output_dir, f'000{os.path.basename(xml_file).replace(".xml", ".csv")}')
            df = pd.DataFrame(data_rows, columns=headers)
            df.to_csv(file_name, index=False)
            logging.info(f"Exported data from {xml_file} to {file_name}")
        else: 
            logging.info(f"No valid data found in {xml_file}. Skipping export.")
    
    except Exception as e:
        logging.error(f"Error processing {xml_file}: {e}")

def process_well_data_for_all_subdirectories(main_dir, output_base_dir):
    # Loop through each well directory in the main directory
    for well_dir in os.listdir(main_dir):
        well_path = os.path.join(main_dir, well_dir)
        
        # Skip non-directory files
        if not os.path.isdir(well_path):
            continue
        
        logging.info(f"Processing well directory: {well_path}")
        
        # Create output directory for the well (well directory should be the subdirectory)
        well_name = os.path.basename(well_path)
        output_dir = os.path.join(output_base_dir, well_name)
        
        # Create the output directory for the well if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory for well: {output_dir}")
        
        # Recursively go through all subdirectories and find XML files
        for root, dirs, files in os.walk(well_path):
            logging.info(f"Processing directory: {root}")
            # Look through all directories, including nested ones
            for file in files:
                if file.endswith('.xml'):
                    xml_file_path = os.path.join(root, file)
                    logging.info(f"Processing XML file: {xml_file_path}")
                    parse_xml_to_dict(xml_file_path, output_dir)

# Example usage
main_dir = '/Users/musthafa/softway/DAI/DATASET/sitecom14.statoil.no/Norway-Statoil-NO 15_$47$_9-F-7'  # Main well directory
output_base_dir = '/Users/musthafa/softway/DAI/spark-data-processing/well_data/Norway-Statoil-NO 15_$47$_9-F-7'  # Output base directory
process_well_data_for_all_subdirectories(main_dir, output_base_dir)
