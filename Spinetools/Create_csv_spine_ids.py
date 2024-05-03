import os
import csv

def create_csv_from_folder(folder_path, csv_filename, output_directory):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Check if the CSV file already exists in the output directory
    csv_path = os.path.join(output_directory, csv_filename)
    if os.path.exists(csv_path):
        print(f"CSV file '{csv_filename}' already exists in the specified output directory.")
        return

    # Write filenames and IDs to a CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for file in files:
            file_id, filename = file.split('_', 1)  # Split the filename at the first underscore
            csv_writer.writerow([file_id, file])

    print(f"CSV file '{csv_filename}' created successfully with IDs and filenames from folder '{folder_path}' in the specified output directory.")


#########################################################################

# Example usage
folder_path = 'G:/Mit drev/Uni/6. semester/JK_bachelor/Data/Verse20_validation_unpacked_spinetools/raw'
csv_filename = 'spine_ids_verse_val.csv'
output_directory = 'G:/Mit drev/Uni/6. semester/JK_bachelor/Spinetools'

# Call the function to create the CSV file
create_csv_from_folder(folder_path, csv_filename, output_directory)
