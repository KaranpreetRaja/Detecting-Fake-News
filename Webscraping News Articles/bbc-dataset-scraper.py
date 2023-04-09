import os
import csv

# Function to process each text file and return the processed content
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        content = ' '.join([line.strip() for line in lines[1:]])
    return content

# Main function to process all files and write them to a CSV file
def process_folder(folder_path, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['text', 'value'])

        for label in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label)
            if os.path.isdir(label_path):
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.txt'):
                        file_path = os.path.join(label_path, file_name)
                        content = process_file(file_path)
                        csv_writer.writerow([content, 0])

if __name__ == '__main__':
    input_folder = 'Webscraping News Articles/bbc/'  # Change this to the path of your dataset folder
    output_csv = 'output.csv'  # Change this to the desired output CSV file path
    process_folder(input_folder, output_csv)
