import csv

input = "age.csv"
output = "age.csv"

def is_empty_row(row):
    # Check if all fields in the row except the first one are empty (whitespace only)
    return all(cell.strip() == "" for cell in row[1:])
  
with open(input,'r') as input_file, open(output, 'w', newline='') as output_file:
  csv_writer = csv.writer(output_file, quoting=csv.QUOTE_NONE, escapechar=' ')
  
  for line in input_file:
    row = line.strip().split(',')
    if not is_empty_row(row):
      csv_writer.writerow(row)

print(f"Data extracted and saved to {output}")