import csv

input = "mobile_app_user_dataset_1.csv"
output = "output.csv"

def is_empty_row(row):
    second_number = row[1].strip()
    return not second_number or not second_number.isdigit()

# Function to calculate the sum of ones and zeros in a row
def calculate_sum(row):
    row_sum = 0
    for cell in row[1:]:
        cell = cell.strip()
        if cell:
            try:
                if '.' in cell:
                    # Handle float values
                    row_sum += int(float(cell))
                else:
                    row_sum += int(cell)
            except ValueError:
                # Handle non-numeric values
                row_sum += 0
    return row_sum

# Open the input and output files
with open(input, 'r') as input_file, open(output, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file, quoting=csv.QUOTE_NONE, escapechar=' ')
    linecount = 0

    for line in input_file:
        line = line.strip()
        if linecount < 3 or not line:
            linecount += 1
        else:
            # Split the line by commas to get the row
            row = line.split(',')
            if not is_empty_row(row):
                row_sum = calculate_sum(row)
                if row_sum % 2 == 0:
                    # If the sum is even, append '0' to the end
                    rest_of_text = ",".join([row[0]] + [cell if cell.strip() else "0" for cell in row[1:]] + ['0'])
                else:
                    # If the sum is odd, append '1' to the end
                    rest_of_text = ",".join([row[0]] + [cell if cell.strip() else "0" for cell in row[1:]] + ['1'])
                csv_writer.writerow([rest_of_text])
            linecount += 1

print(f"Data extracted and saved to {output}")