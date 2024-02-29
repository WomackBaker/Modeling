import csv
input = "output.csv"
output1 = "agelow.csv"
output2 = "agemiddle.csv"
output3 = "agehigh.csv"


with open(input, 'r') as input_file, \
     open(output1, 'w', newline='') as output_file_1, \
     open(output2, 'w', newline='') as output_file_2, \
     open(output3, 'w', newline='') as output_file_3:

    csv_writer_1 = csv.writer(output_file_1)
    csv_writer_2 = csv.writer(output_file_2)
    csv_writer_3 = csv.writer(output_file_3)

    for line in input_file:
      line= line.strip()
      row = line.split(',')
      age = int(row[1])
      if age > 25:
        if age< 51:
          csv_writer_2.writerow(row)
        else:
          csv_writer_3.writerow(row)
      else:
        csv_writer_1.writerow(row)

print(f"Data extracted and saved to {output1, output2, output3}")
        