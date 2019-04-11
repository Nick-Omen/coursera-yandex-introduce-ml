def save_answer(output_file, text):
    with open(output_file, 'w') as file:
        file.write(str(text))
        print('Output written to file `{}`'.format(output_file))
