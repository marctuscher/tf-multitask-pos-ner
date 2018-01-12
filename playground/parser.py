import time
import argparse

POS, NER = 0, 1

def determine_type(name):
    filetype = name.split('.')[-1]
    if filetype == 'txt':
        return POS
    elif filetype == 'iob':
        return NER
    else:
        return -1

def parse_pos(filename):
    with open(filename, 'rb') as file:
        pass

def parse_ner(filename):
    """
    ner:
    format:
    word \t tag
    One word per line, sentences separated by newline.
    Additionally, documents are separated by a line
    -DOCSTART-	|O
    followed by an empty line.

    data is annoated in IOB-Format, I-LABEL denotes that a NE span starts B-LABEL denotes that a NE span continues, O means outside of an NE
    we have the NE types 
    PER (person)
    ORG (organization)
    LOC (location)
    MISC (misscelaneous)
    """
    with open(filename, 'rb') as file:
        line = file.readline()
        while '-DOCSTART-' in line or line == '':
            line = file.readline()
        
        
        


def main():
    parser = argparse.ArgumentParser('Parse a file and pickle it')
    parser.add_argument('filename', help='the file to parse')
    args = parser.parse_args()
    filetype = determine_type(args.filename)
    if filetype == -1:
        print('invalid filetype provided')
        quit()
    elif filetype == POS:
        parse_pos(args.filename)
    else:
        parse_ner(args.filename)

if __name__ == '__main__':
    main()