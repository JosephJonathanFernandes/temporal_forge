from nlp import parse_text
import json

def main():
    with open('sample_input.txt', 'r', encoding='utf-8') as f:
        s = f.read()
    records = parse_text(s)
    print(json.dumps(records, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
