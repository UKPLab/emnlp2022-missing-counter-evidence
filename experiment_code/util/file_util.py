import codecs
import json
import zipfile
from typing import Dict, Iterable


def read_json(src: str) -> Dict:
    """
    Read a json file
    :param src: Path to json file
    """
    with codecs.open(src, encoding='utf-8') as f_in:
        return json.load(f_in)


def write_json(data: Dict, dest: str, pretty: bool = False) -> None:
    """
    Write a json file
    :param data: Data (dictionary) to write
    :param dest: Path of the json file.
    :param pretty: If true, json file is prettified for human readability.
    """
    with codecs.open(dest, 'w', encoding='utf-8') as f_out:
        if pretty:
            f_out.write(json.dumps(data, indent=4))
        else:
            json.dump(data, f_out)


def write_file(data: str, dest: str):
    """
    Write a file
    :param data: Textual data
    :param dest: File path.
    """
    with codecs.open(dest, 'w', encoding='utf-8') as f_out:
        f_out.write(data)


def read_file(src: str):
    """
    Read a file
    :param src: Path to file.
    """
    with codecs.open(src, encoding='utf-8') as f_in:
       return f_in.read()


def read_jsonl_lines(src: str) -> Iterable[Dict]:
    """
    Read a jsonl file
    :param src: Path to the file.
    """
    with codecs.open(src, encoding='utf-8') as f_in:
        for line in f_in.readlines():
            yield json.loads(line.strip())


def write_jsonl_lines(data: Iterable[Dict], dest: str, mode: str = 'w') -> None:
    """
    Write a jsonl file
    :param data: List of dictionaries to write
    :param dest: Path of jsonl file
    :param mode: Mode to decide whether to write ("w") or append ("a")
    """
    with codecs.open(dest, mode, encoding='utf-8') as f_out:
        for sample in data:
            f_out.write(json.dumps(sample) + '\n')


def write_text(data: str, dest: str) -> None:
    """
    Write a text file
    :param data: Text to write
    :param dest: Path of file.
    """
    with codecs.open(dest, 'w', encoding='utf-8') as f_out:
        f_out.write(data)


def append_jsonl_lines(data: Iterable[Dict], dest: str) -> None:
    """
    Append to a jsonl file
    :param data: new data
    :param dest: path to file
    """
    with codecs.open(dest, 'a', encoding='utf-8') as f_out:
        for sample in data:
            f_out.write(json.dumps(sample) + '\n')


def unzip(zip_file_path: str, destination_directory: str):
    """
    Unzip a file.
    :param zip_file_path: Path to zip file
    :param destination_directory: Path to destination directory
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_in:
        zip_in.extractall(destination_directory)
