import re


def hebrew_only(text):
    """
    This function returns a list with only the Hebrew characters and the characters '>', '<'.
    :param text: string to be filtered
    :return: string with only the Hebrew characters
    """
    hebrew_string = re.sub(r'[^א-ת><]', ' ', text)
    # split the string into a list of words with more than ten whitespaces
    hebrew_string = re.split(r'\s{10,}', hebrew_string)
    # remove empty strings from the list
    hebrew_string = list(filter(None, hebrew_string))

    return hebrew_string


if __name__ == '__main__':
    # test with txt file
    with open("C:/Users/mshil/שולחן העבודה/datathon2022/appendix/locations.txt", 'r', encoding="utf-8") as f:
        text = f.read()
    print(hebrew_only(text))