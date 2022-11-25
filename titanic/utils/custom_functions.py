import re

def get_title(passenger:str) -> str:
    r""" Retrieve the title of a specific passanger name.

    It searches for the title in the name of a person:
    [Mrs, Mr, Miss, Master, Other] and returns the title.

    Parameters
    ----------
    passanger : string
        The name of the passanger including the title of the person.

    Returns
    -------
    string
        The title of the passanger.
    """ 
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'