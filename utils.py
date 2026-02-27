from typing import Any


def pad(inp:list[Any],size:int,padChr:Any=0)->list[Any]:
    if len(inp)>size:
        return inp[0:size]
    else:
        inp+=[padChr]*(size-len(inp))
    return inp

def getEnvVars(readOut : str) -> dict[str,str]:
    """
    Parses variables from an environment file.

    :param readOut: Contents of an environment file.
    :returns out: Dictionary containing extracted variable pairs
    :rtype: dict[str,str]
    """
    li : str
    out : dict[str,str] = {}
    for li in readOut.split('\n'):
        if li == '':
            continue
        varName : str
        val : str
        varName, val = li.split('=',1)
        out[varName] = val.strip("'\"\\")
    return out