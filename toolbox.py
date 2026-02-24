from typing import Any


def pad(inp:list[Any],size:int,padChr:Any=0)->list[Any]:
    if len(inp)>size:
        return inp[0:size]
    else:
        inp+=[padChr]*(size-len(inp))
    return inp