import csv
import numpy as np
import numpy.typing as npt
from typing import Any
# TODO:
    # 1. add multithreading (OPTIONAL)


def tokenizeLine( #tokenizes a single line of any size
        msg: str,
        nulTok: int = 0,
        eosTok: int = 1,
        dType = np.uint32,
        tokDict : dict[str,int] = {'a':4,'b':5,'c':6,'d':7,'e':8,'f':9,'g':10,'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,'n':17,'o':18,'p':19,'q':20,'r':21,'s':22,'t':23,'u':24,'v':25,'w':26,'x':27,'y':28,'z':29,' ':30,'.':31,',':32,'\'':33,'/':34,'\"':35,':':36,';':37,'1':38,'2':39,'3':40,'4':41,'5':42,'6':43,'7':44,'8':45,'9':46,'0':47}
        ) -> npt.NDArray[np.uint8 | np.uint32 | np.uint16]:# if you go any higher fuck off
    """Tokenizes a single line of text into an array of token indices.
    Args:
        msg: The string to tokenize.
        nulTok: Token to use for unknown characters.
        eosTok: Token to use at the end of the message.
        dType: Numpy data type of the output array.
        tokDict: The dictionary mapping characters to token indices.
    Returns:
        An array of token indices.
    """
    msgarr: list[str]
    entry: npt.NDArray[np.uint8 | np.uint32 | np.uint16] 

    entry = np.zeros(len(msg)+1,dtype=dType)
    msgarr = list(msg)

    for id,letter in enumerate(msgarr):
        try:
            entry[id] = tokDict[letter]
        except IndexError:
            entry[id] = nulTok
    entry[-1] = eosTok
    return entry

def dynamicTokenize(
        lines: list[str],
        nulTok: int = 0,
        eosTok: int = 1,
        dType = np.uint32,
        tokDict : dict[str,int] = {'a':4,'b':5,'c':6,'d':7,'e':8,'f':9,'g':10,'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,'n':17,'o':18,'p':19,'q':20,'r':21,'s':22,'t':23,'u':24,'v':25,'w':26,'x':27,'y':28,'z':29,' ':30,'.':31,',':32,'\'':33,'/':34,'\"':35,':':36,';':37,'1':38,'2':39,'3':40,'4':41,'5':42,'6':43,'7':44,'8':45,'9':46,'0':47}
        ) -> npt.NDArray[np.uint8 | np.uint32 | np.uint16]:# if you go any higher fuck off
    """Tokenizes a list of strings into a numpy array of token indices.
    Args:
        lines: List of strings to tokenize.
        nulTok: Token to use for unknown characters.
        eosTok: Token to use at the end of each line.
        dType: Numpy data type of output array.
        tokDict: Dict mapping chars to token indices.
    Returns:
        Numpy array of token indices.
    """

    x:list[npt.NDArray[np.uint8 | np.uint32 | np.uint16]]=[]
    line:str
    for line in lines:
        x.extend(tokenizeLine(line,nulTok,eosTok,dType,tokDict))
    y : npt.NDArray[np.uint8 | np.uint32 | np.uint16] = np.array(x)
    return y


def detokenizeLine( #detokenizes a single line of any size
        msg: npt.NDArray[np.uint8 | np.uint32 | np.uint16] | list[int],
        nulTok: int = 0,
        tokDict : dict[str,int] = {'a':4,'b':5,'c':6,'d':7,'e':8,'f':9,'g':10,'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,'n':17,'o':18,'p':19,'q':20,'r':21,'s':22,'t':23,'u':24,'v':25,'w':26,'x':27,'y':28,'z':29,' ':30,'.':31,',':32,'\'':33,'/':34,'\"':35,':':36,';':37,'1':38,'2':39,'3':40,'4':41,'5':42,'6':43,'7':44,'8':45,'9':46,'0':47}
        ) -> str:
    """Detokenizes a single line of token indices into a string.
    Args:
        msg: Array of token indices to detokenize.
        nulTok: Token to use for unknown characters.
        tokDict: Dict mapping chars to token indices.
        Returns: Detokenized string."""
    out : str = ''
    charDict : dict[int,str]= {v: k for k, v in tokDict.items()}
    for t in msg:
        try:
            out+=charDict[t]
        except IndexError:
            out+=charDict[nulTok]
    return out

def dynamicDetokenize(
    line: npt.NDArray[np.uint8 | np.uint32 | np.uint16] | list[int],
    nulTok : int = 0,
    eosTok: int = 1,
    lnkTok: int=2,
    dType = np.uint32,
    tokDict : dict[str,int] = {'a':4,'b':5,'c':6,'d':7,'e':8,'f':9,'g':10,'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,'n':17,'o':18,'p':19,'q':20,'r':21,'s':22,'t':23,'u':24,'v':25,'w':26,'x':27,'y':28,'z':29,' ':30,'.':31,',':32,'\'':33,'/':34,'\"':35,':':36,';':37,'1':38,'2':39,'3':40,'4':41,'5':42,'6':43,'7':44,'8':45,'9':46,'0':47}
    ) -> list[str]:
    """Detokenizes a single line of any size, splits on eosTok
     Args:
        line: Array of token indices to detokenize.
        nulTok: Replaces all characters unable to be represented.
        eosTok: Token that indicates end of message.
        lnkTok: (unimplemented)
        dType: Numpy data type of input array.
        tokDict: Dict mapping chars to token indices.
     Returns:
        List of detokenized messages, split on eosTok."""
    plines:list[list[int]] = [[]]
    ct = 0
    out : list[str] = []
    for tok in line:
        if tok == eosTok:
            plines.append([])
            ct+=1
        else:
            plines[ct].append(tok)
    for line in plines:
        tmp : str = detokenizeLine(line,nulTok=nulTok,tokDict=tokDict)
        if len(line)<2:
            continue
        out.append(tmp)
    return out




