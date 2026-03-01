import csv
import numpy as np
import numpy.typing as npt
from typing import Any
# TODO:
    # 1. add multithreading (OPTIONAL)


class charVocab():
    vocabDict : dict[str,int]
    tokenDict : dict[int,str]
    nulTok : tuple[int,str]
    eomTok : tuple[int,str]
    def __init__(self,nulTok=(0,'�'),eomTok=(1,'\n')) -> None:
        self.nulTok = nulTok
        self.eomTok = eomTok
        self.vocabDict = {nulTok[1]:nulTok[0],eomTok[1]:eomTok[0]}
        self.tokenDict = {nulTok[0]:nulTok[1],eomTok[0]:eomTok[1]}
        return
    def __len__(self) -> int:
        """Get the current length of vocab."""
        return len(self.vocabDict)
    def __contains__(self, item : int | str) -> bool:
        """Check if the specified token/index is set
        Args:
            item: The token/index to check"""
        if type(item) == int:
            return type(self.tokenDict.get(item))==int
        elif type(item) == str:
            return self.vocabDict.get(item)==str
        else:
            raise TypeError
    def __delitem__(self, key : int | str) -> None:
        """Deletes the specified token/index.
        Args:
            key: The token/index to delete."""
        if type(key) == int:
            x = self.tokenDict[key]
            del self.tokenDict[key]
            del self.vocabDict[x]
        elif type(key) == str:
            x = self.vocabDict[key]
            del self.vocabDict[key]
            del self.tokenDict[x]
        else:
            raise TypeError
        return
    def __getitem__(self, key: int | str) -> int | str:
        if type(key) == int:
            x = self.tokenDict.get(key)
            if x == None:
                return self.nulTok[1]
            return x
        elif type(key) == str:
            x = self.vocabDict.get(key)
            if x == None:
                return self.nulTok[0]
            return x
        else:
            raise TypeError
    def __setitem__(self, key: int | str, value: int | str) -> None:
        if type(key) == int and type(value) == str:
            self.tokenDict[key] = value
            self.vocabDict[value] = key
        elif type(key) == str and type(value) == int:
            self.tokenDict[value] = key
            self.vocabDict[key] = value
        else:
            raise TypeError
        return
    def getFreeIndices(self, ct:int) -> list[int]:
        x = len(self.tokenDict)
        out = []
        while len(out)<ct:
            if self.tokenDict.get(x) == None:
                out.append(x)
            x+=1
        return out

    def addCharacters(self, chrs : list[str]) -> None:
        indices : list[int] = self.getFreeIndices(len(chrs))
        for i in indices:
            self[i] = chrs.pop(-1)
        return
    def tokenizeLine(self,chrs:str)-> list[int]: 
        out : list[int]= []
        for c in chrs:
            out.append(self.vocabDict.get(c)) # pyright: ignore[reportArgumentType]
        out.append(self.eomTok[0])
        return out
    def tokenizeLines(self,lines:list[str]) -> list[list[int]]:
        out : list[list[int]]  = []
        for line in lines:
            out.append(self.tokenizeLine(line))
        return out
    def detokenizeLine(self,toks:list[int]) -> str:
        out : str = ''
        for tok in toks:
            out+=self.tokenDict.get(tok,self.nulTok[1])
        return out
    def detokenizeLines(self,toksList:list[list[int]]) -> list[str]:
        out : list[str] = []
        for toks in toksList:
            out.append(self.detokenizeLine(toks))
        return out

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
    msgarr: list[str] = list(msg)
    entry: npt.NDArray[np.uint8 | np.uint32 | np.uint16] 

    entry = np.zeros(len(msg)+1,dtype=dType)

    for id,letter in enumerate(msgarr):
        try:
            entry[id] = tokDict[letter]
        except KeyError:
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
        except KeyError:
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
    """Detokenizes an array of any size, splits on eosTok
     Args:
        line: Array of token indices to detokenize.
        nulTok: Replaces all characters unable to be represented.
        eosTok: Token that indicates end of message.
        lnkTok: (unimplemented)
        dType: Numpy data type of input array.
        tokDict: Dict mapping chars to token indices.
     Returns:
        List of detokenized strings, split on eosTok."""
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




