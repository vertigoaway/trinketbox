import csv
import numpy as np
import numpy.typing as npt
from typing import Any
# TODO:
    # 1. detect and remove links inside of contents
    # 2. de-discordify discordStaticToken
    # 3. add unique userids to msgs (OPTIONAL)
    # 4. add multithreading (OPTIONAL)
    # 5. use numpy scalars?



def discordStaticTokenize( 
        readOut: list[list[str]],
        msgSize: int = 256,
        nulTok: int = 0,
        eosTok: int = 1,
        dType = np.uint32,
        tokDict: dict[str,int] = {'a':4,'b':5,'c':6,'d':7,'e':8,'f':9,'g':10,'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,'n':17,'o':18,'p':19,'q':20,'r':21,'s':22,'t':23,'u':24,'v':25,'w':26,'x':27,'y':28,'z':29,' ':30,'.':31,',':32,'\'':33,'/':34,'\"':35,':':36,';':37,'1':38,'2':39,'3':40,'4':41,'5':42,'6':43,'7':44,'8':45,'9':46,'0':47}, 
        ) -> npt.NDArray[np.uint8 | np.uint32 | np.uint16]:# if you go any higher fuck off
    """
    Tokenizes all messages from a discord chat export with a static size
    Args:
        readOut: Contents of a CSV discord chat export.
        msgSize: Size of all messages.
        nulTok: Replaces all characters unable to be represented.
        eosTok: Token at the end of each message.
        dType: Numpy data type of output array.
        tokDict: Dictionary mapping characters to token indices.
    :returns entries: (msgIndex, position) array of tokenized messages.
    :rtype: npt.NDArray
    """
    MAX : int = np.iinfo(dType).max
    msg: str
    msgarr: list[str]
    entries: npt.NDArray[np.uint8 | np.uint32 | np.uint16] 

    ct: int = 0

    for row in readOut:
        ct += 1 # count rows to statically allocate numpy array
    entries = np.zeros((ct, msgSize), dtype=dType)  # where we store the output
    ct = 0 #optional, used to calc tok/s
    for row in readOut:
        msg = row[3]

        try:
            if msg[0:1] == ["> "]:  # checks if message is a reply
                msg = msg.split("\n", 1)[-1]  # remove quoted text, leave response
        except: #the message isnt even 2 long??
            continue
        try:
            msg = msg[0 : msgSize - 1]  # get rid of text above size
        except:
            pass
        msg += "" * (msgSize - 1 - len(msg)) #fatten up to size

        msgarr = list(msg)
        for id, letter in enumerate(msgarr):
            try:
                entries[ct][id] = tokDict[letter]
            except:
                entries[ct][id] = nulTok
                continue
        entries[ct][-1] = eosTok 


        ct += 1
    return entries

def __tokenizeLine( #tokenizes a single line of any size, makes life easier
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
        except:
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
    MAX : int = np.iinfo(dType).max
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
        x.extend(__tokenizeLine(line,nulTok,eosTok,dType,tokDict))
    y : npt.NDArray[np.uint8 | np.uint32 | np.uint16] = np.array(x) #ohhhmy god bruh shutUP!!
    return y


def __detokenizeLine( #detokenizes a single line of any size
        msg: npt.NDArray[np.uint8 | np.uint32 | np.uint16] | list[int],
        nulTok: int = 0,
        tokDict : dict[str,int] = {'a':4,'b':5,'c':6,'d':7,'e':8,'f':9,'g':10,'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,'n':17,'o':18,'p':19,'q':20,'r':21,'s':22,'t':23,'u':24,'v':25,'w':26,'x':27,'y':28,'z':29,' ':30,'.':31,',':32,'\'':33,'/':34,'\"':35,':':36,';':37,'1':38,'2':39,'3':40,'4':41,'5':42,'6':43,'7':44,'8':45,'9':46,'0':47}
        ) -> str:
    """Detokenizes a single line of token indices into a string.
    Args:
        msg: Array of token indices to detokenize.
        nulTok: Token to use for unknown characters.
        tokDict: Dict mapping chars to token indices.
        Returns:Detokenized string."""
    out : str = ''
    charDict : dict[int,str]= {v: k for k, v in tokDict.items()}
    for t in msg:
        try:
            out+=charDict[t]
        except:
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
        tmp : str = __detokenizeLine(line,nulTok=nulTok,tokDict=tokDict)
        if len(line)<2:
            continue
        out.append(tmp)
    return out






if __name__ == '__main__':
    file = "data.csv"
    csvfile = open(file, "r")
    
    readout = list(csv.reader(csvfile))
    goongagas = []
    for r in readout:
        goongagas.append(r[3])
    readout = goongagas
    x = dynamicTokenize(readout)
    y = dynamicDetokenize(x)
    print(y)