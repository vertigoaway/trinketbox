import csv
import numpy as np
import numpy.typing as npt
import time
from typing import Any
## lazily tokenizes all characters from an exported discord chat csv ##
# message size is static, most uses for this are for AI/ML

# TODO:
    # 1. detect and remove links inside of contents
    # 2. de-discordify discordStaticToken
    # 3. utilize __tokenizeLine
    # 4. userid blacklist (OPTIONAL)
    # 5. add unique userids to msgs (OPTIONAL)
    # 6. add multithreading (OPTIONAL)
    # 7. make pylance SHUT THE FUCK UP!!!!
    # 8. use numpy scalars?


# accepted format:
    # "AuthorID","Author","Date","Content","Attachments","Reactions"
#only uses the content and attachments :p

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

    :param readOut: Contents of a CSV discord chat export.
    :param msgSize: Size of all messages.
    :param nulTok: Replaces all characters unable to be represented.
    :param eosTok: Token at the end of each message.
    :param dType: Numpy data type of output array.
    :returns entries: [msgIndex][position]
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

    x:list[npt.NDArray[np.uint8 | np.uint32 | np.uint16]]=[]
    line:str
    for line in lines:
        x.extend(__tokenizeLine(line,nulTok,eosTok,dType,tokDict))
    y : npt.NDArray[np.uint8 | np.uint32 | np.uint16] = np.array(x) #ohhhmy god bruh shutUP!!
    return y


def __detokenizeLine( #detokenizes a single line of any size
        msg: npt.NDArray[np.uint8 | np.uint32 | np.uint16] | list[int],
        nulTok: int = 0,
        dType = np.uint32,
        tokDict : dict[str,int] = {'a':4,'b':5,'c':6,'d':7,'e':8,'f':9,'g':10,'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,'n':17,'o':18,'p':19,'q':20,'r':21,'s':22,'t':23,'u':24,'v':25,'w':26,'x':27,'y':28,'z':29,' ':30,'.':31,',':32,'\'':33,'/':34,'\"':35,':':36,';':37,'1':38,'2':39,'3':40,'4':41,'5':42,'6':43,'7':44,'8':45,'9':46,'0':47}
        ) -> str:
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
        tmp : str = __detokenizeLine(line,dType=dType,nulTok=nulTok,tokDict=tokDict)
        if len(line)<2:
            continue
        out.append(tmp)
    return out






if __name__ == '__main__':
    file = "ai/data/data.csv"
    csvfile = open(file, "r")
    
    readout = list(csv.reader(csvfile))
    goongagas = []
    for r in readout:
        goongagas.append(r[3])
    readout = goongagas
    x = dynamicTokenize(readout)
    y = dynamicDetokenize(x)
    print(y)