import csv
import numpy as np
import numpy.typing as npt
import time
from typing import Any
## lazily tokenizes all characters from an exported discord chat csv ##
# message size is static, most uses for this are for AI/ML

# TODO:
    # 1. detect and remove links inside of contents (L:44-45)
    # 3. userid blacklist (OPTIONAL)
    # 4. add unique userids to msgs (OPTIONAL)
    # 5. add multithreading (OPTIONAL)


# accepted format:
    # "AuthorID","Author","Date","Content","Attachments","Reactions"
#only uses the content and attachments :p



def charTokenize(
        readOut: list[list[str]],
        msgSize: int = 256,
        nulTok: int = 0,
        eosTok: int = 1,
        lnkTok: int = 2,
        dType = np.uint32 
        ) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | npt.NDArray[np.uint32]:# if you go any higher fuck off
    """
    Parses variables from an environment file.

    :param readOut: Contents of a CSV discord chat export.
    :param msgSize: Size of all messages.
    :param nulTok: Replaces all characters unable to be represented.
    :param eosTok: Token at the end of each message.
    :param lnkTok: Token indicating a link was attached.
    :param dType: Numpy data type of output array.
    :returns entries: [msgIndex][position]
    :rtype: npt.NDArray
    """
    MAX : int = np.iinfo(dType).max
    msg: str
    link: str
    msgarr: list[str]
    entries: npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | npt.NDArray[np.uint32] 
    tokDict : dict[str,int] = {'a':4,'b':5,'c':6,'d':7,'e':8,'f':9,'g':10,'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,'n':17,'o':18,'p':19,'q':20,'r':21,'s':22,'t':23,'u':24,'v':25,'w':26,'x':27,'y':28,'z':29,' ':30,'.':31,',':32,'\'':33,'/':34,'\"':35,':':36,';':37,'1':38,'2':39,'3':40,'4':41,'5':42,'6':43,'7':44,'8':45,'9':46,'0':47}

    ct: int = 0
    x: float = time.time()

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
    print(f"time taken: {time.time()-x}s")
    print(f"token speed: {(msgSize)*ct/(time.time()-x)}/s")
    return entries



if __name__ == '__main__':
    file = "dataset/data.csv"
    csvfile = open(file, "r")
    readout = list(csv.reader(csvfile))

    charTokenize(readout)
