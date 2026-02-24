import csv 
import numpy as np
import numpy.typing as npt
import time



# accepted format:
    # "AuthorID","Author","Date","Content","Attachments","Reactions"
#only uses the content and attachments :p

def charTokenize(
        readOut: list[list[str]],
        nulTok: int = 0, # when letter's id is too big
        eosTok: int = 1, # character added at end of sentence
        lnkTok: int = 2, # character added when a link is present
        dType = np.uint32
        ) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | npt.NDArray[np.uint32]:# if you go any higher fuck off
    MAX : int = np.iinfo(dType).max
    msg: str
    link: str
    msgarr: list[str]
    entries: npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | npt.NDArray[np.uint32] 


    ct: int = 0
    x: float = time.time()

    for row in readOut:
        ct += 1 
    entries = np.zeros((ct), dtype=dType)  
    ct = 0 
    for row in readOut:
        msg = row[3]
        link = row[4]

        if link != "":  
            msg += chr(lnkTok) 
        try:
            if msg[0:1] == ["> "]: 
                msg = msg.split("\n", 1)[-1]
        except:
            continue
        
        

        msgarr = list(msg)
        for id, letter in enumerate(msgarr):
            if ord(letter) > MAX: 
                entries[ct][id] = nulTok
                continue
            entries[ct][id] = ord(letter)
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