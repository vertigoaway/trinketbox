import csv
import numpy as np
import numpy.typing as npt
import time

## lazily tokenizes all characters from an exported discord chat csv ##
# message size is static, most uses for this are for AI/ML

# TODO:
    # 1. detect and remove links inside of contents (L:44-45)
    # 2. remove usage of unicode and implement a better system (L:59-64)
        # prob a fat dictionary
    # 3. userid blacklist (OPTIONAL)
    # 4. add unique userids to msgs (OPTIONAL)
    # 5. add multithreading (OPTIONAL)


# accepted format:
    # "AuthorID","Author","Date","Content","Attachments","Reactions"
#only uses the content and attachments :p

def charTokenize(
        readOut: list[list[str]],
        msgSize: int = 256,
        nulTok: int = 0, # when letter's id is too big
        eosTok: int = 1, # character added at end of sentence
        lnkTok: int = 2, # character added when a link is present
        dType = np.uint32 #
        ) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | npt.NDArray[np.uint32]:# if you go any higher fuck off
    MAX : int = np.iinfo(dType).max
    msg: str
    link: str
    msgarr: list[str]
    entries: npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | npt.NDArray[np.uint32] 

    ct: int = 0
    x: float = time.time()

    for row in readOut:
        ct += 1 # count rows to statically allocate numpy array
    entries = np.zeros((ct, msgSize), dtype=dType)  # where we store the output
    ct = 0 #optional, used to calc tok/s
    for row in readOut:
        msg = row[3]
        link = row[4]

        if link != "":  # theres a link in the msg!!!
            msg += chr(lnkTok) 
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
            if ord(letter) > MAX: #too big for current dtype :(
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