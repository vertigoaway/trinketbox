import torch
import numpy as np
import numpy.typing as npt

from griot import char
from griot import word
from griot import numpyWord
from typing import cast
def logitsToId(rawLogits:torch.LongTensor | torch.Tensor,
               timeSteps : int,
               batchSize: int, 
               k : int = 7
               ) -> torch.Tensor: 
    """Converts raw logits to token indices by sampling top K tokens per time step.
    Args:
        rawLogits: (batchSize, timeSteps, vocLen).
        timeSteps: Number of time steps in the output sequence.
        batchSize: Number of batches.
        k: Number of tokens to sample"""
    chosenId = np.zeros(shape=(batchSize,timeSteps))
    for batch in range(batchSize):
        for stamp in range(timeSteps):
            idVals,tokenIds = torch.topk(rawLogits[batch][stamp],k=k,dim=-1)
            
            npIdVals  : np.ndarray = idVals.cpu().detach().numpy()
            npTokenIds : np.ndarray = tokenIds.cpu().detach().numpy()
            del idVals
            del tokenIds
            try:
                chosenId[batch][stamp] = np.random.choice(npTokenIds, size=1, p=npIdVals/npIdVals.sum())
            except ValueError: # can fail if negative prob is given
                chosenId[batch][stamp] = npTokenIds[0] # take most likely one
    chosenIdTensored : torch.Tensor = torch.tensor(chosenId, dtype=torch.long)
    del chosenId
    #chosenId shape: (batchSize, timeSteps)
    return chosenIdTensored


def IdsToChrs(tokenIds : npt.NDArray[np.uint8 | np.uint32 | np.uint16] ,voc:char.Vocab | word.Vocab | numpyWord.StrictVocab) -> list[str]:
    """Converts token indices to characters. 
    Args: 
        voc: Dict mapping chars to indices.
        tokenIds: Shape (batchSize, timeSteps).
    """
    #in shape (batchSize, timeSteps)
    out : list[str] = []
    for b in tokenIds: # batch
        out.append(voc.detokenizeLine(b)) # pyright: ignore[reportAttributeAccessIssue]

    return out

def inferenceResponse(model,inp: str,
                      voc: char.Vocab | word.Vocab | numpyWord.StrictVocab ,
                      eosTok:int=1,outSize:int=1,device:str='cpu'
                      ) -> tuple[str,list]:
    """Generates a response from the given context.
    Args:
        model: The model to use.
        inp: context to generate a response for.
        voc: Dict mapping chars to indices
        eosTok: Token that indicates end of response
    Returns:
        String containing detokenized response"""
    context : torch.types._TensorOrTensors = torch.LongTensor(voc.tokenizeLine(inp)+[voc.eomTok[0]]) # pyright: ignore[reportAttributeAccessIssue]
    a : torch.Tensor
    b : int = -100
    outStr : str = ''
    outArr : list[int]= []
    while b!=eosTok: #a ton of casting happens due to torch using wide types. no perf impact
        a = logitsToId(model(cast(torch.Tensor,context).unsqueeze(0).to(device)),timeSteps=outSize,batchSize=1)
        context = torch.cat([cast(torch.Tensor,context[outSize:]), a.squeeze().view(1)])
        b = cast(int,a.to('cpu').view(-1)[0].item())
        outStr += cast(str,voc[b]) # pyright: ignore[reportArgumentType]
        outArr.append(b)
    return (outStr,outArr)


def basicInterface(model, voc: char.Vocab | word.StrictVocab | numpyWord.StrictVocab, memory:list[str]=[], timeSteps:int=512,filler:str='�',word:bool=True) -> None:
    if len(memory)<timeSteps:
        memory += [filler]*(timeSteps-len(memory))
    print(memory)
    cont : bool = True
    while cont:
        tmp : str = input('>>').strip().lower()
        if tmp == 'exit':
            cont = False
            continue
        memory = memory[len(tmp):] + list(tmp)
        response : str = cast(str,voc.detokenizeLine(inferenceResponse(model,str(memory),voc)[-1]))
        print(response)

        memory = memory[len(response):] + list(response) 
    return