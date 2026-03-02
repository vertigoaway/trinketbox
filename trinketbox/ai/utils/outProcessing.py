import torch
import numpy as np
import numpy.typing as npt

from griot import char
from griot import word
def logitsToId(rawLogits:torch.LongTensor | torch.Tensor,
               timeSteps : int,
               batchSize: int) -> torch.Tensor: 
    """Converts raw logits to token indices by sampling top 3 tokens per time step.
    Args:
        rawLogits: (batchSize, timeSteps, vocLen).
        timeSteps: Number of time steps in the output sequence.
        batchSize: Number of batches."""
    chosenId = np.zeros(shape=(batchSize,timeSteps))
    for batch in range(batchSize):
        for stamp in range(timeSteps):
            idVals,tokenIds = torch.topk(rawLogits[batch][stamp],k=3,dim=-1)
            idVals = idVals.cpu().detach().numpy()
            tokenIds = tokenIds.cpu().detach().numpy()
            try:
                chosenId[batch][stamp] = np.random.choice(tokenIds, size=1, p=idVals/idVals.sum())
            except ValueError: # can fail if negative prob is given
                chosenId[batch][stamp] = tokenIds[0] # take most likely one
    chosenId = torch.tensor(chosenId, dtype=torch.long)
    #chosenId shape: (batchSize, timeSteps)
    return chosenId


def IdsToChrs(tokenIds : npt.NDArray[np.uint8 | np.uint32 | np.uint16] ,voc:char.Vocab | word.Vocab) -> list[str]:
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
                      voc: char.Vocab | word.Vocab ,
                      eosTok:int=1,outSize:int=1,device:str='cpu'
                      ) -> str:
    """Generates a response from the given context.
    Args:
        model: The model to use.
        inp: context to generate a response for.
        voc: Dict mapping chars to indices
        eosTok: Token that indicates end of response
    Returns:
        String containing detokenized response"""

    context : torch.types._TensorOrTensors = torch.LongTensor(voc.tokenizeLine(inp)) # pyright: ignore[reportAttributeAccessIssue]
    a = 0
    out = ''
    while a!=eosTok:
        a = logitsToId(model(context.unsqueeze(0).to(device)),timeSteps=outSize,batchSize=1)
        context = torch.cat([context[outSize:], a.squeeze().view(1)])
        a = a.to('cpu').view(-1)[0].item()
        out += voc.tokenDict[a] # pyright: ignore[reportArgumentType]
    return out


def basicInterface(model, voc: char.Vocab | word.Vocab, memory:str='', timeSteps:int=512,filler:str='�') -> None:
    if len(memory)<timeSteps:
        memory+=filler*(len(memory)-timeSteps)
    print(memory)
    cont = True
    while cont:
        tmp = input('>>').strip()
        if tmp == 'EXIT':
            cont = False
            continue
        tmp = tmp.lower()
        memory = memory[len(tmp):] + tmp
        response = inferenceResponse(model,memory,voc)
        memory = memory[len(response):] + response
        print(response)
    return None