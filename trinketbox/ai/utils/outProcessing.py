import torch
import numpy as np
import numpy.typing as npt

import charTokenizer as cT


def logitsToId(rawLogits:torch.LongTensor | torch.Tensor,timeSteps : int ,batchSize: int ,vocLen: int) -> torch.Tensor: 
    #rawLogits: (batchSize, timeSteps, vocLen)
    chosenId = np.zeros(shape=(batchSize,timeSteps))
    for batch in range(batchSize):
        for stamp in range(timeSteps):
            idVals,tokenIds = torch.topk(rawLogits[batch][stamp],k=3,dim=-1)
            idVals = idVals.cpu().detach().numpy()
            tokenIds = tokenIds.cpu().detach().numpy()
            try:
                chosenId[batch][stamp] = np.random.choice(tokenIds, size=1, p=idVals/idVals.sum())
            except: # can fail if negative prob is given
                chosenId[batch][stamp] = tokenIds[0] # take most likely one
    chosenId = torch.tensor(chosenId, dtype=torch.long)
    #chosenId shape: (batchSize, timeSteps)
    return chosenId


def IdsToChrs(tokenIds : npt.NDArray[np.uint8 | np.uint32 | np.uint16] ,voc:dict[str,int]) -> list[str]:
    cov = {i: s for s, i in voc.items()}
    #in shape (batchSize, timeSteps)
    out : list[str] = []
    for b in tokenIds: # batch
        out.append('')
        for i in b: # time step
            try:
                out[-1] += cov[int(i)]
            except:
                out[-1] += '�'
    return out

def inferenceResponse(model,inp: str,
                      voc:dict[str,int],
                      eosTok:int=1,outSize:int=1,device:str='cpu'
                      ) -> str:
    cov = {i: s for s, i in voc.items()}
    vocSize = len(voc)
    context : torch.types._TensorOrTensors = torch.LongTensor(cT.__tokenizeLine(inp,tokDict=voc))
    a = 0
    while a!=eosTok:
        a = logitsToId(model(context.unsqueeze(0).to(device)),timeSteps=outSize,batchSize=1,vocLen=vocSize)
        context = torch.cat([context[outSize:], a.squeeze().view(1)])
        a = a.to('cpu').view(-1)[0].item()
        print(cov[a],end='',flush=True) # pyright: ignore[reportArgumentType]
    out = cT.__detokenizeLine(context.cpu().numpy(),tokDict=voc)
    return out