import unittest
import trinketbox.ai.utils.charTokenizer as cT
import numpy as np
class testTokenizer(unittest.TestCase):
    def testSingleLineTokenization(self):
        
        expected = np.array([11,8,15,15,18,30,0,18,21,15,7,0,30,47,38,39,40,41,42,43,44,45,46,47,30,0,0,8,22,23,30,1],dtype=np.uint16)
        
        self.assertEqual(cT.tokenizeLine('hello World! 01234567890 TTest ',
                            tokDict=voc,
                            dType=np.uint16,nulTok=0,eosTok=1).all(),expected.all()) # pyright: ignore[reportArgumentType]
        
        
        
        return
    def testSingleLineDetokenization(self):
        input = np.array([11,8,15,15,18,30,0,18,21,15,7,0,30,47,38,39,40,41,42,43,44,45,46,47,30,0,0,8,22,23,30,1],dtype=np.uint16)
        expected = 'hello �orld� 01234567890 ��est \n'
        self.assertEqual(cT.detokenizeLine(input,nulTok=0,tokDict=voc,),expected)
    def testMultilineTokenization(self):
        #not implemented
        return
    def testMultilineDetokenization(self):
        #not implemented
        return

    def testCharVocabClassDictionarySync(self):

        return
    def testCharVocabClassTokenizationDetokenization(self):
        voc = cT.charVocab()

        voc.addCharacters(list('abcdefghijklmnopqrstuvwxyz '))

        x = voc.tokenizeLine('apple ')
        y = voc.detokenizeLine(x)
        self.assertEqual(x,[28, 13, 13, 17, 24, 2, 1])
        self.assertEqual(y,'apple \n')
        return
    def testCharVocabClassCharacterDuplication(self):

        return
if __name__ == '__main__':
    voc = {'�':0,chr(10):1,'-':2,'_':3,
                            'a':4,'b':5,'c':6,'d':7,'e':8,'f':9,'g':10,
                            'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,
                            'n':17,'o':18,'p':19,'q':20,'r':21,'s':22,
                            't':23,'u':24,'v':25,'w':26,'x':27,'y':28,
                            'z':29,' ':30,'.':31,',':32,'\'':33,'/':34,
                            '\"':35,':':36,';':37,'1':38,'2':39,'3':40,
                            '4':41,'5':42,'6':43,'7':44,'8':45,'9':46,
                            '0':47,}
    unittest.main()
