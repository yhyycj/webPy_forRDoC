#folder
#run RDoc
import sys
from Step3_getPrediction import getPredict
import time

def runRDoc():
    ensembledRlts = getPredict(matrix_train_csv = r'.\data\Matrix_Train_328.txt', matrix_test_csv = r'.\data\Matrix_Validate_105.txt', models = ['all'], ensemble = 'vote', balance = None, toBinary = None)
    for fileName, score in ensembledRlts[['fileName', 'Score']].values:
        if score.lower() == 'absent':
            category = 0
        elif score.lower() == 'mild':
            category = 1
        elif score.lower() == 'moderate':
            category = 2
        elif score.lower() == 'severe':
            category = 3
        writeResult2File(str(category))
        print 'predict risk level for {pid}'.format(pid = str(category))
        time.sleep(1)
        
def writeResult2File(category):
    fout = open('result', 'w')
    fout.write(category)
    fout.close()

if __name__ == '__main__':
    for i in range(1000):
        runRDoc()