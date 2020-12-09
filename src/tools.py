import pickle

def save_pkl(mydict, filename='file.pkl'):
    output = open(filename, 'wb')
    pickle.dump(mydict, output)
    output.close()

def load_pkl(filename):
    file = open(filename, 'rb')
    mydict = pickle.load(file)
    file.close()
    return mydict