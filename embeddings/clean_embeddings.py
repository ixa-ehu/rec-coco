with open('glove.6B.300d.csv', 'r') as f:
    lines = f.readlines()
    with open('glove.6B.300d.clean.csv', 'w') as fo:
        for line in lines:
            if line[0][0] == '"':
                pass
            else:
                fo.write(line)
    fo.close()
f.close()
            
