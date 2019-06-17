if __name__ == "__main__":
    
    
    
    lineRead = open("C:/Users/MinhND34/AppData/Local/Programs/Python/Python37/work/wordTest.txt", mode="r", encoding="utf-8-sig")
    
    thisList = lineRead.read()
    LIST = []
    line = thisList.split(", ")
    print(line[3])
    print(line)