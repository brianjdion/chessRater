def main():
    # Your main code goes here
    #print("Hello, world!")
    with open('club_games_data.csv', 'r') as f:
        counter= 0 
        columns = ""
        pgn = "1."
        values ="-A"
        new_file= []
        curr = ""
        for line in f:
            if counter == 0: 
                columns = line
                #continue
            # if counter ==70 : 
            #     break
            s = line.strip()
            if line.startswith(values):
                #print(f"c {counter}: Values: {s} \n")
                #values+=25
                #new_file.append(line.strip())
                curr = line.strip()
            if line.startswith(pgn):
                #print("**************PGN notation**************")
               # print(f"c {counter}: PGN : {s} \n")
                #pgn+=25
                # print("check " + new_file[-1]+""+line)
                # print("after")
                toAdd=  curr+""+line
               # print(toAdd)
                #print("check " + curr+""+line)
                #print("after")
                #new_file.append(new_file[-1]+line)
                new_file.append(toAdd)

            counter+=1
    with open('dataset.txt','w') as w:
        w.write(columns)
        for line in new_file:
            w.write(line)


if __name__ == "__main__":
    main()