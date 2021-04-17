import os
import sys
import shutil
phase = 'training'

if phase == 'training':
    end_frames=["153", "446", "232", "143", "313", "296", "269", "799", "389", "802", "293", "372", "77", "339", "105", "375", "208", "144", "338", "1058", "836"]
    sequences = [i for i  in range(21)]
    sign = "val"
else:
    end_frames=["464", "146", "242", "256", "420", "808", "113", "214", "164", "348", "1175", "773", "693", "151", 
        "849", "700", "509", "304", "179", "403", "172", "202", "435", "429", "315", "175", "169", "84", "174"]
    sequences = [i for i  in range(29)]
    sign = "test"
try:
    os.mkdir("./merged_pointrcnn")
except:
    shutil.rmtree("./merged_pointrcnn")
    os.mkdir("./merged_pointrcnn")
for seq in sequences:
    print("Processing Sequence {}".format(seq))
    car_f = open("./pointrcnn_Car_{}/".format(sign) + "%04d.txt"%(seq),'r')
    cyc_f = open("./pointrcnn_Cyclist_{}/".format(sign) + "%04d.txt"%(seq),'r')
    ped_f = open("./pointrcnn_Pedestrian_{}/".format(sign) +"%04d.txt"%(seq),'r')
    out_f = open("./merged_pointrcnn/%04d.txt"%(seq),'w')
    last_car = -1
    last_ped = -1
    last_cyc = -1
    line = ""
    for frame in range(int(end_frames[seq])+1):
        while(True):
            if last_car <= frame -1:
                car_line = car_f.readline()
                if car_line:
                    car_line = car_line.replace("\n","")
                    car_line = car_line.replace(","," ")
                    car_line = car_line.split(" ")
                    last_car = int(car_line[0])
                    # car_line.pop(0)
                    #switch alpha and score index
                    temp = car_line[6]
                    car_line[6] = car_line[-1]
                    car_line[-1] = temp
                    car_line.insert(2,car_line.pop(6))
                    car_line.insert(2,'-1')
                    car_line.insert(2,'-1')

                    car_line = " ".join(car_line)
                    if last_car == frame: 
                        line += car_line + '\n'
                        last_car -=1
            elif last_car == frame: 
                line += car_line + '\n'
                last_car -=1

            if last_cyc <= frame -1:
                cyc_line = cyc_f.readline()
                if cyc_line:
                    cyc_line = cyc_line.replace("\n","")
                    cyc_line = cyc_line.replace(","," ")
                    cyc_line = cyc_line.split(" ")
                    last_cyc = int(cyc_line[0])
                    # cyc_line.pop(0)
                    #switch alpha and score index
                    temp = cyc_line[6]
                    cyc_line[6] = cyc_line[-1]
                    cyc_line[-1] = temp
                    cyc_line.insert(2,cyc_line.pop(6))
                    cyc_line.insert(2,'-1')
                    cyc_line.insert(2,'-1')

                    cyc_line = " ".join(cyc_line)
                    if last_cyc == frame: 
                        line += cyc_line + '\n'
                        last_cyc -=1
            elif last_cyc == frame:
                line += cyc_line + '\n'
                last_cyc -=1

            if last_ped <= frame -1:
                ped_line = ped_f.readline()
                if ped_line:
                    ped_line = ped_line.replace("\n","")
                    ped_line = ped_line.replace(","," ")
                    ped_line = ped_line.split(" ")
                    last_ped = int(ped_line[0])
                    # ped_line.pop(0)
                    #switch alpha and score index
                    temp = ped_line[6]
                    ped_line[6] = ped_line[-1]
                    ped_line[-1] = temp
                    ped_line.insert(2,ped_line.pop(6))

                    ped_line.insert(2,'-1')
                    ped_line.insert(2,'-1')
                    ped_line = " ".join(ped_line)
                    if last_ped == frame: 
                        line += ped_line + '\n'
                        last_ped -= 1
            elif last_ped == frame:
                line += ped_line + '\n'
                last_ped -= 1

            if ((last_car>frame or not  car_line) and (last_cyc>frame or not cyc_line ) and (last_ped>frame or not ped_line )):
                break
    # print(line,out_f)
    out_f.write(line)
    out_f.close()
    cyc_f.close()
    ped_f.close()
    car_f.close()
            