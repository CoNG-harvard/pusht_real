import rtde_control
import rtde_receive
import time

if __name__ == '__main__':
    rtde_c = rtde_control.RTDEControlInterface("192.168.1.10")
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.10")
    rtde_c.setTCP()
    
    
    while True:
        print(rtde_r.getActualTCPPose())
        
        time.sleep(0.1)
        
        if 0xFF == ord('q'):
            break
        