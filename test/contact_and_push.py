import rtde_control

rtde_c = rtde_control.RTDEControlInterface("192.168.1.10")
speed = [0, 0, -0.100, 0, 0, 0]
rtde_c.moveUntilContact(speed)

rtde_c.stopScript()