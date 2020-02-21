import week_9.arm_manipulations as am
from week_9.arm import Arm

my_arm = Arm(3, 1)
am.move_arm(my_arm, [0, 0], 1, 100)
print(my_arm.joint_angle)
my_arm.show()
