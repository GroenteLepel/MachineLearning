import week_9.arm_manipulations as am
from week_9.arm import Arm
import matplotlib.pyplot as plt

my_arm = Arm(3, 0.01)
am.move_arm(my_arm, [-1, -1], 1, 100)
print(my_arm.joint_angle)
plt.show()
