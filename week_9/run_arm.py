import week_9.arm_manipulations as am
from week_9.arm import Arm
import matplotlib.pyplot as plt

# construct arm
my_arm = Arm(3, 0.01)

# move arm to desired coordinate using the move_arm() method
am.move_arm(my_arm, [0, 0], 1, 30)

# feedback
print(my_arm.joint_angle)
plt.show()
