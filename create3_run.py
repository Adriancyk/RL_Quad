from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Create3
from irobot_edu_sdk.music import Note

robot = Create3(Bluetooth())


@event(robot.when_play)
async def draw_square(robot):

    # The "_" means that we are not using the temporal variable to get any value when iterating.
    # So the purpose of this "for" loop is just to repeat 4 times the actions inside it:
    for _ in range(1):
        await robot.arc_right(360*8, 125) # deg, cm

robot.play()